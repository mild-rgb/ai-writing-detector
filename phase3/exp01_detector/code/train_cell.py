#!/usr/bin/env python3
"""phase 3 exp01: fine-tune ModernBERT as a human-vs-AI detector.

One JSON record appended per run. Dispatches on the cell's `task`:
  binary  all seven generators seen in training; scored PER GENERATOR
  logo    one generator held out whole (its questions removed from train AND
          val, both the AI answer and its paired human answer), then scored on
          exactly those questions. The real generalization test.

THE REPRODUCIBILITY CONTRACT. Phase 2's exp01 captured no environment at all and
called it the cheapest gap it could have closed. Every record written here
carries, self-contained:
  * the full `pip freeze`, torch/transformers/CUDA/cuDNN versions, GPU model and
    driver version, and the numerics-relevant torch flags   (`env`)
  * both seeds SEPARATELY -- weight init and data order      (`seeds`)
  * every hyperparameter and the exact recipe                (`hp`)
  * sha256 of docs.jsonl and splits.json, plus a hash of this cell's own id
    lists, so the inputs are pinned                          (`data`)
  * the code path, gradient checkpointing explicitly among it (`code_path`)
  * per-document probabilities, for test AND val             (`probs`)

Per-document probabilities are a HARD RULE, not a nicety. In phase 2 they made
the entire analysis re-derivable by a second party with no GPU, and that is how
six errors were caught.

WHAT THIS DOES NOT PROMISE. Not bitwise determinism, and unavoidable
floating-point nondeterminism is not a failure. Phase 2 measured the same seed
with gradient checkpointing toggled moving accuracy 0.9347 -> 0.9034 purely
through FP non-associativity; a different GPU or CUDA version does the same. The
standard is that CONCLUSIONS reproduce -- ordinal and directional findings across
seeds, reported as mean +/- SD -- not that a single number reappears. Pin and log
everything specifiable, then expect wobble and say so.
"""
import argparse, dataclasses, hashlib, json, math, os, random, shutil, sys, time
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import auc, confusion                               # noqa: E402
from report import (run_key, binary_report, best_threshold,      # noqa: E402
                    latest_checkpoint)
import env_capture                                               # noqa: E402

# ---- pre-registered recipe -------------------------------------------------
# Fixed in advance. NOT tuned against test. `lr` is swept down only if training
# diverges, and any such change is recorded in the run's `hp` block, which makes
# the deviation visible rather than silent.
HP = dict(
    lr=3e-5,                    # large model; drop to 2e-5/1e-5 only on divergence
    batch=8,
    grad_accum=1,
    max_epochs=3,               # 2-3; best epoch chosen on val, not test
    warmup_ratio=0.1,
    weight_decay=8e-6,          # ModernBERT's own recipe, not BERT's 0.01
    # Early stopping is OFF by default, which is what `patience=3` MEANT under
    # phase 2's epoch cadence (patience >= max_epochs => best-epoch selection,
    # never an actual stop). Under a step cadence the same 3 would mean 3 evals
    # = 300 steps, which is early stopping by accident rather than by design --
    # a silent recipe change hiding inside a storage change. So the unit is
    # named, and None means "never stop early, just select the best step".
    early_stop_patience_evals=None,
    # Which dev metric picks the best checkpoint. 'loss' by default: dev AUC
    # SATURATES at 1.000 on this corpus, and once it ties, "best" is decided by
    # whichever step first touched the ceiling -- which is downstream of
    # floating-point noise. Measured directly: the same seed on an L4 and an
    # A100 topped out at step 300 and step 200 respectively, and the resulting
    # balanced accuracy differed by 2.4 points while AUC moved in the 6th
    # decimal. Cross-entropy does not saturate, so the choice stays meaningful
    # and favours the better-calibrated checkpoint. AUC is still the primary
    # REPORTED metric; this is only what selects.
    select_metric='loss',       # 'loss' (lower better) or 'auc' (higher better)
    max_length=2048,            # corpus max is ~1800 tok; chosen to avoid truncation
    decision_threshold=0.5,     # pre-declared; a dev-chosen threshold is ALSO reported
    label_smoothing=0.0,
    lr_scheduler='linear',
    optim='adamw_torch',
    # Checkpoint/eval cadence. In `hp` and therefore in the run key, because it
    # sets how often the dev set is scored and so which step gets selected as
    # best -- it is part of the recipe, not a storage detail. save_total_limit
    # and mirroring are NOT here: they change disk, never a number.
    ckpt_strategy='steps',      # 'steps' for warm restarts; 'epoch' for phase-2 parity
    ckpt_steps=100,             # ~1/5 epoch at 3.7k docs, batch 8
)

# MODEL_KW (sdpa, reference_compile=False) lives in report.py so the run key and
# the model construction cannot drift apart.
from report import MODEL_KW  # noqa: E402,F811


def _json_safe(o):
    """Last-resort coercion for anything that reached the record still wearing a
    numpy type. Never silently drops a value -- unknown objects still raise."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'unserialisable {type(o).__name__} in run record: {o!r}')


class DS(Dataset):
    """Pre-tokenised. Records how many documents the max_length cap truncated,
    because a truncation rate that changes between runs changes the task."""
    def __init__(self, docs, ids, field, tok, maxlen, lab_of):
        texts = [docs[i][field] for i in ids]
        full = tok(texts, truncation=False)['input_ids']
        self.lengths = [len(x) for x in full]
        self.n_truncated = sum(1 for n in self.lengths if n > maxlen)
        self.e = tok(texts, truncation=True, max_length=maxlen, padding=False)
        self.y = [lab_of(docs[i]) for i in ids]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {'input_ids': self.e['input_ids'][i],
                'attention_mask': self.e['attention_mask'][i], 'labels': self.y[i]}

    def stats(self):
        L = sorted(self.lengths)
        return {'n': len(L), 'tokens_median': L[len(L)//2], 'tokens_max': L[-1],
                'tokens_p99': L[min(len(L)-1, int(0.99*len(L)))],
                'n_truncated': self.n_truncated,
                'truncated_frac': round(self.n_truncated/len(L), 5)}


def run(cell, init_seed, data_seed, docs, model_name, pooling, grad_ckpt,
        outpath, workdir, env, data_ident, hp, save_total_limit=2,
        keep_checkpoints=False, final_model_dir=None):
    # The run key is computed FIRST: it names the checkpoint directory, so a
    # warm restart can only ever resume a run with byte-identical identity --
    # same cell, both seeds, model, pooling, gradient checkpointing, every
    # hyperparameter and the train/test id hashes. A leftover checkpoint from a
    # different recipe cannot be picked up by accident.
    key, key_hash = run_key(cell, init_seed, data_seed, model_name, pooling, grad_ckpt, hp)
    ckpt_dir = os.path.join(workdir, key_hash)
    field = cell['text_field']
    classes = cell['classes']
    cidx = {c: i for i, c in enumerate(classes)}
    lk = cell['label_key']
    lab_of = lambda d: cidx[d[lk]]                                  # noqa: E731
    K = len(classes)

    tr_ids, va_ids, te_ids = cell['train_ids'], cell['val_ids'], cell['test_ids']
    # re-assert the split invariants HERE too: prep.py checked them when the file
    # was written, this checks the file that was actually loaded
    qtr = {docs[i]['q_id'] for i in tr_ids}
    assert not (qtr & {docs[i]['q_id'] for i in te_ids}), 'TRAIN/TEST QUESTION LEAK'
    assert not (qtr & {docs[i]['q_id'] for i in va_ids}), 'TRAIN/VAL QUESTION LEAK'
    if cell['heldout_generator']:
        g = cell['heldout_generator']
        assert not any(docs[i]['generator'] == g for i in tr_ids + va_ids), \
            'HELD-OUT GENERATOR PRESENT IN TRAIN/VAL'

    tok = AutoTokenizer.from_pretrained(model_name)

    # --- seeds, kept separate ------------------------------------------------
    # init_seed governs classifier-head initialisation; it is set immediately
    # before from_pretrained so nothing else consumes the stream. data_seed
    # governs shuffling and data order, and is handed to the Trainer alone.
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)
    random.seed(init_seed)
    # `classifier_pooling` and `reference_compile` are CONFIG fields, not
    # from_pretrained kwargs -- transformers 5.15.1 forwards unrecognised kwargs
    # straight into the model __init__ and raises. Set them on the config, then
    # read them back off the loaded model so the record states what the model
    # ACTUALLY has rather than what was asked for. Asking and getting are not the
    # same thing, and the record is only worth anything if it reports the latter.
    cfg = AutoConfig.from_pretrained(model_name, num_labels=K)
    cfg.classifier_pooling = pooling
    cfg.reference_compile = MODEL_KW['reference_compile']
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=cfg, attn_implementation=MODEL_KW['attn_implementation'])

    effective = {
        'attn_implementation': getattr(model.config, '_attn_implementation',
                                       MODEL_KW['attn_implementation']),
        'reference_compile': getattr(model.config, 'reference_compile', None),
        'classifier_pooling': getattr(model.config, 'classifier_pooling', None),
        'torch_dtype': str(next(model.parameters()).dtype)}
    assert effective['classifier_pooling'] == pooling, \
        f"classifier_pooling did not take: asked {pooling}, got {effective['classifier_pooling']}"
    assert effective['attn_implementation'] == 'sdpa', \
        f"attn_implementation did not take: got {effective['attn_implementation']} -- "\
        f"flash_attention_2 has a known NaN bug and must not be silently substituted"
    assert effective['reference_compile'] is False, \
        f"reference_compile did not take: got {effective['reference_compile']}"

    def compute_metrics(p):
        pr = torch.softmax(torch.tensor(p.predictions), -1).numpy()
        lab = ['ai' if y == cidx['ai'] else 'human' for y in p.label_ids]
        return {'sel': auc(list(pr[:, cidx['ai']]), lab)}

    steps_per_epoch = math.ceil(len(tr_ids) / (hp['batch'] * hp['grad_accum']))
    warmup_steps = round(hp['warmup_ratio'] * steps_per_epoch * hp['max_epochs'])

    # Full checkpoints -- model, optimizer, scheduler and RNG state -- so a
    # restart WARM-starts mid-run rather than beginning the epoch again. HF
    # writes optimizer.pt / scheduler.pt / rng_state.pth alongside the weights;
    # this is deliberately not a weights-only save.
    by_steps = hp['ckpt_strategy'] == 'steps'
    total_steps = math.ceil(steps_per_epoch * hp['max_epochs'])
    total_evals = (math.ceil(total_steps / hp['ckpt_steps']) if by_steps
                   else hp['max_epochs'])
    # None => a patience the run cannot reach, i.e. pure best-checkpoint
    # selection with no early stop.
    patience_evals = hp['early_stop_patience_evals'] or (total_evals + 1)
    if hp['select_metric'] == 'loss':
        sel_metric, sel_greater = 'eval_loss', False
    elif hp['select_metric'] == 'auc':
        sel_metric, sel_greater = 'sel', True
    else:
        raise SystemExit(f"unknown select_metric {hp['select_metric']!r}")
    cadence = ({'eval_strategy': 'steps', 'save_strategy': 'steps',
                'eval_steps': hp['ckpt_steps'], 'save_steps': hp['ckpt_steps']}
               if by_steps else
               {'eval_strategy': 'epoch', 'save_strategy': 'epoch'})

    ta = dict(
        output_dir=ckpt_dir, learning_rate=hp['lr'],
        per_device_train_batch_size=hp['batch'],
        gradient_accumulation_steps=hp['grad_accum'],
        per_device_eval_batch_size=16, num_train_epochs=hp['max_epochs'],
        warmup_steps=warmup_steps, weight_decay=hp['weight_decay'],
        lr_scheduler_type=hp['lr_scheduler'], optim=hp['optim'],
        label_smoothing_factor=hp['label_smoothing'],
        save_total_limit=save_total_limit, **cadence,
        load_best_model_at_end=True,
        metric_for_best_model=sel_metric, greater_is_better=sel_greater,
        logging_strategy='epoch', report_to=[],
        seed=init_seed, data_seed=data_seed,
        bf16=True, fp16=False,                 # bf16, never fp16
        disable_tqdm=True,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={'use_reentrant': False} if grad_ckpt else None)

    # Fail fast and by name on transformers version drift. An unknown keyword
    # otherwise surfaces as a TypeError from inside __init__ partway through a
    # sweep; worse, silently DROPPING one would quietly change the recipe (bf16,
    # a seed, the checkpoint cadence) with nothing in the record to show it. So:
    # name the offenders, refuse to run, never drop.
    known = {f.name for f in dataclasses.fields(TrainingArguments)}
    unknown = sorted(set(ta) - known)
    if unknown:
        raise SystemExit(
            f'TrainingArguments in transformers {transformers.__version__} does not '
            f'accept: {unknown}. Do not simply delete them -- decide what each one '
            f'was doing and whether the recipe still holds without it.')
    args = TrainingArguments(**ta)

    ds_tr = DS(docs, tr_ids, field, tok, hp['max_length'], lab_of)
    ds_va = DS(docs, va_ids, field, tok, hp['max_length'], lab_of)
    ds_te = DS(docs, te_ids, field, tok, hp['max_length'], lab_of)

    trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                      processing_class=tok, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=patience_evals)])
    resume = latest_checkpoint(ckpt_dir)
    if resume:
        print(f'  WARM START from {os.path.basename(resume)}', flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    train_s = time.time() - t0

    def predict(ds, ids):
        pr = trainer.predict(ds)
        P = torch.softmax(torch.tensor(pr.predictions), -1).numpy()
        return P, [float(x) for x in P[:, cidx['ai']]], [docs[i]['label'] for i in ids]

    Pva, pva, lva = predict(ds_va, va_ids)
    Pte, pte, lte = predict(ds_te, te_ids)                 # test: touched once
    t_dev, ba_dev = best_threshold(pva, lva)

    rec = {
        'run_key_hash': key_hash, 'run_key': key,
        'task': cell['task'], 'cell': cell['cell'],
        'text': 'raw' if field == 'text' else 'norm', 'text_field': field,
        'heldout_generator': cell['heldout_generator'], 'classes': classes,
        'model': model_name,
        'seeds': {'weight_init': init_seed, 'data_order': data_seed,
                  'note': 'init seeds torch/numpy/random before from_pretrained; '
                          'data_order is TrainingArguments.data_seed (sampler/shuffle). '
                          'Validation is the corpus dev split, so no carve seed exists.'},
        'hp': hp, 'model_kw': MODEL_KW,
        'code_path': {
            'gradient_checkpointing': grad_ckpt,
            'gradient_checkpointing_use_reentrant': False if grad_ckpt else None,
            'attn_implementation': effective['attn_implementation'],
            'reference_compile': effective['reference_compile'],
            'classifier_pooling': effective['classifier_pooling'],
            'requested_model_kw': MODEL_KW,
            'effective_model_config': effective,
            'precision': 'bf16', 'val_source': 'corpus_dev_split',
            'trainer_class': 'transformers.Trainer',
            'loss': 'default_cross_entropy_no_class_weights',
            'early_stopping': {
                'patience_evals': patience_evals,
                'total_evals_possible': total_evals,
                'can_fire': bool(hp['early_stop_patience_evals']),
                'unit': 'evaluations' if by_steps else 'epochs',
                'note': 'patience is counted in EVALUATIONS. Under a step cadence '
                        'that is not the same as epochs, and conflating them turns '
                        'best-checkpoint selection into an accidental early stop.'},
            'checkpoint_selection': {'metric': sel_metric,
                                     'greater_is_better': sel_greater,
                                     'source': 'corpus dev split'},
            'checkpointing': {'strategy': hp['ckpt_strategy'],
                              'every_steps': hp['ckpt_steps'] if by_steps else None,
                              'full_state': True, 'save_total_limit': save_total_limit,
                              'dir': ckpt_dir},
            'resumed_from_checkpoint': os.path.basename(resume) if resume else None,
            'resume_note': ('a warm-started run restores optimizer, scheduler and RNG '
                            'state, but its FP trajectory still need not match an '
                            'uninterrupted run bit for bit -- recorded so a reader can '
                            'tell the two apart'),
            'note': 'gradient checkpointing is recorded because phase 2 measured it '
                    'moving accuracy 0.9347 -> 0.9034 at identical seed and data.'},
        'data': data_ident | {'cell_id_hashes': cell['id_hashes']},
        'env': env,
        'tokenisation': {'train': ds_tr.stats(), 'val': ds_va.stats(), 'test': ds_te.stats()},
        'n': {'train': len(tr_ids), 'val': len(va_ids), 'test': len(te_ids)},
        # what SELECTED the checkpoint, named, alongside what the selected
        # checkpoint actually scores -- conflating the two is how a saturating
        # metric hides behind a number that looks like a result
        'val_best_metric': trainer.state.best_metric,
        'val_best_metric_name': sel_metric,
        'val_selection_note': ('AUC is the primary reported metric; this is only '
                               'what chose the checkpoint. The selected checkpoint\'s '
                               'own dev scores are in `val`.'),
        'epochs_run': trainer.state.epoch,
        'global_step': trainer.state.global_step,
        'train_seconds': round(train_s, 1),
        'seconds_per_step': (round(train_s / trainer.state.global_step, 3)
                             if trainer.state.global_step else None),
        'log_history': trainer.state.log_history,
        'threshold_dev_chosen': t_dev, 'balanced_acc_dev_at_dev_threshold': ba_dev,
        'val': binary_report(pva, lva, hp['decision_threshold']),
        'test': binary_report(pte, lte, hp['decision_threshold']),
        'test_at_dev_threshold': binary_report(pte, lte, t_dev),
        'confusion': confusion([int(s > hp['decision_threshold']) for s in pte],
                               [cidx[l] for l in lte], K),
    }

    # --- per generator, ALWAYS. Never a pooled rate ---------------------------
    # Detection ranges 7-80% across these models; a pooled number averages things
    # that are not alike. Each generator is scored against the SAME human
    # documents, so the human side is shared and only the AI side varies.
    gens = sorted({docs[i]['generator'] for i in te_ids if docs[i]['label'] == 'ai'})
    hu = [(s, 'human') for s, i in zip(pte, te_ids) if docs[i]['label'] == 'human']
    per_gen = {}
    for g in gens:
        sub = [(s, 'ai') for s, i in zip(pte, te_ids) if docs[i]['generator'] == g] + hu
        per_gen[g] = binary_report([s for s, _ in sub], [l for _, l in sub],
                                   hp['decision_threshold'])
        per_gen[g]['recall_at_dev_threshold'] = (
            sum(1 for s, i in zip(pte, te_ids) if docs[i]['generator'] == g and s > t_dev)
            / max(1, sum(1 for i in te_ids if docs[i]['generator'] == g)))
    rec['per_generator'] = per_gen
    rec['per_generator_note'] = ('AI side varies by generator; the human side is the '
                                 'same document set in every row. Report the spread, '
                                 'never the pooled rate.')

    # --- whitespace control ---------------------------------------------------
    # Newline count alone as a detector on this exact test set. Under `norm` it is
    # 0.500 by construction (no newlines survive), which proves any `norm` result
    # is not whitespace. Costs nothing and needs no GPU.
    nl = [float(docs[i][field].count('\n')) for i in te_ids]
    rec['newline_baseline_auc'] = auc(nl, lte)
    rec['newline_baseline_note'] = ('newline count alone as a detector, on this exact '
                                    'test set. Under `norm` it is 0.500 by construction '
                                    'because no newline survives, which is the proof '
                                    'that a `norm` result is not whitespace.')

    # --- per-document probabilities: the hard rule ---------------------------
    rec['probs'] = {i: [float(x) for x in Pte[j]] for j, i in enumerate(te_ids)}
    rec['val_probs'] = {i: [float(x) for x in Pva[j]] for j, i in enumerate(va_ids)}

    # Save the model BEFORE writing the record, so the record can name where it
    # went. Written the other way round first, and `final_model_dir` came out
    # null in every record -- recoverable only because the directory is named by
    # run_key_hash, which is luck rather than design.
    if final_model_dir:
        dest = os.path.join(final_model_dir, key_hash)
        trainer.save_model(dest)
        tok.save_pretrained(dest)
        with open(os.path.join(dest, 'run_record_meta.json'), 'w') as fh:
            json.dump({k: rec[k] for k in ('run_key_hash', 'run_key', 'cell', 'classes',
                                           'seeds', 'hp', 'code_path', 'data', 'test')},
                      fh, indent=1, default=_json_safe)
        rec['final_model_dir'] = dest
        print(f'  saved final model -> {dest}', flush=True)

    with open(outpath, 'a') as fh:
        # default= is a backstop, not the fix: numpy scalars are cast where they
        # are produced. It is here because losing a finished training run to a
        # serialisation error is the most expensive possible way to find one.
        fh.write(json.dumps(rec, default=_json_safe) + '\n')

    T = rec['test']
    spread = (f"{min(v['recall_ai'] for v in per_gen.values()):.2f}-"
              f"{max(v['recall_ai'] for v in per_gen.values()):.2f}" if per_gen else '-')
    print(f"DONE {rec['cell']:32s} init={init_seed} data={data_seed} pool={pooling} "
          f"AUC={T['auc']:.4f} BA={T['balanced_acc']:.4f} recall/gen[{spread}] "
          f"devAUC={rec['val']['auc']:.4f} ({sel_metric}={rec['val_best_metric']:.4f}) "
          f"ep={rec['epochs_run']:.1f} {train_s:.0f}s",
          flush=True)
    del model, trainer
    torch.cuda.empty_cache()
    if not keep_checkpoints:
        # the record is the artifact; the checkpoint was only insurance against a
        # mid-run death, and that risk is over
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', default='/content/exp01/docs.jsonl')
    ap.add_argument('--splits', default='/content/exp01/splits.json')
    ap.add_argument('--card', default='/content/exp01/dataset_card.json')
    ap.add_argument('--out', default='/content/exp01/results.jsonl')
    ap.add_argument('--envdir', default='/content/exp01/env')
    ap.add_argument('--workdir', default='/content/work')
    ap.add_argument('--model', default='answerdotai/ModernBERT-large')
    ap.add_argument('--init-seeds', default='0,1,2,3,4',
                    help='weight-init seeds; >=5 -- three cannot support a variance claim')
    ap.add_argument('--data-seeds', default='',
                    help='data-order seeds, paired positionally with --init-seeds. '
                         'Default: mirror the init seeds.')
    ap.add_argument('--pooling', default='cls', choices=['cls', 'mean'],
                    help='ModernBERT classifier_pooling; test both, our docs are long')
    ap.add_argument('--tasks', default='binary,logo')
    ap.add_argument('--cells', default='', help='comma-separated exact cell names')
    ap.add_argument('--text', default='norm,raw', help='which text fields to run')
    ap.add_argument('--no-grad-ckpt', action='store_true',
                    help='disable gradient checkpointing. CHANGES RESULTS -- recorded.')
    ap.add_argument('--lr', type=float, default=None, help='override lr (recorded in hp)')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--max-length', type=int, default=None)
    ap.add_argument('--ckpt-strategy', default=None, choices=['steps', 'epoch'],
                    help="checkpoint/eval cadence. 'steps' warm-starts finely; "
                         "'epoch' reproduces phase 2's cadence. Recorded in hp.")
    ap.add_argument('--ckpt-steps', type=int, default=None,
                    help='steps between full checkpoints when --ckpt-strategy=steps')
    ap.add_argument('--select-metric', default=None, choices=['loss', 'auc'],
                    help="dev metric that picks the best checkpoint. 'loss' (default) "
                         "because dev AUC saturates and ties are broken by FP noise.")
    ap.add_argument('--early-stop-patience-evals', type=int, default=None,
                    help='stop after N consecutive EVALUATIONS without improvement. '
                         'Omitted = never stop early, just select the best checkpoint '
                         '(the pre-registered behaviour). Recorded in hp either way.')
    ap.add_argument('--save-total-limit', type=int, default=2,
                    help='full checkpoints kept on disk per run (the best one is '
                         'always protected). Affects disk, never a number.')
    ap.add_argument('--keep-checkpoints', action='store_true',
                    help='do not delete a run\'s checkpoints once it has finished')
    ap.add_argument('--final-model-dir', default=None,
                    help='persist each finished model here, keyed by run hash. Needed '
                         'if a run is to be reused later for external evaluation.')
    ap.add_argument('--limit', type=int, default=0, help='stop after N runs')
    ap.add_argument('--dry-run', action='store_true', help='list runs, train nothing')
    a = ap.parse_args()

    hp = dict(HP)
    for k, v in (('lr', a.lr), ('max_epochs', a.epochs), ('max_length', a.max_length),
                 ('ckpt_strategy', a.ckpt_strategy), ('ckpt_steps', a.ckpt_steps),
                 ('early_stop_patience_evals', a.early_stop_patience_evals),
                 ('select_metric', a.select_metric)):
        if v is not None:
            hp[k] = v
    grad_ckpt = not a.no_grad_ckpt

    docs = {d['doc_id']: d for d in map(json.loads, open(a.docs))}
    S = json.load(open(a.splits))
    card = json.load(open(a.card))
    data_ident = {
        'docs_sha256': card['docs_sha256'], 'splits_sha256': card['splits_sha256'],
        'docs_path': os.path.abspath(a.docs), 'splits_path': os.path.abspath(a.splits),
        'n_docs': card['n_docs'], 'n_questions': card['n_questions'],
        'corpus_complete': card['complete'],
        'corpus_dataset_sha256_short': card['corpus_dataset_sha256_short'],
        'corpus_split_files_sha256_short': card['corpus_split_files_sha256_short'],
        'generator_counts': card['generator_counts']}
    # the loaded file must BE the file the card describes
    live = hashlib.sha256(open(a.docs, 'rb').read()).hexdigest()
    if live != card['docs_sha256']:
        raise SystemExit(f'docs.jsonl hash {live[:16]} does not match dataset_card '
                         f"{card['docs_sha256'][:16]}. Re-run prep.py.")

    here = os.path.dirname(os.path.abspath(__file__))
    env = env_capture.snapshot(code_paths=[os.path.join(here, f) for f in
                                           ('train_cell.py', 'metrics.py', 'env_capture.py')])
    os.makedirs(a.envdir, exist_ok=True)
    with open(f"{a.envdir}/{env['env_hash']}.json", 'w') as fh:
        json.dump(env, fh, indent=1, default=str)
    print(env_capture.summary_line(env), flush=True)
    if not card['complete']:
        print('WARNING: dataset_card says the corpus is INCOMPLETE. This is a '
              'partial-corpus run; its numbers are not the headline.', flush=True)

    init_seeds = [int(x) for x in a.init_seeds.split(',') if x != '']
    data_seeds = ([int(x) for x in a.data_seeds.split(',') if x != ''] if a.data_seeds
                  else list(init_seeds))
    if len(data_seeds) != len(init_seeds):
        raise SystemExit('--data-seeds must be the same length as --init-seeds')
    seed_pairs = list(zip(init_seeds, data_seeds))

    tasks = set(a.tasks.split(','))
    want_text = set(a.text.split(','))
    want_cells = set(x for x in a.cells.split(',') if x)
    cells = [c for c in S['cells']
             if c['task'] in tasks
             and ('raw' if c['text_field'] == 'text' else 'norm') in want_text
             and (not want_cells or c['cell'] in want_cells)]
    cells.sort(key=lambda c: (c['task'] != 'binary', c['cell']))

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            done.add(json.loads(l)['run_key_hash'])
    todo = []
    for c in cells:
        for isd, dsd in seed_pairs:
            _, kh = run_key(c, isd, dsd, a.model, a.pooling, grad_ckpt, hp)
            if kh not in done:
                todo.append((c, isd, dsd))
    if a.limit:
        todo = todo[:a.limit]
    print(f'{len(todo)} runs to do ({len(done)} already recorded in {a.out})', flush=True)
    for c, isd, dsd in todo:
        _, kh = run_key(c, isd, dsd, a.model, a.pooling, grad_ckpt, hp)
        warm = latest_checkpoint(os.path.join(a.workdir, kh))
        print(f"  {c['cell']:32s} init={isd} data={dsd}"
              + (f'   WARM START from {os.path.basename(warm)}' if warm else ''), flush=True)
    if a.dry_run:
        return

    failures = []
    for n, (c, isd, dsd) in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {c['task']} {c['cell']} init={isd} data={dsd}", flush=True)
        try:
            run(c, isd, dsd, docs, a.model, a.pooling, grad_ckpt, a.out, a.workdir,
                env, data_ident, hp, save_total_limit=a.save_total_limit,
                keep_checkpoints=a.keep_checkpoints, final_model_dir=a.final_model_dir)
        except Exception as e:                          # noqa: BLE001 - never abort a sweep
            import traceback
            failures.append({'cell': c['cell'], 'init_seed': isd, 'data_seed': dsd,
                             'model': a.model, 'pooling': a.pooling,
                             'error': type(e).__name__, 'msg': str(e)[:400],
                             'env_hash': env['env_hash']})
            with open(a.out.replace('.jsonl', '_failures.jsonl'), 'a') as fh:
                fh.write(json.dumps(failures[-1]) + '\n')
            print(f"FAILED {c['cell']} init={isd}: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
    print(f'\nSWEEP COMPLETE. {len(failures)} failed run(s).', flush=True)
    for f in failures:
        print(f"  FAILED {f['cell']} init={f['init_seed']} {f['error']}", flush=True)


if __name__ == '__main__':
    main()

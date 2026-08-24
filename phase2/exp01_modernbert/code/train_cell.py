"""exp01: fine-tune ModernBERT per cell x seed. Appends one JSON line per run.

Dispatches on the cell's `task` / `label_key` / `classes`.
  A_4way          4 classes, balanced, macro-F1 early stop, no class weights
  B_unseen_assign 3 classes train, test is ONE unseen class -> assignment dist
  C_binary        2 classes, class weights for the 2:1 imbalance, AUC early stop

HYPERPARAMETERS ARE FIXED IN ADVANCE and are NOT tuned against test.
Early stopping uses a validation slice carved out of TRAIN BY QUESTION ID, so a
question's human answer and its AI answers never straddle the split.
Test is touched exactly once per run, for scoring.
"""
import json, os, sys, random, argparse, time, itertools
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import auc, balanced_accuracy, stratified_auc, macro_f1, confusion

# ---- pre-registered recipe -------------------------------------------------
HP = dict(lr=5e-5, batch=8, max_epochs=10, warmup_ratio=0.1, weight_decay=0.01,
          patience=3, val_question_frac=0.15, max_length=8192, decision_threshold=0.5)


class DS(Dataset):
    def __init__(self, docs, ids, field, tok, maxlen, lab_of):
        self.e = tok([docs[i][field] for i in ids], truncation=True,
                     max_length=maxlen, padding=False)
        self.y = [lab_of(docs[i]) for i in ids]
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return {'input_ids': self.e['input_ids'][i],
                'attention_mask': self.e['attention_mask'][i], 'labels': self.y[i]}


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop('labels')
        out = model(**inputs)
        w = None if self.class_weights is None else self.class_weights.to(out.logits.device)
        loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w)
        return (loss, out) if return_outputs else loss


def run(cell, seed, docs, model_name, outpath, workdir):
    field, task = cell['text_field'], cell['task']
    classes = cell['classes']
    cidx = {c: i for i, c in enumerate(classes)}
    lk = cell['label_key']
    lab_of = lambda d: cidx[d[lk]]
    K = len(classes)
    tok = AutoTokenizer.from_pretrained(model_name)

    # --- validation carve: BY QUESTION, never by document --------------------
    train_ids = cell['train_ids']
    qs = sorted({docs[i]['q_id'] for i in train_ids})
    rng = random.Random(f"val|{seed}|{cell['cell']}")
    rng.shuffle(qs)
    val_q = set(qs[:round(len(qs) * HP['val_question_frac'])])
    tr_ids = [i for i in train_ids if docs[i]['q_id'] not in val_q]
    va_ids = [i for i in train_ids if docs[i]['q_id'] in val_q]
    assert not ({docs[i]['q_id'] for i in tr_ids} & {docs[i]['q_id'] for i in va_ids})
    assert not ({docs[i]['q_id'] for i in train_ids} &
                {docs[i]['q_id'] for i in cell['test_ids']}), 'TRAIN/TEST QUESTION LEAK'
    assert len({lab_of(docs[i]) for i in va_ids}) == K, 'val missing a class'

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=K)

    # class weights ONLY for C (A and B are balanced by construction)
    cw = None
    if task == 'C_binary':
        cnt = [sum(1 for i in tr_ids if lab_of(docs[i]) == c) for c in range(K)]
        cw = torch.tensor([len(tr_ids)/(K*n) for n in cnt], dtype=torch.float)

    is_binary = (K == 2)
    def compute_metrics(p):
        pr = torch.softmax(torch.tensor(p.predictions), -1).numpy()
        t = list(p.label_ids)
        if is_binary:
            return {'sel': auc(list(pr[:, 1]), ['ai' if y == 1 else 'human' for y in t])}
        return {'sel': macro_f1(list(pr.argmax(1)), t, K)[0]}

    # transformers v5 removed warmup_ratio; compute the identical step count so the
    # pre-registered 0.1 warmup ratio is preserved exactly, not silently dropped.
    import math
    steps_per_epoch = math.ceil(len(tr_ids) / HP['batch'])
    warmup_steps = round(HP['warmup_ratio'] * steps_per_epoch * HP['max_epochs'])

    args = TrainingArguments(
        output_dir=f'{workdir}/ckpt',
        learning_rate=HP['lr'], per_device_train_batch_size=HP['batch'],
        per_device_eval_batch_size=16, num_train_epochs=HP['max_epochs'],
        warmup_steps=warmup_steps, weight_decay=HP['weight_decay'],
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model='sel',
        greater_is_better=True, logging_strategy='no', report_to=[],
        seed=seed, data_seed=seed, bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(), disable_tqdm=True,
        # Activation memory only: recomputes activations instead of storing them.
        # Mathematically IDENTICAL to not using it -- same batch, same LR, same
        # gradients. Required because the longest doc (2162 tok) makes a batch of 8
        # pad to 17,296 token-positions, 7x typical, which OOMs a 23GB L4.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})

    trainer = WeightedTrainer(
        class_weights=cw, model=model, args=args,
        train_dataset=DS(docs, tr_ids, field, tok, HP['max_length'], lab_of),
        eval_dataset=DS(docs, va_ids, field, tok, HP['max_length'], lab_of),
        processing_class=tok, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=HP['patience'])])
    t0 = time.time(); trainer.train(); train_s = time.time() - t0

    # --- test: touched once ---------------------------------------------------
    te_ids = cell['test_ids']
    pred = trainer.predict(DS(docs, te_ids, field, tok, HP['max_length'],
                              lambda d: cidx.get(d[lk], 0)))   # B: true class unseen
    P = torch.softmax(torch.tensor(pred.predictions), -1).numpy()
    pidx = list(P.argmax(1))
    nl = [docs[i]['newlines'] for i in te_ids]

    rec = {'task': task, 'cell': cell['cell'], 'text': 'raw' if field == 'text' else 'norm',
           'seed': seed, 'model': model_name, 'classes': classes,
           'val_sel_best': trainer.state.best_metric, 'epochs_run': trainer.state.epoch,
           'train_seconds': round(train_s, 1), 'n_train': len(tr_ids),
           'n_val': len(va_ids), 'n_test': len(te_ids), 'hp': HP}

    if task == 'B_unseen_assign':
        # every test doc is the held-out generator; report where they land
        n = len(te_ids)
        rec['assignment'] = {c: pidx.count(k)/n for k, c in enumerate(classes)}
        rec['assignment_counts'] = {c: pidx.count(k) for k, c in enumerate(classes)}
        rec['human_share'] = rec['assignment']['HUMAN']
        rec['mean_prob'] = {c: float(P[:, k].mean()) for k, c in enumerate(classes)}
        rec['heldout_generator'] = cell['cell'].split('|')[1]
    else:
        tidx = [cidx[docs[i][lk]] for i in te_ids]
        mf1, perf1 = macro_f1(pidx, tidx, K)
        rec['accuracy'] = sum(1 for p, t in zip(pidx, tidx) if p == t)/len(tidx)
        rec['macro_f1'] = mf1
        rec['per_class_f1'] = dict(zip(classes, perf1))
        rec['confusion'] = confusion(pidx, tidx, K)   # M[true][pred]
        if task == 'A_4way':
            pw = {}
            for a, b in itertools.combinations(range(K), 2):
                sub = [(P[j, a]/(P[j, a]+P[j, b]+1e-12), tidx[j])
                       for j in range(len(tidx)) if tidx[j] in (a, b)]
                pw[f'{classes[a]}|{classes[b]}'] = auc([s for s, _ in sub],
                                                       [t for _, t in sub], pos=a)
            rec['pairwise_auc'] = pw
        if task == 'C_binary':
            p_ai = list(P[:, cidx['ai']])
            lab = [docs[i]['label'] for i in te_ids]
            rec['auc'] = auc(p_ai, lab)
            rec['balanced_acc'] = balanced_accuracy(p_ai, lab, HP['decision_threshold'])
            if field == 'text':
                sa, kept, tot = stratified_auc(p_ai, lab, nl, k=3)
                rec.update({'strat_auc_k3': sa, 'strat_pairs_kept': kept,
                            'strat_pairs_total': tot})
    # full precision: 6dp rounding collapsed saturated pairs into ties and made
    # pairwise AUC non-reproducible from the mirror
    rec['probs'] = {i: [float(x) for x in P[j]] for j, i in enumerate(te_ids)}

    with open(outpath, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')
    head = (f"assign HUMAN={rec['human_share']:.3f}" if task == 'B_unseen_assign'
            else (f"AUC={rec['auc']:.3f} BA={rec['balanced_acc']:.3f}" if task == 'C_binary'
                  else f"acc={rec['accuracy']:.3f} macroF1={rec['macro_f1']:.3f}"))
    print(f"DONE {rec['cell']:34s} seed={seed} {head} val={rec['val_sel_best']:.3f} "
          f"ep={rec['epochs_run']:.0f} {train_s:.0f}s", flush=True)
    del model, trainer; torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', default='/content/exp01/docs.jsonl')
    ap.add_argument('--splits', default='/content/exp01/splits.json')
    ap.add_argument('--out', default='/content/exp01/results.jsonl')
    ap.add_argument('--workdir', default='/content/work')
    ap.add_argument('--model', default='answerdotai/ModernBERT-base')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--tasks', default='A_4way,B_unseen_assign,C_binary')
    a = ap.parse_args()

    docs = {d['doc_id']: d for d in map(json.loads, open(a.docs))}
    S = json.load(open(a.splits))
    seeds = [int(x) for x in a.seeds.split(',')]
    order = {t: n for n, t in enumerate(a.tasks.split(','))}
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            done.add((r['cell'], r['seed'], r['model']))
    cells = sorted([c for c in S['cells'] if c['task'] in order],
                   key=lambda c: order[c['task']])
    todo = [(c, s) for c in cells for s in seeds if (c['cell'], s, a.model) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already recorded)", flush=True)
    failures = []
    for n, (c, s) in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {c['task']} {c['cell']} seed={s}", flush=True)
        try:
            run(c, s, docs, a.model, a.out, a.workdir)
        except Exception as e:                      # noqa: BLE001 - must not abort the sweep
            import traceback, torch as _t
            failures.append({'cell': c['cell'], 'seed': s, 'model': a.model,
                             'error': type(e).__name__, 'msg': str(e)[:400]})
            with open(a.out.replace('.jsonl', '_failures.jsonl'), 'a') as fh:
                fh.write(json.dumps(failures[-1]) + '\n')
            print(f"FAILED {c['cell']} seed={s}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            traceback.print_exc()
            _t.cuda.empty_cache()
    print(f"\nSWEEP COMPLETE. {len(failures)} failed run(s).", flush=True)
    for f in failures:
        print(f"  FAILED {f['cell']} seed={f['seed']} {f['error']}", flush=True)


if __name__ == '__main__':
    main()

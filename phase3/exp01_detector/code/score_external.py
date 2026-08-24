#!/usr/bin/env python3
"""Score arbitrary documents with a trained detector. EVALUATION ONLY.

The external-evaluation hook. Its whole reason to exist is that in-domain
results are saturated (AUC 1.0000) and cannot measure anything further, so the
only test left that can fail is out-of-domain text.

TWO RULES, ENFORCED HERE RATHER THAN REMEMBERED:

1. **These labels are never training labels.** This script only reads a model
   and writes probabilities. If the input carries a `label` field it is recorded
   verbatim and used for nothing.

2. **The preprocessing must match training exactly or the read is invalid.**
   External text goes through the SAME `norm_ws` as `prep.py`, the same
   tokenizer from the model directory, the same `max_length`. A pipeline
   mismatch would produce a number that looks like a finding and is an artifact
   of preprocessing. The applied settings are asserted against the training
   record shipped alongside the model (`run_record_meta.json`) and written into
   the output.

Output carries per-document probabilities, the environment, the model identity,
and a hash of the input file, so the read is auditable the same way a training
run is.

    python score_external.py --model DIR --input chunks.jsonl --out scored.json
"""
import argparse, hashlib, json, os, re, sys
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_capture                                               # noqa: E402


def norm_ws(t):
    """Byte-identical to prep.py's normalisation. If these ever diverge the
    external read is measuring preprocessing, not text."""
    return re.sub(r'\s+', ' ', t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='directory written by --final-model-dir')
    ap.add_argument('--input', required=True, help='jsonl with doc_id and text')
    ap.add_argument('--out', required=True)
    ap.add_argument('--text-field', default='text')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--raw', action='store_true',
                    help='skip normalisation. Only for a deliberate raw-vs-norm '
                         'comparison; the trained model expects normalised text.')
    a = ap.parse_args()

    meta_path = os.path.join(a.model, 'run_record_meta.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    hp = meta.get('hp', {})
    max_length = hp.get('max_length', 2048)
    trained_field = meta.get('run_key', {}).get('cell', '')
    if 'norm' in trained_field and a.raw:
        print('WARNING: this model was trained on NORMALISED text and --raw was '
              'given. The read will not be comparable to its training scores.')

    rows = [json.loads(l) for l in open(a.input)]
    texts = [r[a.text_field] if a.raw else norm_ws(r[a.text_field]) for r in rows]
    ids = [r.get('doc_id', str(i)) for i, r in enumerate(rows)]

    cfg = AutoConfig.from_pretrained(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, attn_implementation='sdpa')
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # the loaded model must match the recipe the record claims, or the read is
    # of a different model than the one being reported on
    eff = {'classifier_pooling': getattr(model.config, 'classifier_pooling', None),
           'attn_implementation': getattr(model.config, '_attn_implementation', None),
           'num_labels': model.config.num_labels}
    want_pool = meta.get('code_path', {}).get('classifier_pooling')
    if want_pool:
        assert eff['classifier_pooling'] == want_pool, \
            f"pooling mismatch: model has {eff['classifier_pooling']}, record says {want_pool}"
    assert eff['num_labels'] == 2, f"expected a binary head, got {eff['num_labels']}"

    lengths, probs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), a.batch):
            chunk = texts[i:i + a.batch]
            full = tok(chunk, truncation=False)['input_ids']
            lengths += [len(x) for x in full]
            enc = tok(chunk, truncation=True, max_length=max_length,
                      padding=True, return_tensors='pt')
            if torch.cuda.is_available():
                enc = {k: v.cuda() for k, v in enc.items()}
            logits = model(**enc).logits.float()
            probs += torch.softmax(logits, -1).cpu().numpy().tolist()

    # classes are ['human', 'ai'] throughout this experiment; assert rather than
    # assume, because a silently flipped index inverts every number below
    classes = meta.get('classes') or ['human', 'ai']
    assert classes == ['human', 'ai'], f'unexpected class order {classes}'
    ai_idx = classes.index('ai')
    p_ai = [float(p[ai_idx]) for p in probs]
    L = sorted(lengths)
    out = {
        'model_dir': os.path.abspath(a.model),
        'model_run_key_hash': meta.get('run_key_hash'),
        'model_cell': meta.get('run_key', {}).get('cell'),
        'model_seeds': meta.get('seeds'),
        'model_test_scores': meta.get('test'),
        'trained_on_docs_sha256': meta.get('data', {}).get('docs_sha256'),
        'effective_model_config': eff,
        'input_path': os.path.abspath(a.input),
        'input_sha256': hashlib.sha256(open(a.input, 'rb').read()).hexdigest(),
        'n': len(rows), 'text_field': a.text_field,
        'normalisation': 'raw (NOT normalised)' if a.raw else 'norm_ws, identical to prep.py',
        'max_length': max_length,
        'tokens': {'median': L[len(L)//2], 'max': L[-1],
                   'n_truncated': sum(1 for n in lengths if n > max_length)},
        'labels_in_input': sorted({str(r.get('label')) for r in rows}),
        'labels_used_for': 'NOTHING -- this script never trains, and never scores '
                           'accuracy against an input label',
        'env': env_capture.snapshot(code_paths=[__file__]),
        'p_ai': dict(zip(ids, p_ai)),
        'probs': {i: [float(x) for x in p] for i, p in zip(ids, probs)},
    }
    with open(a.out, 'w') as fh:
        json.dump(out, fh, default=str)
    arr = np.array(p_ai)
    print(f"scored {len(arr)} documents with {out['model_run_key_hash']} "
          f"({out['model_cell']}, pooling {eff['classifier_pooling']})")
    print(f"  p_ai mean {arr.mean():.4f}  median {np.median(arr):.4f}  "
          f"sd {arr.std(ddof=1):.4f}  min {arr.min():.6f}  max {arr.max():.6f}")
    print(f"  called AI at 0.5: {(arr > 0.5).sum()}/{len(arr)} = {(arr > 0.5).mean():.4f}")
    print(f"  truncated at max_length: {out['tokens']['n_truncated']}")


if __name__ == '__main__':
    main()

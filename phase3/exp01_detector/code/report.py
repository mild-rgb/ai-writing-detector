#!/usr/bin/env python3
"""Reporting and run-bookkeeping helpers with no GPU or transformers dependency.

Deliberately separate from train_cell.py so the run-identity and scoring logic
can be unit-tested, and re-run over stored per-document probabilities, on a
machine with no GPU at all. That was phase 2's layer-2 property and it is what
made its analysis exact.
"""
import hashlib, json, os
import numpy as np
from metrics import auc, balanced_accuracy

# ModernBERT specifics that are load-bearing. Imported by train_cell so the run
# key and the model construction cannot drift apart.
#   sdpa           flash_attention_2 has a known NaN bug -- do not switch it on
#   reference_compile=False  the torch.compile path is fragile on Colab and
#                            changes numerics run to run
MODEL_KW = dict(attn_implementation='sdpa', reference_compile=False)


def run_key(cell, init_seed, data_seed, model_name, pooling, grad_ckpt, hp, model_kw=MODEL_KW):
    """Identity of a run. Everything that can move a number is in it, so the
    resume check cannot silently confuse two different recipes."""
    k = {'cell': cell['cell'], 'init_seed': init_seed, 'data_seed': data_seed,
         'model': model_name, 'classifier_pooling': pooling,
         'gradient_checkpointing': grad_ckpt,
         'test_ids_hash': cell['id_hashes']['test_ids'],
         'train_ids_hash': cell['id_hashes']['train_ids'],
         'hp': hp, 'model_kw': model_kw}
    return k, hashlib.sha256(json.dumps(k, sort_keys=True).encode()).hexdigest()[:16]


def binary_report(p_ai, labels, threshold):
    """The standard block, computed the same way everywhere it appears.

    Inputs may arrive as numpy scalars straight off a softmax; everything
    returned here is a plain Python float, because this block goes into a JSON
    record and a numpy float32 raises at serialisation time -- after the GPU
    work is done and paid for.
    """
    p_ai = [float(s) for s in p_ai]
    threshold = float(threshold)
    n_ai = sum(1 for l in labels if l == 'ai')
    n_hu = len(labels) - n_ai
    if n_ai == 0 or n_hu == 0:
        return {'n': len(labels), 'n_ai': n_ai, 'n_human': n_hu, 'auc': None}
    tp = sum(1 for s, l in zip(p_ai, labels) if s > threshold and l == 'ai')
    fp = sum(1 for s, l in zip(p_ai, labels) if s > threshold and l != 'ai')
    return {'n': len(labels), 'n_ai': n_ai, 'n_human': n_hu,
            'auc': auc(p_ai, labels),
            'balanced_acc': balanced_accuracy(p_ai, labels, threshold),
            'accuracy': (tp + (n_hu - fp)) / len(labels),
            'recall_ai': tp / n_ai, 'fpr_human': fp / n_hu,
            'specificity': 1 - fp / n_hu,
            'mean_p_ai_on_ai': float(np.mean([s for s, l in zip(p_ai, labels) if l == 'ai'])),
            'mean_p_ai_on_human': float(np.mean([s for s, l in zip(p_ai, labels) if l != 'ai'])),
            'threshold': threshold}


def best_threshold(p_ai, labels):
    """Threshold maximising balanced accuracy ON VAL. Reported alongside the
    pre-declared 0.5 -- never chosen on test."""
    # float() not just round(): round(np.float32) returns np.float32, which then
    # rides into the record as the chosen threshold and makes the whole record
    # unserialisable AFTER training has finished. Cast at every boundary.
    cands = sorted(set([0.5] + [float(round(float(s), 4)) for s in p_ai]))
    best, bt = -1.0, 0.5
    for t in cands:
        b = balanced_accuracy(p_ai, labels, t)
        if b == b and b > best:
            best, bt = b, t
    return bt, best




def latest_checkpoint(d):
    """Newest `checkpoint-N` under d, or None. Used to warm-start after a kernel
    or VM death mid-run."""
    if not os.path.isdir(d):
        return None
    cks = [x for x in os.listdir(d) if x.startswith('checkpoint-')
           and os.path.isfile(os.path.join(d, x, 'trainer_state.json'))]
    if not cks:
        return None
    return os.path.join(d, max(cks, key=lambda x: int(x.split('-')[1])))

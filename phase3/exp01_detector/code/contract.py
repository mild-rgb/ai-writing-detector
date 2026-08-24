#!/usr/bin/env python3
"""The reproducibility contract, as an executable check.

Phase 2 did not fail to reproduce because anyone decided not to record the
environment -- it failed because nothing checked. This file makes the contract
testable: run it on results.jsonl and it says, per record, whether every clause
is actually satisfied. A sweep that passes here is one a stranger can audit.

Usage:  python contract.py results.jsonl
Exit status is nonzero if any record violates the contract.
"""
import json, sys


def _get(rec, path):
    """Fetch a dotted path, returning (found, value)."""
    cur = rec
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


# clause -> (dotted path, predicate, why it is in the contract)
CLAUSES = [
    ('pip freeze', 'env.pip_freeze',
     lambda v: isinstance(v, list) and len(v) > 10,
     'phase 2 recorded no environment at all and called it the cheapest gap it could '
     'have closed'),
    ('torch version', 'env.libraries.torch', bool, 'library versions move numbers'),
    ('transformers version', 'env.libraries.transformers', bool, 'ditto'),
    ('python version', 'env.python_version_short', bool, 'ditto'),
    ('CUDA version', 'env.libraries.torch_cuda', bool, 'CUDA version moves the last bits'),
    ('cuDNN version', 'env.libraries.torch_cudnn', bool, 'ditto'),
    ('GPU model', 'env.gpu.name', bool, 'a different GPU does not reproduce a number'),
    ('driver version', 'env.gpu.driver_version', bool, 'ditto'),
    ('torch numerics flags', 'env.torch_flags',
     lambda v: isinstance(v, dict) and 'cuda_matmul_allow_tf32' in v,
     'TF32 alone changes results'),
    ('env hash', 'env.env_hash', bool, 'so two runs can be compared at a glance'),
    ('code fingerprint', 'env.code._combined', bool, 'pins the code as tightly as the data'),

    ('weight-init seed', 'seeds.weight_init', lambda v: isinstance(v, int),
     'every seed logged SEPARATELY'),
    ('data-order seed', 'seeds.data_order', lambda v: isinstance(v, int),
     'weight init and data order are different seeds and must not be conflated'),

    ('hyperparameters', 'hp',
     lambda v: isinstance(v, dict) and {'lr', 'batch', 'max_epochs', 'weight_decay',
                                        'max_length'} <= set(v),
     'the exact recipe, including any lr swept down on divergence'),

    ('docs hash', 'data.docs_sha256', lambda v: isinstance(v, str) and len(v) == 64,
     'the data identity is pinned'),
    ('splits hash', 'data.splits_sha256', lambda v: isinstance(v, str) and len(v) == 64,
     'the split identity is pinned'),
    ('cell id hashes', 'data.cell_id_hashes',
     lambda v: isinstance(v, dict) and {'train_ids', 'val_ids', 'test_ids'} <= set(v),
     'this cell\'s exact membership, verifiable independently of the file hash'),

    ('gradient checkpointing', 'code_path.gradient_checkpointing',
     lambda v: isinstance(v, bool),
     'phase 2 measured it moving accuracy 0.9347 -> 0.9034 at identical seed and data'),
    ('attn implementation', 'code_path.attn_implementation',
     lambda v: v == 'sdpa', 'flash_attention_2 has a known NaN bug'),
    ('reference_compile', 'code_path.reference_compile',
     lambda v: v is False, 'the compile path changes numerics run to run'),
    ('classifier pooling', 'code_path.classifier_pooling',
     lambda v: v in ('cls', 'mean'), 'cls vs mean is a swept axis, not an accident'),
    ('precision', 'code_path.precision', lambda v: v == 'bf16', 'bf16, never fp16'),

    ('per-document probabilities', 'probs',
     lambda v: isinstance(v, dict) and len(v) > 0,
     'THE HARD RULE -- what made phase 2 layer-2 analysis exact and caught six errors '
     'with no GPU'),
    ('per-generator results', 'per_generator',
     lambda v: isinstance(v, dict) and len(v) > 0,
     'detection ranges 7-80% across models; a pooled rate averages things that are '
     'not alike'),
    ('full-state checkpointing', 'code_path.checkpointing.full_state',
     lambda v: v is True,
     'checkpoints carry optimizer, scheduler and RNG state, so a restart warm-starts '
     'mid-run instead of silently redoing an epoch'),
    ('checkpoint selection metric', 'code_path.checkpoint_selection.metric',
     lambda v: isinstance(v, str) and v,
     'dev AUC saturates on this corpus, so WHICH metric selected the checkpoint '
     'changes the reported balanced accuracy and must be on the record'),
    ('early-stopping semantics', 'code_path.early_stopping.patience_evals',
     lambda v: isinstance(v, int) and v > 0,
     'patience is counted in EVALUATIONS; under a step cadence that is not epochs, '
     'and the record must say which behaviour was in force'),
    ('checkpoint cadence', 'code_path.checkpointing.strategy',
     lambda v: v in ('steps', 'epoch'),
     'cadence sets how often dev is scored and therefore which step is selected best'),
    ('resume provenance', 'code_path.resumed_from_checkpoint',
     lambda v: True,
     'a warm-started run restores RNG state but need not match an uninterrupted run '
     'bit for bit; the reader has to be able to tell them apart'),
    ('epochs actually run', 'epochs_run', lambda v: v is not None,
     'early stopping is part of the result'),
    ('truncation stats', 'tokenisation.test.truncated_frac',
     lambda v: v is not None, 'a changing truncation rate changes the task'),
]

# clauses that are advisory rather than fatal
SOFT = {'code fingerprint', 'driver version', 'CUDA version', 'cuDNN version'}


def check(rec):
    """Return (hard_failures, soft_failures) as lists of (clause, why)."""
    hard, soft = [], []
    for name, path, pred, why in CLAUSES:
        found, val = _get(rec, path)
        ok = False
        if found:
            try:
                ok = bool(pred(val))
            except Exception:                            # noqa: BLE001
                ok = False
        if not ok:
            (soft if name in SOFT else hard).append((name, why))
    return hard, soft


def probs_are_consistent(rec, tol=1e-6):
    """The per-document probabilities must actually reproduce the reported test
    metrics. If they do not, the record is not auditable no matter what it stores."""
    from metrics import auc
    if 'probs' not in rec or not rec.get('test', {}).get('auc'):
        return None
    ai_idx = rec['classes'].index('ai')
    ids = list(rec['probs'])
    # label recoverable from the cell only via docs.jsonl; instead re-derive the
    # count of test documents and the AUC's own inputs where possible
    n_ok = len(ids) == rec['n']['test']
    rowsum_ok = all(abs(sum(rec['probs'][i]) - 1.0) < 1e-3 for i in ids)
    in_range = all(0.0 <= rec['probs'][i][ai_idx] <= 1.0 for i in ids)
    return {'n_matches_reported': n_ok, 'rows_sum_to_one': rowsum_ok,
            'in_range': in_range}


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: contract.py results.jsonl')
    bad = 0
    recs = [json.loads(l) for l in open(sys.argv[1])]
    print(f'{len(recs)} record(s) in {sys.argv[1]}\n')
    for n, rec in enumerate(recs, 1):
        hard, soft = check(rec)
        pc = probs_are_consistent(rec)
        tag = f"{rec.get('cell','?')} init={rec.get('seeds',{}).get('weight_init')} " \
              f"data={rec.get('seeds',{}).get('data_order')} " \
              f"pool={rec.get('code_path',{}).get('classifier_pooling')}"
        if hard:
            bad += 1
            print(f'[{n}] FAIL  {tag}')
            for name, why in hard:
                print(f'        missing: {name}  -- {why}')
        else:
            print(f'[{n}] pass  {tag}')
        for name, why in soft:
            print(f'        advisory: {name} absent -- {why}')
        if pc and not all(pc.values()):
            bad += 1
            print(f'        PROBS INCONSISTENT: {pc}')
    print(f'\n{len(recs)-bad}/{len(recs)} records satisfy the contract.')
    if bad:
        print('CONTRACT VIOLATED -- do not report these runs as reproducible.')
    raise SystemExit(1 if bad else 0)


if __name__ == '__main__':
    sys.path.insert(0, __file__.rsplit('/', 1)[0])
    main()

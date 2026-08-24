#!/usr/bin/env python3
"""Permanent pre-sweep fixture: exercise every task path WITHOUT reading test.

Why this exists. It caught a bug that would have invalidated all 60 runs while
leaving a results file that looked completely normal: `compute_metrics` was
never passed to the Trainer, so `metric_for_best_model='sel'` was never found,
early stopping was silently disabled, and every run would have trained the full
10 epochs with no validation-based model selection. transformers announces this
only as a one-line warning buried in stderr.

The trick that makes it safe to run any time: the fabricated cells draw BOTH
their train and their "test" ids from TRAINING questions only, and assert
disjointness from every real test id in splits.json. No pre-registered test
document is ever read, so running this costs nothing in terms of the
"test is touched exactly once" rule.

Run before any sweep, and after any change to train_cell.py or any library
upgrade. Exits non-zero on failure.

    python smoke_test.py --docs docs.jsonl --splits splits.json
"""
import sys, os, json, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', default='/content/exp01/docs.jsonl')
    ap.add_argument('--splits', default='/content/exp01/splits.json')
    ap.add_argument('--model', default='answerdotai/ModernBERT-large')
    ap.add_argument('--out', default='/content/smoke.jsonl')
    ap.add_argument('--workdir', default='/content/work')
    ap.add_argument('--epochs', type=int, default=2, help='smoke only; HP is restored after')
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_cell

    docs = {d['doc_id']: d for d in map(json.loads, open(a.docs))}
    S = json.load(open(a.splits))
    real_test = {i for c in S['cells'] for i in c['test_ids']}
    qtr = sorted(set(S['q_train']))
    hold, keep = set(qtr[:30]), set(qtr[30:70])

    saved = train_cell.HP['max_epochs']
    train_cell.HP['max_epochs'] = a.epochs
    failures = []
    try:
        for task, must_have in (('A_4way', 'pairwise_auc'),
                                ('B_unseen_assign', 'human_share'),
                                ('C_binary', 'auc')):
            C = [c for c in S['cells'] if c['task'] == task][0]
            fake = dict(C)
            fake['cell'] = f'SMOKE-{task}|not-a-result'
            fake['train_ids'] = [i for i in C['train_ids'] if docs[i]['q_id'] in keep]
            fake['test_ids'] = [i for i in C['train_ids'] if docs[i]['q_id'] in hold]
            assert not (set(fake['test_ids']) & real_test), \
                'FIXTURE BROKEN: smoke would read a real test document'
            assert not (set(fake['train_ids']) & set(fake['test_ids']))
            r = train_cell.run(fake, 0, docs, a.model, a.out, a.workdir)
            # the assertions that matter
            if must_have not in r:
                failures.append(f'{task}: missing {must_have}')
            if r['val_sel_best'] is None:
                failures.append(f'{task}: val metric absent -> EARLY STOPPING IS DISABLED')
            if not (0 < r['epochs_run'] <= a.epochs):
                failures.append(f"{task}: implausible epochs_run {r['epochs_run']}")
            print(f"   -> {task}: stopped epoch {r['epochs_run']:.0f}, "
                  f"val={r['val_sel_best']:.3f}, has {must_have}: {must_have in r}", flush=True)
    finally:
        train_cell.HP['max_epochs'] = saved
        if os.path.exists(a.out):
            os.remove(a.out)

    if failures:
        print('\nSMOKE FAILED:'); [print('  -', f) for f in failures]
        return 1
    print(f'\nSMOKE PASSED — all three task paths OK, HP restored '
          f'(max_epochs={train_cell.HP["max_epochs"]}), no test document read')
    return 0

if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""How much of the detector is available from ~40 mechanical regexes?

Phase 2 carried a newline baseline: newline count alone as a detector, which is
0.500 by construction once whitespace is normalised, and that is what proves a
`norm` result is not whitespace. But **normalising whitespace does not remove
Reddit**. The human class is real ELI5 answers and still carries `Edit:` lines,
`/r/` references, markdown bullets, first-person voice and mild profanity after
every run of whitespace has been collapsed to a single space. Those survive
`norm` intact.

So this is the wider control: fit a logistic regression on nothing but the
presence/absence of `phase3/scripts/markers.py`'s marker set -- the project's
own mechanical definitions, regexes rather than judgements -- and score it the
same way a trained detector is scored, on the same splits, per generator.

Read it as a FLOOR, not as a rival. If the transformer beats it comfortably it is
reading prose. If it does not, the expensive model is doing what forty regexes
do, and the honest headline is the floor.

No GPU. No transformers. Runs in seconds.

    python marker_baseline.py --field text_norm
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(EXP, '..', 'scripts')))
from metrics import auc                                          # noqa: E402
from report import binary_report                                 # noqa: E402
from markers import MARKERS                                      # noqa: E402


def featurise(texts):
    """Binary presence matrix, one column per marker. Deliberately presence and
    not count: a count would smuggle in length, which is matched between the
    classes by construction and must not become the signal."""
    return np.array([[1.0 if test(t) else 0.0 for _, _, test in MARKERS]
                     for t in texts], dtype=np.float64)


def fit_logistic(X, y, l2=1.0, iters=300, lr=0.5):
    """Plain L2 logistic regression by gradient descent. Written out rather than
    imported so this control has no dependency that could fail to install on the
    machine doing the auditing -- the whole point is that it runs anywhere."""
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y)
        g[:-1] += l2 * w[:-1] / len(y)          # no penalty on the intercept
        w -= lr * g
    return w


def predict(X, w):
    return 1.0 / (1.0 + np.exp(-(np.hstack([X, np.ones((len(X), 1))]) @ w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', default=f'{EXP}/docs.jsonl')
    ap.add_argument('--splits', default=f'{EXP}/splits.json')
    ap.add_argument('--field', default='text_norm', choices=['text', 'text_norm'])
    ap.add_argument('--cell', default=None,
                    help='score a specific cell (default: the matching binary cell)')
    ap.add_argument('--top', type=int, default=12, help='single markers to list')
    a = ap.parse_args()

    docs = {d['doc_id']: d for d in map(json.loads, open(a.docs))}
    S = json.load(open(a.splits))
    want = a.cell or ('binary|' + ('norm' if a.field == 'text_norm' else 'raw'))
    cell = next(c for c in S['cells'] if c['cell'] == want)
    tr, te = cell['train_ids'], cell['test_ids']
    print(f"cell {cell['cell']}   field {a.field}   "
          f"train {len(tr)}  test {len(te)}   {len(MARKERS)} markers\n")

    Xtr = featurise([docs[i][a.field] for i in tr])
    ytr = np.array([1.0 if docs[i]['label'] == 'ai' else 0.0 for i in tr])
    Xte = featurise([docs[i][a.field] for i in te])
    lte = [docs[i]['label'] for i in te]

    # --- single strongest markers, no fitting at all ------------------------
    print(f'strongest SINGLE markers (presence alone, AUC on test):')
    singles = []
    for k, (fam, name, _) in enumerate(MARKERS):
        col = list(Xte[:, k])
        r = float(np.mean(Xte[:, k]))
        singles.append((abs(auc(col, lte) - 0.5), auc(col, lte), fam, name, r))
    print(f"  {'AUC':>5s} {'|dev|':>6s}  {'family':9s} {'marker':42s} fires on")
    for dev, u, fam, name, r in sorted(singles, reverse=True)[:a.top]:
        lean = 'AI' if u > 0.5 else 'human'
        print(f"  {u:.3f} {dev:+6.3f}  {fam:9s} {name:42s} {r*100:5.1f}% "
              f"of test docs, leans {lean}")

    # --- the fitted floor ---------------------------------------------------
    w = fit_logistic(Xtr, ytr)
    p = list(predict(Xte, w))
    rep = binary_report(p, lte, 0.5)
    print(f"\nFITTED MARKER FLOOR (logistic on marker presence, trained on the "
          f"cell's own train split)")
    print(f"  AUC {rep['auc']:.4f}   balanced acc {rep['balanced_acc']:.4f}   "
          f"recall_ai {rep['recall_ai']:.3f}   fpr_human {rep['fpr_human']:.3f}")

    print('\n  per generator (AI side varies; human side is the same documents):')
    hu = [(s, 'human') for s, i in zip(p, te) if docs[i]['label'] == 'human']
    rows = []
    for g in sorted({docs[i]['generator'] for i in te if docs[i]['label'] == 'ai'}):
        sub = [(s, 'ai') for s, i in zip(p, te) if docs[i]['generator'] == g] + hu
        rr = binary_report([s for s, _ in sub], [l for _, l in sub], 0.5)
        rows.append((rr['auc'], g, rr['recall_ai']))
    for u, g, rc in sorted(rows, reverse=True):
        print(f"    {g:34s} AUC {u:.4f}   recall {rc:.3f}")
    if len(rows) > 1:
        print(f"    spread {min(r[0] for r in rows):.4f} - {max(r[0] for r in rows):.4f}")

    print(f"\n  Read this as the floor the transformer has to clear. A trained "
          f"detector that\n  lands near it is reading markers, not prose.")


if __name__ == '__main__':
    main()

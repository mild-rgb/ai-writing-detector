#!/usr/bin/env python3
"""The bag-of-words floor. The control that actually matters here.

`newline_baseline_auc` proves a `norm` result is not whitespace.
`marker_baseline.py` proves it is not the ~40 mechanical Reddit/LLM markers.
Neither is a hard enough test. This one is:

    a logistic regression on binary word-presence, no ordering, no syntax,
    no embeddings, no GPU -- and on this corpus it reaches AUC 0.999.

That reframes any transformer number. A fine-tuned ModernBERT-large scoring
~1.000 is not evidence that it understands prose; it is evidence that the two
classes are nearly separable in plain word space, which bag-of-words already
told us for free. Report the transformer ABOVE this floor, never against 0.5.

Also runs the two diagnostics that say WHY, because "0.999" on its own invites
the wrong conclusion:
  --words-only   deletes all punctuation, to check the score is not carried by
                 parentheses and slashes (it is not)
  --blacklist    checks whether the floor prompt's banned vocabulary is doing
                 the work (it is not -- it fires on too few human documents)

    python bow_baseline.py --field text_norm
"""
import argparse, collections, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from metrics import auc                                          # noqa: E402
from report import binary_report                                 # noqa: E402

# from phase3/corpus/floor/prompt.txt -- the words the generator is forbidden
BANNED = ['delve', 'crucial', 'vital', 'pivotal', 'robust', 'leverage', 'navigate',
          'tapestry', 'testament', 'underscore', 'showcase', 'multifaceted', 'realm']
BANNED_PHRASES = ['it is important to note', 'it is worth noting', "it's worth noting",
                  'that said', 'some argue', 'on the other hand']


def toks(t, words_only=False):
    """Binary presence of word types. Presence, not counts, so length -- which is
    matched between the classes by construction -- cannot become the signal."""
    return set(re.findall(r"[a-z']+", t.lower()) if words_only
               else re.findall(r"[a-z']+|[^\sa-zA-Z0-9]", t.lower()))


def fit(X, y, l2=1.0, iters=400, lr=1.0):
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y)
        g[:-1] += l2 * w[:-1] / len(y)
        w -= lr * g
    return w


def score(X, w):
    return list(1.0 / (1.0 + np.exp(-(np.hstack([X, np.ones((len(X), 1))]) @ w))))


def run(docs, field, words_only, n_features=4000):
    tr = [d for d in docs if d['split'] == 'train']
    te = [d for d in docs if d['split'] == 'test']
    dfh, dfa = collections.Counter(), collections.Counter()
    nh = sum(1 for d in tr if d['label'] == 'human')
    na = len(tr) - nh
    for d in tr:
        (dfh if d['label'] == 'human' else dfa).update(toks(d[field], words_only))
    rows = sorted(((abs(dfh[w]/nh - dfa[w]/na), w, dfh[w]/nh, dfa[w]/na)
                   for w in set(dfh) | set(dfa) if dfh[w] + dfa[w] >= 30), reverse=True)
    vocab = [r[1] for r in rows[:n_features]]
    vi = {w: i for i, w in enumerate(vocab)}

    def X(ds):
        M = np.zeros((len(ds), len(vocab)))
        for r, d in enumerate(ds):
            for w in toks(d[field], words_only):
                if w in vi:
                    M[r, vi[w]] = 1.0
        return M

    y = np.array([1.0 if d['label'] == 'ai' else 0.0 for d in tr])
    w = fit(X(tr), y)
    p = score(X(te), w)
    lte = [d['label'] for d in te]
    return rows, binary_report(p, lte, 0.5), p, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', default=f'{EXP}/docs.jsonl')
    ap.add_argument('--field', default='text_norm', choices=['text', 'text_norm'])
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--external', default=None,
                    help='score an external jsonl (doc_id, text) with the SAME fitted '
                         'floor, to see whether the trivial model agrees with the '
                         'transformer off-distribution')
    a = ap.parse_args()
    docs = [json.loads(l) for l in open(a.docs)]
    te = [d for d in docs if d['split'] == 'test']
    th = sum(1 for d in te if d['label'] == 'human')
    ta = len(te) - th

    rows, rep, p, te_docs = run(docs, a.field, words_only=False)
    print(f'field {a.field}   test {th} human / {ta} AI\n')
    print('single features with the widest class gap (train document frequency):')
    print(f"  {'gap':>6s} {'P(human)':>9s} {'P(ai)':>7s}  feature")
    for gap, w, ph, pa in rows[:a.top]:
        print(f"  {gap:6.3f} {ph:9.3f} {pa:7.3f}  {w!r}")

    print(f"\nBAG-OF-WORDS FLOOR (words + punctuation)")
    print(f"  AUC {rep['auc']:.4f}   balanced acc {rep['balanced_acc']:.4f}   "
          f"recall_ai {rep['recall_ai']:.3f}   fpr_human {rep['fpr_human']:.3f}")
    hu = [(s, 'human') for s, d in zip(p, te_docs) if d['label'] == 'human']
    for g in sorted({d['generator'] for d in te_docs if d['label'] == 'ai'}):
        sub = [(s, 'ai') for s, d in zip(p, te_docs) if d['generator'] == g] + hu
        print(f"    {g:34s} AUC {auc([x for x, _ in sub], [l for _, l in sub]):.4f}")

    _, rep_w, _, _ = run(docs, a.field, words_only=True)
    print(f"\nsame, WORDS ONLY (every punctuation mark deleted)")
    print(f"  AUC {rep_w['auc']:.4f}   balanced acc {rep_w['balanced_acc']:.4f}")
    print("  -> punctuation is not carrying the result; ordinary vocabulary is.")

    def banned(t):
        tl = t.lower()
        return (any(re.search(rf'\b{w}\w*\b', tl) for w in BANNED)
                or any(ph in tl for ph in BANNED_PHRASES))
    bh = sum(1 for d in te if d['label'] == 'human' and banned(d[a.field]))
    ba_ = sum(1 for d in te if d['label'] == 'ai' and banned(d[a.field]))
    print(f"\nthe floor prompt's BANNED VOCABULARY, as a detector on its own")
    print(f"  human docs with >=1 banned term {bh}/{th} = {bh/th:.3f}   "
          f"AI {ba_}/{ta} = {ba_/ta:.3f}")
    print(f"  balanced accuracy of 'no banned term => AI': "
          f"{((ta-ba_)/ta + bh/th)/2:.4f}")
    print("  -> the ban is obeyed, but fires on too few human documents to carry "
          "the task.\n     The separation is NOT the prompt's blacklist.")

    if a.external:
        # refit on the full train split, then score the external file. Same
        # normalisation as training, applied here rather than assumed.
        ext = [json.loads(l) for l in open(a.external)]
        tr = [d for d in docs if d['split'] == 'train']
        dfh, dfa = collections.Counter(), collections.Counter()
        nh = sum(1 for d in tr if d['label'] == 'human'); na = len(tr) - nh
        for d in tr:
            (dfh if d['label'] == 'human' else dfa).update(toks(d[a.field]))
        rws = sorted(((abs(dfh[w]/nh - dfa[w]/na), w) for w in set(dfh) | set(dfa)
                      if dfh[w] + dfa[w] >= 30), reverse=True)
        vocab = [w for _, w in rws[:4000]]; vi = {w: i for i, w in enumerate(vocab)}
        def M(items, key):
            X = np.zeros((len(items), len(vocab)))
            for r, it in enumerate(items):
                for w in toks(key(it)):
                    if w in vi:
                        X[r, w and vi[w]] = 1.0
            return X
        norm = lambda t: re.sub(r'\s+', ' ', t).strip()
        w = fit(M(tr, lambda d: d[a.field]),
                np.array([1.0 if d['label'] == 'ai' else 0.0 for d in tr]))
        pe = score(M(ext, lambda r: norm(r['text'])), w)
        pe_arr = np.array(pe)
        print(f"\nEXTERNAL: {a.external}  n={len(ext)}")
        print(f"  bag-of-words p_ai  mean {pe_arr.mean():.4f}  median "
              f"{np.median(pe_arr):.4f}  sd {pe_arr.std(ddof=1):.4f}")
        print(f"  called AI at 0.5: {(pe_arr > 0.5).sum()}/{len(pe_arr)} = "
              f"{(pe_arr > 0.5).mean():.3f}")
        print(f"  saturated (<=0.01 or >=0.99): "
              f"{((pe_arr <= 0.01) | (pe_arr >= 0.99)).sum()}/{len(pe_arr)}")
        json.dump({r.get('doc_id', str(i)): float(v) for i, (r, v) in
                   enumerate(zip(ext, pe))}, open(a.external + '.bow.json', 'w'))
        print(f"  per-document scores -> {a.external}.bow.json")


if __name__ == '__main__':
    main()

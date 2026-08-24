"""Shared metrics. AUC is rank-based with tie correction (matches phase 1)."""

def auc(scores, labels, pos='ai'):
    """P(score(pos) > score(neg)), ties counted as 0.5."""
    y = [1 if l == pos else 0 for l in labels]
    n1, n0 = sum(y), len(y) - sum(y)
    if n1 == 0 or n0 == 0:
        return float('nan')
    pairs = sorted(zip(scores, y))
    r = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 1) / 2.0
        for k in range(i, j):
            r[k] = avg
        i = j
    s1 = sum(rk for rk, (_, yy) in zip(r, pairs) if yy == 1)
    return (s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def balanced_accuracy(scores, labels, threshold, pos='ai'):
    """(recall + specificity) / 2 at a fixed, pre-declared threshold."""
    tp = sum(1 for s, l in zip(scores, labels) if s > threshold and l == pos)
    nai = sum(1 for l in labels if l == pos)
    fp = sum(1 for s, l in zip(scores, labels) if s > threshold and l != pos)
    nhu = len(labels) - nai
    if nai == 0 or nhu == 0:
        return float('nan')
    return (tp / nai + 1 - fp / nhu) / 2.0


def stratified_auc(scores, labels, strat_values, k=3, pos='ai'):
    """AUC computed WITHIN newline-count bins only, pairs pooled across bins.

    Cross-bin pairs are deleted, so the stratifying variable carries no
    information about the label. Equal-count bins by rank of strat_values.
    Returns (auc, pairs_kept, pairs_total).
    """
    n = len(scores)
    order = sorted(range(n), key=lambda i: strat_values[i])
    bin_of = {}
    for rank, i in enumerate(order):
        bin_of[i] = min(k - 1, rank * k // n)
    conc = disc = tie = 0
    for b in range(k):
        idx = [i for i in range(n) if bin_of[i] == b]
        ai = [scores[i] for i in idx if labels[i] == pos]
        hu = [scores[i] for i in idx if labels[i] != pos]
        for a in ai:
            for h in hu:
                if a > h:
                    conc += 1
                elif a < h:
                    disc += 1
                else:
                    tie += 1
    kept = conc + disc + tie
    nai = sum(1 for l in labels if l == pos)
    total = nai * (n - nai)
    if kept == 0:
        return float('nan'), 0, total
    return (conc + 0.5 * tie) / kept, kept, total


def macro_f1(pred_idx, true_idx, n_classes):
    """Unweighted mean of per-class F1."""
    fs = []
    for c in range(n_classes):
        tp = sum(1 for p, t in zip(pred_idx, true_idx) if p == c and t == c)
        fp = sum(1 for p, t in zip(pred_idx, true_idx) if p == c and t != c)
        fn = sum(1 for p, t in zip(pred_idx, true_idx) if p != c and t == c)
        fs.append(0.0 if 2*tp + fp + fn == 0 else 2*tp / (2*tp + fp + fn))
    return sum(fs) / n_classes, fs


def confusion(pred_idx, true_idx, n_classes):
    """M[true][pred] counts."""
    m = [[0]*n_classes for _ in range(n_classes)]
    for p, t in zip(pred_idx, true_idx):
        m[t][p] += 1
    return m

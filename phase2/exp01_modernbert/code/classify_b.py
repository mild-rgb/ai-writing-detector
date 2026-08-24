#!/usr/bin/env python3
"""Apply Task B's PRE-COMMITTED reporting rule (FINDINGS.md) mechanically.

Written before any Task B number existed, so case assignment is computed, not
eyeballed. Do not edit the thresholds to fit a result.

  h    = share of the 88 held-out-generator docs assigned to HUMAN
  skew = larger generator share / (sum of both generator shares)
         0.50 = even split between the two seen generators, 1.00 = all on one

  case 1  Machine, spread   h < 0.33 and skew < 0.70   -> prediction 5 holds
  case 2  Human             h >= 0.50                  -> fingerprint hypothesis
  case 3  Confident wrong   h < 0.33 and skew >= 0.70  -> good detection, bad attribution
  case 4  Ambiguous         anything else              -> reported as ambiguous

A case is claimed ONLY if it holds in all three seeds. Otherwise: unstable.
Case 3 is NOT a confirmation of prediction 5 (satisfied on its letter, by an
unanticipated mechanism). If case 3 fires, check whether the attracting
generator is the SAME across seeds and across raw/norm; if not, it is
concentration without meaning.
"""
import json, sys, os, statistics

CI_HALF_88 = 0.098   # binomial 95% half-width at n=88, p=1/3; smallest resolvable gap

def h_and_skew(rec):
    a = rec['assignment']
    h = a['HUMAN']
    gens = {k: v for k, v in a.items() if k != 'HUMAN'}
    tot = sum(gens.values())
    top = max(gens, key=gens.get)
    skew = (gens[top] / tot) if tot > 0 else float('nan')
    return h, skew, top, gens

def case_of(h, skew):
    if h >= 0.50:                      return 2, 'Human (fingerprint hypothesis)'
    if h < 0.33 and skew < 0.70:       return 1, 'Machine, spread (pred 5 holds)'
    if h < 0.33 and skew >= 0.70:      return 3, 'Confident wrong generator'
    return 4, 'Ambiguous'

def main(path):
    R = [json.loads(l) for l in open(path)]
    B = [r for r in R if r.get('task') == 'B_unseen_assign']
    if not B:
        print('no Task B records yet'); return
    cells = {}
    for r in B:
        cells.setdefault(r['cell'], []).append(r)
    for name in sorted(cells):
        rs = sorted(cells[name], key=lambda r: r['seed'])
        print(f"\n=== {name}  ({len(rs)}/3 seeds) ===")
        cs, tops = [], []
        for r in rs:
            h, skew, top, gens = h_and_skew(r)
            c, lab = case_of(h, skew)
            cs.append(c); tops.append(top)
            g = '  '.join(f"{k.split('/')[-1]} {v:.3f}" for k, v in sorted(gens.items()))
            print(f"  seed {r['seed']}: h={h:.3f}  skew={skew:.3f}  -> case {c} ({lab})")
            print(f"           {g}   ep={r['epochs_run']:.0f}")
        if len(rs) < 3:
            print("  VERDICT: incomplete, no case claimed"); continue
        overlap_power(rs)
        if len(set(cs)) == 1:
            c = cs[0]
            print(f"  VERDICT: case {c} in all 3 seeds -> CLAIMED")
            if c == 3:
                same = len(set(tops)) == 1
                print(f"           attracting generator consistent across seeds: {same}"
                      f" ({set(t.split('/')[-1] for t in tops)})")
                print("           NOTE: case 3 is NOT a confirmation of prediction 5.")
        else:
            print(f"  VERDICT: seeds disagree {cs} -> UNSTABLE, no case claimed")

    # prediction 6: grok's h must exceed both other folds' h by > 0.098 in ALL seeds
    print("\n=== PREDICTION 6 (grok goes to HUMAN most often) ===")
    by = {}
    for r in B:
        g = r['cell'].split('|')[1]
        by.setdefault((g, r['text']), {})[r['seed']] = h_and_skew(r)[0]
    for cond in ('raw', 'norm'):
        folds = {g: v for (g, c), v in by.items() if c == cond}
        if len(folds) < 3 or any(len(v) < 3 for v in folds.values()):
            print(f"  {cond}: incomplete"); continue
        ok = all(all(folds['grok-4.6'][s] - folds[o][s] > CI_HALF_88
                     for o in folds if o != 'grok-4.6') for s in (0, 1, 2))
        print(f"  {cond}: grok h per seed {[round(folds['grok-4.6'][s],3) for s in (0,1,2)]}")
        for o in folds:
            if o != 'grok-4.6':
                print(f"        {o} h {[round(folds[o][s],3) for s in (0,1,2)]}")
        print(f"        exceeds both by >{CI_HALF_88} in all 3 seeds: {ok}"
              f" -> {'CONFIRMED' if ok else 'consistent-but-not-resolvable / not confirmed'}")

def overlap_power(rs):
    """Can the seed-overlap statistic resolve a stable hard core? Compute the
    independence EXPECTATION before interpreting the observation.

    Added after a reasoning error: observing 0 triple-overlaps was reported as
    evidence against a stable subset, when expected-under-independence was 0.008
    -- i.e. the test had no power and 0 was what independence predicted.
    """
    cls = rs[0]['classes']; hi = cls.index('HUMAN'); N = rs[0]['n_test']
    sets = []
    for r in rs:
        sets.append({d for d, p in r['probs'].items()
                     if max(range(len(p)), key=lambda i: p[i]) == hi})
    k = [len(x) for x in sets]
    e3 = N * (k[0]/N) * (k[1]/N) * (k[2]/N) if N else 0
    e2 = sum(N*(k[i]/N)*(k[j]/N) for i, j in ((0,1),(0,2),(1,2))) - 3*e3
    obs3 = len(set.intersection(*sets))
    union = set().union(*sets)
    obs2 = sum(1 for d in union if sum(d in x for x in sets) >= 2)
    print(f"  overlap check: counts {k}  distinct {len(union)}")
    print(f"    triple: observed {obs3}, expected under independence {e3:.3f}")
    print(f"    pair  : observed {obs2}, expected under independence {e2:.2f}")
    # A statistic is informative if the OBSERVATION is unlikely under independence,
    # not merely if its expectation is large. Checking only the pair statistic once
    # nearly buried a triple-overlap observed against an expectation of 0.006.
    pair_signal = obs2 > 0 and obs2 >= 2 * max(e2, 1e-9)
    trip_signal = obs3 > 0 and obs3 >= 2 * max(e3, 1e-9)
    if not pair_signal and not trip_signal:
        print(f"    -> UNINFORMATIVE: neither statistic departs from independence "
              f"(pair {obs2} vs {e2:.2f}, triple {obs3} vs {e3:.3f}).")
    else:
        if trip_signal:
            print(f"    -> TRIPLE departs from independence: {obs3} observed vs "
                  f"{e3:.3f} expected. Run a Monte Carlo p-value and apply "
                  f"multiplicity correction across cells before claiming anything.")
        if pair_signal:
            print(f"    -> PAIR departs from independence: {obs2} observed vs "
                  f"{e2:.2f} expected. Same caveats.")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/results.jsonl')

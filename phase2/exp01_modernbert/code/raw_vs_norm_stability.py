#!/usr/bin/env python3
"""Does normalisation stabilise? Compute over EVERY comparable raw/norm pair.

Written after reporting "normalisation stabilises" as a general finding when it
held for Task A accuracy and one of two Task B folds. The failure mode is that
the comparator which comes to mind is the one that made the contrast salient.
So this enumerates all pairs and lets the output decide.
"""
import json, statistics, sys, os

d = os.path.dirname(os.path.abspath(__file__)) + '/../results'
R = [json.loads(l) for l in open(f'{d}/results.jsonl')]

def rng(v):
    return max(v) - min(v)

rows = []
# Task A: accuracy
A = {c: [r for r in R if r['cell'] == c] for c in ('4way|raw', '4way|norm')}
if all(len(v) == 3 for v in A.values()):
    rows.append(('Task A accuracy', rng([r['accuracy'] for r in A['4way|raw']]),
                 rng([r['accuracy'] for r in A['4way|norm']])))
    rows.append(('Task A macro-F1', rng([r['macro_f1'] for r in A['4way|raw']]),
                 rng([r['macro_f1'] for r in A['4way|norm']])))

# Task B: skew and h, per held-out generator
B = [r for r in R if r.get('task') == 'B_unseen_assign']
def skew(r):
    o = {k: v for k, v in r['assignment'].items() if k != 'HUMAN'}
    t = sum(o.values())
    return max(o.values()) / t if t else float('nan')

for g in sorted({r['cell'].split('|')[1] for r in B}):
    pair = {}
    for cond in ('raw', 'norm'):
        rs = [r for r in B if r['cell'] == f'assign|{g}|{cond}']
        if len(rs) == 3:
            pair[cond] = rs
    if len(pair) == 2:
        rows.append((f'Task B skew, {g}',
                     rng([skew(r) for r in pair['raw']]), rng([skew(r) for r in pair['norm']])))
        rows.append((f'Task B h,    {g}',
                     rng([r['human_share'] for r in pair['raw']]),
                     rng([r['human_share'] for r in pair['norm']])))

print(f"{'quantity':28s} {'raw range':>10} {'norm range':>11} {'effect':>22}")
helped = hurt = same = 0
for name, r_, n_ in rows:
    if n_ < r_ / 2:
        v = f'norm MORE stable {r_/max(n_,1e-9):.0f}x'; helped += 1
    elif n_ > r_ * 2:
        v = f'norm LESS stable {n_/max(r_,1e-9):.0f}x'; hurt += 1
    else:
        v = 'no clear change'; same += 1
    print(f"{name:28s} {r_:>10.4f} {n_:>11.4f} {v:>22}")
print(f"\n  norm more stable: {helped}   less stable: {hurt}   unchanged: {same}")
print("  -> 'normalisation stabilises' is NOT general; state it per-quantity.")

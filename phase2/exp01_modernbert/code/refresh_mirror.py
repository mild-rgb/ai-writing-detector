#!/usr/bin/env python3
"""Regenerate results/run_count.txt from results.jsonl. Run after every pull.

Exists because run_count.txt went stale once (said "1 of 60" while the file held
two records) and it is the first file a reader checks. Derives everything from
the data so it cannot disagree with results.jsonl again.
"""
import json, datetime, sys, os

EXPECTED = {'A_4way': 2*3, 'B_unseen_assign': 6*3}   # Task C was CUT by the user
TOTAL = sum(EXPECTED.values())                        # 24, not 60

d = os.path.dirname(os.path.abspath(__file__)) + '/../results'
R = [json.loads(l) for l in open(f'{d}/results.jsonl')]
by_task = {}
for r in R:
    t = r.get('task', 'C_binary' if '|seen|' in r['cell'] or '|unseen|' in r['cell'] else '?')
    by_task.setdefault(t, []).append(r)

lines = [
    f'snapshot: {datetime.datetime.now().isoformat(timespec="seconds")}',
    f'runs: {len(R)} of {TOTAL}  ->  {"COMPLETE" if len(R) >= TOTAL else "PARTIAL"}',
    '',
    'SCOPE: Task C was CUT by the user ("just do A and B"). Denominator is 24, not 60.',
    'Task C runs are 25-60 in the runner order, so C cannot start before B finishes.',
    f'STOP POINT: stop the sweep once {TOTAL} records covering A and B are present.',
    '',
]
for t, exp in EXPECTED.items():
    got = len(by_task.get(t, []))
    lines.append(f'  {t:18s} {got:>2} / {exp}  {"complete" if got >= exp else "in progress"}')
extra = [t for t in by_task if t not in EXPECTED]
if extra:
    lines.append(f'  UNEXPECTED task records present: {extra}')
lines += ['', 'records:']
for r in sorted(R, key=lambda r: (r['cell'], r['seed'])):
    stop = 'CAP' if round(r['epochs_run']) >= r['hp']['max_epochs'] else 'early-stop'
    lines.append(f"  {r['cell']:34s} seed={r['seed']}  ep={r['epochs_run']:.0f} ({stop})")
lines += ['',
 'NOTE: 4way|raw seed=0 was re-run under gradient checkpointing. The pre-checkpointing',
 '      record is archived in superseded_run1_no_gradckpt.jsonl and is NOT counted here.',
 '      Its probs are 6dp-rounded; read its pairwise AUC from the pairwise_auc field.']
open(f'{d}/run_count.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

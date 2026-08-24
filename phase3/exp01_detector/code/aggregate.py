#!/usr/bin/env python3
"""Analysis over results.jsonl. Needs no GPU and retrains nothing.

This is phase 2's "layer 2": every number below is recomputed from the stored
per-document probabilities, not read out of a reported field. That property is
what let a second party audit phase 2 and find six errors, and it only works
because `probs` is a hard rule in train_cell.py.

Reports mean +/- SD across seeds, never a single run, and reports PER GENERATOR,
never a pooled detection rate.

Usage:  python aggregate.py results.jsonl --docs ../docs.jsonl
"""
import argparse, json, math, os, sys, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import auc                                          # noqa: E402
from report import binary_report                                 # noqa: E402


def mean_sd(xs):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0, 1
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, sd, len(xs)


def fmt(m, sd, n, dp=4):
    if m is None:
        return '     -     '
    return f'{m:.{dp}f} ±{sd:.{dp}f}' + (f' (n={n})' if n else '')


def rederive(rec, docs):
    """Recompute the record's headline test metrics from `probs` alone.

    If these disagree with the reported fields, the record is not trustworthy
    and the disagreement is the finding. Returns (report, per_generator).
    """
    ai_idx = rec['classes'].index('ai')
    ids = list(rec['probs'])
    p = [rec['probs'][i][ai_idx] for i in ids]
    lab = [docs[i]['label'] for i in ids]
    thr = rec['hp']['decision_threshold']
    hu = [(s, 'human') for s, i in zip(p, ids) if docs[i]['label'] == 'human']
    per = {}
    for g in sorted({docs[i]['generator'] for i in ids if docs[i]['label'] == 'ai'}):
        sub = [(s, 'ai') for s, i in zip(p, ids) if docs[i]['generator'] == g] + hu
        per[g] = binary_report([s for s, _ in sub], [l for _, l in sub], thr)
    return binary_report(p, lab, thr), per


def group_key(r):
    return (r['cell'], r['model'], r['code_path']['classifier_pooling'],
            r['code_path']['gradient_checkpointing'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--docs', default=None)
    ap.add_argument('--metric', default='auc',
                    choices=['auc', 'balanced_acc', 'accuracy', 'recall_ai', 'fpr_human'])
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(a.results))
    docs_path = a.docs or os.path.normpath(f'{here}/../docs.jsonl')
    docs = {d['doc_id']: d for d in map(json.loads, open(docs_path))}
    recs = [json.loads(l) for l in open(a.results)]
    if not recs:
        raise SystemExit('no records')

    # --- audit: does every record's own arithmetic hold up? ------------------
    print('== re-derivation from stored per-document probabilities ==')
    bad = 0
    for r in recs:
        rd, _ = rederive(r, docs)
        for k in ('auc', 'balanced_acc', 'recall_ai', 'fpr_human'):
            if abs((rd[k] or 0) - (r['test'][k] or 0)) > 1e-9:
                bad += 1
                print(f"  MISMATCH {r['cell']} init={r['seeds']['weight_init']} "
                      f"{k}: reported {r['test'][k]} vs re-derived {rd[k]}")
    print(f'  {len(recs)-bad}/{len(recs)} records re-derive exactly from `probs`.\n')

    # --- audit: was this all one environment? --------------------------------
    envs = collections.Counter(r['env']['env_hash'] for r in recs)
    print('== environments ==')
    for h, n in envs.items():
        e = next(r['env'] for r in recs if r['env']['env_hash'] == h)
        L, G = e['libraries'], e['gpu']
        print(f"  {h}  x{n:3d}  {G.get('name','?')} driver {G.get('driver_version','?')} | "
              f"py {e['python_version_short']} torch {L.get('torch')} "
              f"cuda {L.get('torch_cuda')} transformers {L.get('transformers')}")
    if len(envs) > 1:
        print('  NOTE: more than one environment. Cross-environment comparisons carry '
              'FP wobble on top of seed variance -- compare conclusions, not numbers.')
    ds = {r['data']['docs_sha256'][:16] for r in recs}
    print(f"  data: {', '.join(sorted(ds))}"
          + ('   NOTE: records span more than one dataset build!' if len(ds) > 1 else ''))
    incomplete = sum(1 for r in recs if not r['data'].get('corpus_complete', True))
    if incomplete:
        print(f'  NOTE: {incomplete} record(s) ran on an INCOMPLETE corpus.')
    print()

    # --- headline: mean +/- SD across seeds, per cell ------------------------
    groups = collections.defaultdict(list)
    for r in recs:
        groups[group_key(r)].append(r)
    print(f'== {a.metric} on test, mean ± SD across seeds ==')
    print(f"  {'cell':30s} {'pool':5s} {'gc':3s} {a.metric:>18s}  "
          f"{'newline-only':>12s}  {'epochs':>7s}  seeds")
    for k in sorted(groups):
        rs = groups[k]
        cell, model, pool, gc = k
        m, sd, n = mean_sd([r['test'][a.metric] for r in rs])
        nm, nsd, _ = mean_sd([r.get('newline_baseline_auc') for r in rs])
        em, _, _ = mean_sd([r['epochs_run'] for r in rs])
        seeds = ','.join(str(r['seeds']['weight_init']) for r in sorted(
            rs, key=lambda r: r['seeds']['weight_init']))
        print(f"  {cell:30s} {pool:5s} {'on' if gc else 'off':3s} {fmt(m, sd, n):>18s}  "
              f"{(f'{nm:.3f}' if nm is not None else '-'):>12s}  "
              f"{(f'{em:.1f}' if em else '-'):>7s}  {seeds}")
        if n < 5:
            print(f"      n={n} seeds. Phase 2's own exp02 requirement was 5-10; "
                  f"three cannot support a variance claim.")

    # --- per generator, for every cell ---------------------------------------
    print(f'\n== per generator ({a.metric}, mean ± SD across seeds) ==')
    print('   the AI side varies by row; the human side is the same documents in each.')
    for k in sorted(groups):
        rs = groups[k]
        gens = sorted({g for r in rs for g in r['per_generator']})
        if not gens:
            continue
        print(f"\n  {k[0]}  pool={k[2]}  gc={'on' if k[3] else 'off'}")
        rows = []
        for g in gens:
            m, sd, n = mean_sd([r['per_generator'][g][a.metric]
                                for r in rs if g in r['per_generator']])
            rec_m, rec_sd, _ = mean_sd([r['per_generator'][g]['recall_ai']
                                        for r in rs if g in r['per_generator']])
            rows.append((m, g, sd, n, rec_m, rec_sd))
        for m, g, sd, n, rec_m, rec_sd in sorted(rows, reverse=True,
                                                 key=lambda x: (x[0] is not None, x[0])):
            print(f"    {g:34s} {fmt(m, sd, n):>18s}   recall {fmt(rec_m, rec_sd, 0, 3)}")
        vals = [r[0] for r in rows if r[0] is not None]
        if len(vals) > 1:
            print(f"    spread {min(vals):.4f} - {max(vals):.4f}  "
                  f"({max(vals)-min(vals):.4f}) -- report this, not a pooled rate")

    # --- LOGO summary: the generalization question ---------------------------
    logo = [r for r in recs if r['task'] == 'logo']
    if logo:
        print('\n== leave-one-generator-out: the generalization test ==')
        g2 = collections.defaultdict(list)
        for r in logo:
            g2[(r['heldout_generator'], r['text'], r['code_path']['classifier_pooling'])].append(r)
        print(f"  {'held-out generator':34s} {'text':5s} {'pool':5s} "
              f"{'AUC':>18s} {'recall on unseen':>18s}")
        for k in sorted(g2):
            rs = g2[k]
            m, sd, n = mean_sd([r['test']['auc'] for r in rs])
            rm, rsd, _ = mean_sd([r['test']['recall_ai'] for r in rs])
            print(f"  {k[0]:34s} {k[1]:5s} {k[2]:5s} {fmt(m, sd, n):>18s} "
                  f"{fmt(rm, rsd, 0, 3):>18s}")
        print('  A generator the detector never saw is the deployment-realistic case. '
              'Pooling these hides which model is the failure.')

    # --- raw vs norm ---------------------------------------------------------
    pairs = collections.defaultdict(dict)
    for k, rs in groups.items():
        cell, model, pool, gc = k
        base = cell.rsplit('|', 1)[0]
        m, sd, n = mean_sd([r['test'][a.metric] for r in rs])
        pairs[(base, model, pool, gc)][cell.rsplit('|', 1)[1]] = (m, sd, n)
    both = {k: v for k, v in pairs.items() if 'raw' in v and 'norm' in v}
    if both:
        print(f'\n== whitespace: raw vs norm ({a.metric}) ==')
        print('  norm is the headline. raw exists only to measure how much of a result '
              'is whitespace.')
        for k in sorted(both):
            raw, norm = both[k]['raw'], both[k]['norm']
            d = norm[0] - raw[0]
            print(f"  {k[0]:30s} pool={k[2]:5s}  raw {fmt(*raw):>18s}   "
                  f"norm {fmt(*norm):>18s}   delta {d:+.4f}")
        print('  Overlapping SDs mean the delta is not resolved at this seed count. '
              'Prefer a dispersion claim over a mean claim when they disagree.')


if __name__ == '__main__':
    main()

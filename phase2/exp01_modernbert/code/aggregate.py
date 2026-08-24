"""Aggregate results.jsonl into the report tables DESIGN.md asks for.

Reports mean and SPREAD (max-min across seeds) everywhere, plus the stopping
epoch per cell, and marks any cell with fewer than the expected seeds PARTIAL.
"""
import json, sys, statistics, itertools, argparse, os

SHORT = lambda c: c.split('/')[-1] if '/' in c else c

def spread(v):
    return (max(v) - min(v)) if len(v) > 1 else 0.0

def fmt(v, nd=3):
    if not v: return 'n/a'
    m = statistics.mean(v)
    return f"{m:.{nd}f}" + (f" ±{spread(v)/2:.{nd}f}" if len(v) > 1 else " (1 seed)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results.jsonl')
    ap.add_argument('--frozen', default='frozen_control.json')
    ap.add_argument('--splits', default='splits.json')
    ap.add_argument('--docs', default='docs.jsonl')
    ap.add_argument('--expect-seeds', type=int, default=3)
    a = ap.parse_args()

    R = [json.loads(l) for l in open(a.results)]
    by = {}
    for r in R:
        by.setdefault((r['model'], r['cell']), []).append(r)
    models = sorted({r['model'] for r in R})
    docs = {d['doc_id']: d for d in map(json.loads, open(a.docs))}
    S = json.load(open(a.splits))
    FZ = json.load(open(a.frozen)) if os.path.exists(a.frozen) else None

    for MODEL in models:
        print('=' * 78)
        print(f"MODEL: {MODEL}")
        print('=' * 78)
        sel = {k[1]: v for k, v in by.items() if k[0] == MODEL}

        # ---------------- Task A ----------------
        print("\n### TASK A — 4-way\n")
        for cellname in ('4way|raw', '4way|norm'):
            rs = sel.get(cellname)
            if not rs: continue
            tag = '' if len(rs) >= a.expect_seeds else f'  [PARTIAL {len(rs)}/{a.expect_seeds} seeds]'
            print(f"-- {cellname}{tag}   epochs {[round(r['epochs_run']) for r in rs]}")
            print(f"   accuracy {fmt([r['accuracy'] for r in rs])}   "
                  f"macro-F1 {fmt([r['macro_f1'] for r in rs])}   (chance 0.250)")
            cls = rs[0]['classes']
            print("   per-class F1: " + "  ".join(
                f"{SHORT(c)} {fmt([r['per_class_f1'][c] for r in rs])}" for c in cls))
            n = len(cls)
            M = [[statistics.mean([r['confusion'][i][j] for r in rs]) for j in range(n)] for i in range(n)]
            print(f"\n   confusion (mean over seeds), rows=TRUE cols=PRED, 88 per row:")
            print("        " + "".join(f"{SHORT(c)[:11]:>12s}" for c in cls))
            for i, c in enumerate(cls):
                print(f"   {SHORT(c)[:11]:>11s} " + "".join(f"{M[i][j]:>12.1f}" for j in range(n)))
            print("\n   pairwise AUC (separability of each class pair):")
            for k in rs[0]['pairwise_auc']:
                c1, c2 = k.split('|')
                v = [r['pairwise_auc'][k] for r in rs]
                flag = '   <-- near-indistinguishable' if statistics.mean(v) < 0.65 else ''
                print(f"     {SHORT(c1):>16s} vs {SHORT(c2):<16s} {fmt(v)}{flag}")
            print()

        # ---------------- Task B ----------------
        print("\n### TASK B — unseen-generator assignment\n")
        bcells = sorted(c for c in sel if c.startswith('assign|'))
        if not bcells:
            print("   (no Task B runs recorded yet)")
        print(f"{'cell':32s} {'->HUMAN':>16s} {'->other gens':>16s}   epochs"
              if bcells else "", end='' if not bcells else '\n')
        for cellname in sorted(c for c in sel if c.startswith('assign|')):
            rs = sel[cellname]
            tag = '' if len(rs) >= a.expect_seeds else f' [PARTIAL {len(rs)}]'
            hs = [r['human_share'] for r in rs]
            print(f"{cellname+tag:32s} {fmt(hs):>16s} {fmt([1-x for x in hs]):>16s}   "
                  f"{[round(r['epochs_run']) for r in rs]}")
        for cellname in sorted(c for c in sel if c.startswith('assign|')):
            rs = sel[cellname]
            cls = rs[0]['classes']
            print(f"   {cellname}: " + "  ".join(
                f"{SHORT(c)} {statistics.mean([r['assignment'][c] for r in rs]):.3f}" for c in cls))

        # ---------------- Task C ----------------
        print("\n\n### TASK C — binary\n")
        WCL = {'grok-4.6': 0.887, 'deepseek-v4-pro': 0.749, 'qwen3.8-max': 0.585}
        ccells = sorted(c for c in sel if '|seen|' in c or '|unseen|' in c)
        if not ccells:
            print("   (no Task C runs recorded yet)")
        else:
            print(f"{'cell':32s} {'AUC':>16s} {'balAcc':>16s} {'wc -l':>7} {'LFM2.5':>7}  epochs")
        for cellname in sorted(c for c in sel if '|seen|' in c or '|unseen|' in c):
            rs = sel[cellname]
            tag = '' if len(rs) >= a.expect_seeds else f' [PARTIAL {len(rs)}]'
            g, cond, txt = cellname.split('|')
            wc = WCL[g] if txt == 'raw' else 0.500
            fz = FZ['cells'].get(cellname, {}).get('auc') if FZ else None
            print(f"{cellname+tag:32s} {fmt([r['auc'] for r in rs]):>16s} "
                  f"{fmt([r['balanced_acc'] for r in rs]):>16s} {wc:>7.3f} "
                  f"{(f'{fz:.3f}' if fz is not None else '  n/a'):>7}  "
                  f"{[round(r['epochs_run']) for r in rs]}")

        print("\n--- HEADLINE: norm|unseen 3-fold ---")
        folds = {}
        _hdr_shown = False
        for cellname in sel:
            if cellname.endswith('|unseen|norm'):
                folds[cellname.split('|')[0]] = [r['auc'] for r in sel[cellname]]
        if not folds:
            print("   (no norm|unseen runs recorded yet - headline not computable)")
        if folds:
            per = {g: statistics.mean(v) for g, v in folds.items()}
            allv = [x for v in folds.values() for x in v]
            print("   per fold: " + "   ".join(f"{g} {fmt(folds[g])}" for g in sorted(folds)))
            print(f"   3-fold mean {statistics.mean(list(per.values())):.3f}   "
                  f"across-fold spread {max(per.values())-min(per.values()):.3f}   "
                  f"across-run spread {max(allv)-min(allv):.3f}")
            print(f"   hardest fold: {min(per, key=per.get)}")
            if FZ:
                fzn = [FZ['cells'][c]['auc'] for c in FZ['cells'] if c.endswith('|unseen|norm')]
                if fzn: print(f"   LFM2.5 zero-shot on same cells: {statistics.mean(fzn):.3f}")
            print(f"   wc -l on same cells: 0.500 (by construction)")
            m = statistics.mean(list(per.values()))
            print(f"   >>> vs 0.771 LFM2.5 long3: {'BELOW (pred 10 holds)' if m<0.771 else 'ABOVE (pred 10 WRONG)'}")
            print(f"   >>> vs 0.65 falsification line: {'BELOW — FALSIFIED' if m<0.65 else 'above'}")

if __name__ == '__main__':
    main()

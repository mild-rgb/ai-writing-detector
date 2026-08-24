"""The trivial baseline, computed beside every detector number.

phase1/NARRATIVE.md method note 19: `wc -l` scores AUC 0.76-0.86 on every set
in that repo and, on a threshold fitted in place, balanced accuracy 74.0%
against Haiku 4.5's 57.3% -- and no detector figure in phase 1 was ever quoted
against any baseline at all. A detector result without the dumbest possible
baseline on the same documents is not a result, it is a number.

This computes the dumb baselines for a set of generated answers against the
human answers to the same questions: newline count, paragraph count, mean words
per paragraph, word count. For each it reports the rank AUC with a bootstrap
interval, and the balanced accuracy of the single best fitted threshold.

Three repo lessons are wired in as behaviour, not as comments:

  note 17 -- print distinct-document counts next to row counts, and dedupe
     sources on (id, model): a restart that appended instead of resuming left
     65 duplicate pairs in phase 1 and three reported figures were computed on
     fewer documents than they claimed.
  note 23 -- resample before calling anything unstable. Every per-model cell
     here carries a bootstrap interval, because a 25-document draw has a
     sampling SD around 0.04 and two such draws differ by 0.116 or more about
     4% of the time.
  note 16 -- fail loudly. No bare except, no parse-and-skip, and an ai row
     whose question has no human answer is an error rather than a silent drop.

Usage:
    python3 phase3/scripts/baseline_surface.py ANSWERS.jsonl [MORE.jsonl ...]
        [--humans FILE ...]     human corpora (default: phase1 short + long)
        [--all-humans]          compare against every human answer in the
                                corpora rather than only the questions the ai
                                set actually covers (the default is matched,
                                because swapping the human comparison set moved
                                a phase-1 figure by 0.031 on its own)
        [--boot N]              bootstrap resamples, default 2000, 0 to skip
        [--json OUT]            also write the numbers as JSON

ANSWERS rows need `id`, `model`, `text`. Human rows need `id`, `human_answer`.

Reproduction check, run on phase1/study/phase1/v6/answers.jsonl (300 rows, the
fixed 100-question dev set) against its matched humans -- oriented newline
separability by generator, this script against NARRATIVE.md section 15's `short`
row, which is a different draw of v6 documents:

    generator          here    section 15 (`short`)
    x-ai/grok-4.6      0.854   0.848
    deepseek-v4-pro    0.841   0.800
    qwen3.8-max        0.625   0.631
    pooled             0.773   0.760

Same ordering, same magnitudes, and word count comes out at 0.49 pooled because
generation pins it to the specific human answer. Two independent draws with a
per-draw SD around 0.04, so this is a consistency check and not a replication.
"""
import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_HUMANS = [
    os.path.join(ROOT, "phase1/data/interim/questions_1000.jsonl"),
    os.path.join(ROOT, "phase1/data/interim/questions_longform.jsonl"),
]
BLANK = re.compile(r"\n\s*\n")


# --- features -------------------------------------------------------------
# Every one of these is computable with a shell one-liner. That is the point.

def features(text):
    paras = [p for p in BLANK.split(text.strip()) if p.strip()]
    words = len(text.split())
    return {
        "newlines": text.count("\n"),
        "paragraphs": len(paras),
        "words_per_para": words / len(paras) if paras else 0.0,
        "words": words,
    }


FEATURES = ["newlines", "paragraphs", "words_per_para", "words"]


# --- statistics -----------------------------------------------------------

def auc(pos, neg):
    """P(pos > neg) + 0.5 P(pos == neg), by ranks. Ties matter here: newline
    counts are small integers and a naive comparison would inflate every cell."""
    if not pos or not neg:
        return float("nan")
    merged = sorted([(v, 0) for v in pos] + [(v, 1) for v in neg])
    ranks = {}
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_, cls) in enumerate(merged) if cls == 0)
    n1, n2 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n2)


def boot_auc(pos, neg, b, seed=20260823):
    """Percentile interval over independent resamples of each class."""
    if b <= 0 or not pos or not neg:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    vals = []
    for _ in range(b):
        p = [pos[rnd.randrange(len(pos))] for _ in range(len(pos))]
        n = [neg[rnd.randrange(len(neg))] for _ in range(len(neg))]
        vals.append(auc(p, n))
    vals.sort()
    return (vals[int(0.025 * b)], vals[min(b - 1, int(0.975 * b))])


def best_threshold(pos, neg):
    """The single best fitted rule, both directions. Returns (rule, detection,
    fpr, balanced). Fitted in place on the same documents it scores -- one
    integer, close to free, but fitted, and it must be quoted as such."""
    cands = sorted({v for v in pos + neg})
    mids = [(a + b) / 2 for a, b in zip(cands, cands[1:])] or list(cands)
    best = None
    for t in mids:
        for op in ("<=", ">="):
            if op == "<=":
                det = sum(1 for v in pos if v <= t) / len(pos)
                fpr = sum(1 for v in neg if v <= t) / len(neg)
            else:
                det = sum(1 for v in pos if v >= t) / len(pos)
                fpr = sum(1 for v in neg if v >= t) / len(neg)
            ba = (det + (1 - fpr)) / 2
            if best is None or ba > best[0]:
                best = (ba, f"{op} {t:g}", det, fpr)
    ba, rule, det, fpr = best
    return rule, det, fpr, ba


def best_interval(pos, neg, max_cuts=80):
    """The best rule of the form `a <= x <= b means ai`, and its complement.

    A one-sided threshold is a monotone rule, and rank AUC only sees monotone
    separability. That is fine for one generator, whose paragraph habit sits on
    one side of the human median. It is NOT fine for a blend: mix a generator
    that writes above the human newline median with one that writes below and
    the pooled AUC drifts to 0.5 while both classes stay perfectly separable to
    a rule with two cuts. Pooled AUC near chance is necessary, not sufficient,
    and this is the statistic that says which one you have.

    Infinite cuts are included, so this family contains every one-sided rule and
    its balanced accuracy is always at least the one-sided figure.
    """
    vals = sorted({v for v in pos + neg})
    cuts = [(a + b) / 2 for a, b in zip(vals, vals[1:])]
    if len(cuts) > max_cuts:
        step = len(cuts) / max_cuts
        cuts = [cuts[int(i * step)] for i in range(max_cuts)]
    cuts = [float("-inf")] + cuts + [float("inf")]
    best = None
    for i, lo in enumerate(cuts):
        for hi in cuts[i + 1:]:
            det = sum(1 for v in pos if lo <= v <= hi) / len(pos)
            fpr = sum(1 for v in neg if lo <= v <= hi) / len(neg)
            for inside in (True, False):
                d, f = (det, fpr) if inside else (1 - det, 1 - fpr)
                ba = (d + (1 - f)) / 2
                if best is None or ba > best[0]:
                    a_ = "-inf" if lo == float("-inf") else f"{lo:g}"
                    b_ = "inf" if hi == float("inf") else f"{hi:g}"
                    rule = (f"in [{a_}, {b_}]" if inside
                            else f"outside [{a_}, {b_}]")
                    best = (ba, rule, d, f)
    ba, rule, det, fpr = best
    return rule, det, fpr, ba


# --- loading --------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"FATAL {path}:{lineno}: {e}. Not skipping it -- three "
                         f"phase-1 figures were computed on fewer items than "
                         f"they claimed because a scorer skipped a bad line.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--humans", nargs="*", default=DEFAULT_HUMANS)
    ap.add_argument("--all-humans", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--json", dest="json_out")
    a = ap.parse_args()

    humans = {}
    for path in a.humans:
        if not os.path.exists(path):
            sys.exit(f"FATAL: human corpus not found: {path}")
        for r in load_jsonl(path):
            humans[r["id"]] = r["human_answer"]

    # dedupe on (id, model), last row wins
    pool, dup = {}, 0
    for path in a.answers:
        if not os.path.exists(path):
            sys.exit(f"FATAL: answers file not found: {path}")
        for r in load_jsonl(path):
            for k in ("id", "model", "text"):
                if k not in r:
                    sys.exit(f"FATAL {path}: row missing '{k}': {list(r)}")
            key = (r["id"], r["model"])
            if key in pool:
                dup += 1
            pool[key] = r
    if not pool:
        sys.exit("FATAL: no ai rows loaded")

    missing = sorted({qid for qid, _ in pool if qid not in humans})
    if missing:
        sys.exit(f"FATAL: {len(missing)} ai rows have no human answer for their "
                 f"question (first: {missing[:3]}). Refusing to score against a "
                 f"human set that does not match the ai set.")

    ai_ids = {qid for qid, _ in pool}
    hu_ids = sorted(humans) if a.all_humans else sorted(ai_ids)

    ai_rows = [{"id": qid, "model": m, **features(r["text"])}
               for (qid, m), r in sorted(pool.items())]
    hu_rows = [{"id": qid, **features(humans[qid])} for qid in hu_ids]

    models = sorted({r["model"] for r in ai_rows})
    print(f"ai: {len(ai_rows)} rows over {len(ai_ids)} questions and "
          f"{len(models)} models" + (f"  ({dup} duplicate (id, model) rows "
          f"collapsed)" if dup else ""))
    print(f"human: {len(hu_rows)} documents "
          f"({'full corpora' if a.all_humans else 'matched to the ai question set'})")
    if a.all_humans and len(a.humans) > 1:
        print("  WARNING: --all-humans over more than one corpus pools short- "
              "and long-form humans,\n  whose paragraph habits differ by more "
              "than the class gap being measured. Pass a\n  single --humans "
              "file, or drop --all-humans.")
    print()

    out = {"ai_rows": len(ai_rows), "ai_questions": len(ai_ids),
           "human_docs": len(hu_rows), "duplicates_collapsed": dup,
           "matched_humans": not a.all_humans, "cells": []}

    for feat in FEATURES:
        neg = [r[feat] for r in hu_rows]
        print(f"== {feat} ==   human median {median(neg):g}")
        print(f"{'generator':<24} {'n':>4} {'median':>7} {'AUC':>6} "
              f"{'95% CI':>16} {'sep':>6}   {'best fitted rule':<18} "
              f"{'det':>6} {'fpr':>6} {'bal':>6}")
        seps = {}
        rows = [(m, [r[feat] for r in ai_rows if r["model"] == m]) for m in models]
        if len(models) > 1:
            rows.append(("POOLED", [r[feat] for r in ai_rows]))
        for name, pos in rows:
            if not pos:
                continue
            u = auc(pos, neg)
            lo, hi = boot_auc(pos, neg, a.boot)
            rule, det, fpr, ba = best_threshold(pos, neg)
            irule, idet, ifpr, iba = best_interval(pos, neg)
            sep = max(u, 1 - u)
            flag = "" if sep < 0.70 else ("  <-- separable" if sep < 0.85
                                          else "  <-- SEPARABLE")
            print(f"{name:<24} {len(pos):>4} {median(pos):>7g} {u:>6.3f} "
                  f"[{lo:>5.3f}, {hi:>5.3f}] {sep:>6.3f}   {feat} {rule:<12} "
                  f"{100*det:>5.1f}% {100*fpr:>5.1f}% {100*ba:>5.1f}%{flag}")
            seps[name] = sep
            out["cells"].append({"feature": feat, "generator": name,
                                 "n": len(pos), "auc": u, "ci": [lo, hi],
                                 "separability": sep,
                                 "rule": f"{feat} {rule}", "detection": det,
                                 "fpr": fpr, "balanced": ba,
                                 "interval_rule": f"{feat} {irule}",
                                 "interval_detection": idet,
                                 "interval_fpr": ifpr,
                                 "interval_balanced": iba})
            if iba - ba > 0.03:
                print(f"{'':<24} {'':>4} {'':>7} {'':>6} {'':>16} {'':>6}   "
                      f"{feat} {irule:<12} {100*idet:>5.1f}% {100*ifpr:>5.1f}% "
                      f"{100*iba:>5.1f}%  <-- two-sided rule beats the "
                      f"one-sided one by {100*(iba-ba):.1f} pts")
        per_model = [v for k, v in seps.items() if k != "POOLED"]
        if "POOLED" in seps and per_model and seps["POOLED"] < 0.60 <= max(per_model):
            worst = max(seps, key=lambda k: seps[k] if k != "POOLED" else 0)
            print(f"  WARNING: pooled separability is {seps['POOLED']:.3f} but "
                  f"{worst} alone is {seps[worst]:.3f}. A blend whose members sit "
                  f"on\n  opposite sides of the human median cancels in the "
                  f"pooled rank statistic while staying\n  separable per model "
                  f"and to a two-sided rule. Match the feature per generator; do "
                  f"not\n  read the pooled number as the fix.")
        print()

    print("AUC is P(ai scores higher), so a value BELOW 0.5 means the ai class "
          "has fewer of the\nfeature, not that it is harder to detect. `sep` is "
          "max(auc, 1-auc), the orientation\nphase 1 quotes -- compare that "
          "column, not AUC, against NARRATIVE.md section 15. "
          "The threshold is fitted on the same\ndocuments it scores -- one "
          "integer, close to free, but fitted, and an operating point\nis not "
          "length-portable even when the AUC is.")

    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\n-> {a.json_out}")


def median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


if __name__ == "__main__":
    main()

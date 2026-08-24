"""Marker rates for both classes, with intervals, and the gap between them.

phase 1 method note 9: check your own output for the defect you filtered out of
theirs. The inversion prompt planted `Edit:` in 100% of documents against a
human 11% and nobody ran the scan until after the number was published. This
runs it on both classes at once.

Every marker here is mechanical -- a regex, not a judgement -- so the rate is
reproducible and an AI-vs-human gap is a fact about the corpus rather than a
reading of it. Rates are reported with Wilson intervals and a two-proportion
z-test against the human rate, because the deliverable is "matched to the human
base rate", and matched means inside the interval, not equal to the point
estimate.

    python3 phase3/scripts/markers.py --human-ids dev bench --ai ANSWERS.jsonl

With no --ai it just prints the human base rates, which is what you need before
you can set a draw rate to match them.
"""
import argparse
import collections
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (family, name, test) -- test takes the text and returns True if present.
def _re(pattern, flags=re.I):
    # NOTE: the default is case-INSENSITIVE. A case-sensitive marker must pass
    # flags=0 explicitly. `(?m)^[a-z]` under re.I matches every capitalised
    # paragraph too and scored the human class at 100.0% before this was caught.
    rx = re.compile(pattern, flags)
    return lambda t: bool(rx.search(t))


MARKERS = [
    # --- punctuation the generator normalises, kept here to prove it worked ---
    ("punct", "em dash", _re(r"—")),
    ("punct", "en dash", _re(r"–")),
    ("punct", "curly quote", _re(r"[‘’“”]")),
    ("punct", "ellipsis char", _re(r"…")),
    ("punct", "double space after period", _re(r"\.  +[A-Z]", 0)),
    ("punct", "exclamation", _re(r"!")),
    # Consistency markers. A document that mixes curly and straight punctuation,
    # or spells one ellipsis as a character and the next as three dots, is
    # displaying an EDIT rather than a habit -- a person's client does one or
    # the other throughout. The presence markers above cannot see this: they ask
    # whether a curly quote occurs, and one occurrence matched the human rate
    # perfectly while being a fingerprint. A QA reader found it; this is the
    # counter that would have.
    ("punct", "mixed curly and straight quotes",
     lambda t: bool(re.search(r"[\u2018\u2019]", t)) and bool(re.search(r"'", t))),
    ("punct", "mixed ellipsis styles",
     lambda t: "\u2026" in t and "..." in t),
    # --- the LLM lexicon ---
    ("lexicon", "delve", _re(r"\bdelv(e|es|ing)\b")),
    ("lexicon", "it's important/worth noting", _re(r"\bit(?:'s| is) (?:important|worth) (?:to note|noting|remembering)\b")),
    ("lexicon", "that said", _re(r"\bthat said\b|\bhaving said that\b")),
    ("lexicon", "in essence / essentially", _re(r"\bin essence\b|\bessentially\b")),
    ("lexicon", "fundamentally", _re(r"\bfundamentall?y\b")),
    ("lexicon", "crucial/vital/pivotal", _re(r"\b(crucial|vital|pivotal)\b")),
    ("lexicon", "robust/leverage/navigate", _re(r"\b(robust|leverag(e|es|ing)|navigat(e|es|ing))\b")),
    ("lexicon", "tapestry/testament/underscore/showcase", _re(r"\b(tapestry|testament|underscor(e|es|ing)|showcas(e|es|ing))\b")),
    ("lexicon", "not just X, it's Y", _re(r"\b(?:is|it's|its)\s+not\s+just\b.{0,60}?\b(?:it'?s|but)\b")),
    ("lexicon", "think of it like / imagine", _re(r"\bthink of it (?:like|as)\b|\bimagine\b")),
    ("lexicon", "the key(?) is", _re(r"\bthe key (?:is|here|point)\b")),
    # --- structural scaffolding ---
    ("scaffold", "First,... Second,...", _re(r"\bfirst[,:].{0,400}?\bsecond[,:]", re.I | re.S)),
    ("scaffold", "there are N reasons/ways/types", _re(r"\bthere are (two|three|four|several|a few) (reasons|ways|types|kinds|things|factors)\b")),
    ("scaffold", "numbered list", _re(r"(?m)^\s*\d+[.)]\s+")),
    ("scaffold", "bullet list", _re(r"(?m)^\s*[-*•]\s+")),
    ("scaffold", "markdown emphasis", _re(r"\*\*[^*]+\*\*|(?<!\*)\*[^*\n]+\*(?!\*)")),
    ("scaffold", "header line", _re(r"(?m)^#{1,4}\s+\S")),
    ("scaffold", "TL;DR", _re(r"\bt\.?l;?\s?d\.?r\b")),
    # --- reddit-native, the phase 1 injection set ---
    ("reddit", "Edit: line", _re(r"(?mi)^\s*edit\s*[:—-]")),
    ("reddit", "edit anywhere", _re(r"(?i)\bedit\s*:")),
    ("reddit", "OP/upvote/karma", _re(r"\bOP\b|(?i:upvot|downvot|karma|subreddit|/r/\w+)", 0)),
    ("reddit", "source: I am a", _re(r"\bsource\s*:\s*(i|my)\b")),
    # --- voice ---
    ("voice", "first person I/my", _re(r"(?<![A-Za-z])I(?![A-Za-z'])|(?i:\bmy\b)", 0)),
    ("voice", "stock hedge", _re(r"\b(i'?m no expert|iirc|if i recall|correct me if i'?m wrong|i (?:might|could) be wrong|afaik|not 100% sure|take this with a grain)\b")),
    ("voice", "own hedge (I think/pretty sure)", _re(r"\b(i think|i believe|pretty sure|as far as i know|i guess)\b")),
    ("voice", "personal experience", _re(r"\b(my (job|wife|husband|dad|mom|brother|sister|uncle|aunt|cousin|coworker|boss)|i work(ed)? (in|as|at)|when i was|i used to)\b")),
    ("voice", "mild profanity", _re(r"\b(damn|hell|crap|shit|ass|sucks|stupid|bullshit)\b")),
    # An earlier version of this matched `(?:^|[.!?]\s+)[a-z]`, which reads
    # "i.e. the" and "e.g. a" as lowercase sentence starts and scored 100.0% on
    # the human class. A marker that fires on every document measures nothing.
    ("voice", "paragraph starts lowercase", _re(r"(?m)^[a-z]", 0)),
    ("voice", "question mark", _re(r"\?")),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def two_prop_z(k1, n1, k2, n2):
    """Two-sided p for equality of two rates. Normal approximation, which is
    adequate at the n here and is quoted only as a flag, never as a finding."""
    if not n1 or not n2:
        return float("nan")
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def rates(texts):
    out = collections.OrderedDict()
    for fam, name, test in MARKERS:
        out[(fam, name)] = sum(1 for t in texts if test(t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", nargs="*", default=[])
    ap.add_argument("--human-ids", nargs="*", default=["dev", "bench", "heldout", "pool"],
                    help="partition names from phase3/data/longform_partition.json")
    ap.add_argument("--human-corpus",
                    default=f"{ROOT}/phase1/data/interim/questions_longform.jsonl")
    ap.add_argument("--match-ai-ids", action="store_true",
                    help="restrict humans to the questions the ai set covers")
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--min-gap", type=float, default=0.0,
                    help="only print rows where |ai - human| exceeds this (pts)")
    a = ap.parse_args()

    part = json.load(open(f"{ROOT}/phase3/data/longform_partition.json"))
    want = set()
    for name in a.human_ids:
        if name not in part:
            sys.exit(f"FATAL: no partition named '{name}'. Have: "
                     f"{[k for k in part if isinstance(part[k], list)]}")
        want |= set(part[name])
    humans = {}
    for l in open(a.human_corpus):
        r = json.loads(l)
        if r["id"] in want:
            humans[r["id"]] = r["human_answer"]

    ai = {}
    for path in a.ai:
        if not os.path.exists(path):
            sys.exit(f"FATAL: {path} not found")
        for i, line in enumerate(open(path), 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"FATAL {path}:{i}: {e}")
            ai[(r["id"], r["model"])] = r
    if a.match_ai_ids and ai:
        ids = {qid for qid, _ in ai}
        humans = {k: v for k, v in humans.items() if k in ids}

    h_texts = list(humans.values())
    h_counts = rates(h_texts)
    nh = len(h_texts)
    print(f"human: {nh} long-form documents from {'+'.join(a.human_ids)}")
    if ai:
        print(f"ai:    {len(ai)} documents over "
              f"{len({m for _, m in ai})} models, "
              f"{len({q for q, _ in ai})} questions")
    print()

    a_texts = [r["text"] for r in ai.values()]
    a_counts = rates(a_texts) if a_texts else None
    na = len(a_texts)

    hdr = f"{'family':<9} {'marker':<42} {'human':>7} {'95% CI':>15}"
    if a_counts:
        hdr += f"  {'ai':>7} {'95% CI':>15} {'gap':>7} {'p':>8}"
    print(hdr)
    out = []
    for (fam, name), hk in h_counts.items():
        hp = 100 * hk / nh
        hlo, hhi = wilson(hk, nh)
        row = f"{fam:<9} {name:<42} {hp:>6.1f}% [{hlo:>5.1f},{hhi:>5.1f}]"
        rec = {"family": fam, "marker": name, "human_k": hk, "human_n": nh,
               "human_rate": hp}
        if a_counts:
            ak = a_counts[(fam, name)]
            ap_ = 100 * ak / na
            alo, ahi = wilson(ak, na)
            p = two_prop_z(ak, na, hk, nh)
            gap = ap_ - hp
            flag = ""
            if p < 0.05:
                flag = "  <-- MISMATCHED"
                if ak == 0 and hk > 0:
                    flag = "  <-- ZERO against a nonzero human rate: a fingerprint"
            row += f"  {ap_:>6.1f}% [{alo:>5.1f},{ahi:>5.1f}] {gap:>+6.1f} {p:>8.4f}{flag}"
            rec.update({"ai_k": ak, "ai_n": na, "ai_rate": ap_, "gap": gap, "p": p})
            if abs(gap) < a.min_gap:
                continue
        out.append(rec)
        print(row)

    if a_counts:
        sig = [r for r in out if r.get("p", 1) < 0.05]
        tot = sum(abs(r["gap"]) for r in out)
        print(f"\n{len(sig)} of {len(out)} markers differ at p<0.05; "
              f"total absolute gap {tot:.1f} points across {len(out)} markers.")
        print("Matched means inside the interval, not equal to the point estimate. "
              "A rate of\nexactly 0.0% against a nonzero human rate is the "
              "fingerprint phase 1 spent a week\nremoving from the human class; "
              "it is only harmless where the human rate is too\nsmall for this n "
              "to resolve.")
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)
        print(f"\n-> {a.json_out}")


if __name__ == "__main__":
    main()

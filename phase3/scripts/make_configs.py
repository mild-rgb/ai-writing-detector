"""Emit each prompt's config.json from the MEASURED human marker rates.

The rates are not typed in by hand anywhere. They are read from
phase3/data/human_marker_rates.json, which markers.py wrote by counting the
long-form human class (dev+bench, n=130). If the measurement is re-run the
configs change with it, and "rate-matched to the human base rate" stays a fact
about this corpus rather than a claim about phase 1's short-form corpus, whose
rates are different: first person 56.9% here against 23.3% there, `Edit:` 13.8%
against 5.1%.

Draws whose human rate is below what n=130 can resolve are NOT injected. A rate
of 0.0% [0.0, 2.9] cannot be matched by a draw -- any nonzero rate is as
arbitrary as zero -- so those markers are left to the prompt floor's
prohibition, and the resulting zero is a fingerprint only in principle: at the
n this project judges, it is undetectable. That is the quantitative resolution
of "never prohibit", which is otherwise a rule with no stopping point.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RATES = f"{ROOT}/phase3/data/human_marker_rates.json"
PROMPTS = f"{ROOT}/phase3/adversarial_prompts"

rates = {r["marker"]: r["human_rate"] / 100 for r in json.load(open(RATES))}


def R(name):
    if name not in rates:
        sys.exit(f"FATAL: no measured rate for '{name}'. Run markers.py first.")
    return round(rates[name], 4)


# Instruction text for each drawn feature. The rate is measured; the wording is
# written to ask for the idea in the model's OWN words wherever a fixed string
# would otherwise recur -- v8_invert planted 'I might be wrong' in 56.3% of
# documents against a human 0.0% and that single fact made its result hollow.
# (marker, instruction) or (marker, instruction, instruction-when-not-drawn).
# The third element makes the draw two-sided; see build_system() in generate.py
# for why a spontaneously produced feature needs one.
DRAWS = [
    ("markdown emphasis", "Italicise or bold one short phrase where you want emphasis, using asterisks the way reddit markdown does.",
     "Do not use any asterisks, bold or italics anywhere in this comment."),
    ("numbered list", "Part of this is easier as a short numbered list. Use one, then go back to prose.",
     "Do not use a numbered list."),
    ("bullet list", "Part of this is easier as a few dashed bullet points. Use them, then go back to prose.",
     "Do not use bullet points."),
    ("TL;DR", "Open with a one-line 'TL;DR' summary, then a blank line, then the full explanation.",
     "Do not open with a summary line."),
    ("question mark", "Put a question to the reader somewhere in this, the way you would if you were talking to them."),
    ("Edit: line", "End with a line beginning 'Edit:' that corrects or clarifies something you actually said above. It must refer to a detail that is really in your comment."),
    ("first person I/my", "Write this one in the first person."),
    ("own hedge (I think/pretty sure)", "Hedge one claim in your own words, however you would actually say it. Do not use a stock disclaimer."),
    ("personal experience", "Ground one claim in something you have seen or done yourself. Drop it in mid-thought, not as an opening credential."),
    ("source: I am a", "You do this for a living and you say so, briefly and in passing."),
    ("mild profanity", "You find some part of this a bit stupid and it shows. Blunt, dry, mild swearing is fine."),
    ("exclamation", "Let one sentence end on an exclamation mark."),
]

SHARED_POST = {
    "em_dash": R("em dash"),
    "en_dash": R("en dash"),
    "curly": R("curly quote"),
    "ellipsis": R("ellipsis char"),
    "double_space": R("double space after period"),
    "lowercase_para": R("paragraph starts lowercase"),
}

# Measured shape distribution of the human long-form class (n=130):
# one block 0.8%, two-to-three 10.0%, four-to-eight 67.7%, nine-plus 21.5%.
SHAPE_CHOICE = [
    {"rate": 0.008, "text": "Write it as one unbroken block. No paragraph breaks at all."},
    {"rate": 0.100, "text": "Write it as two or three long paragraphs."},
    {"rate": 0.677, "text": "Write it as somewhere between four and eight paragraphs of uneven length."},
    {"rate": 0.215, "text": "Write it long, with many short paragraphs -- ten or more breaks."},
]

SPECS = {
    "p1_pinned": {},
    "p2_mechanical": {"post_extra": {"repara": 1.0}},
    "p3_shape": {"choice": {"shape": SHAPE_CHOICE}},
    "p4_voice": {},
    "p5_disfluency": {},
    "p6_scope": {},
}

floor = open(f"{PROMPTS}/_floor.txt").read().strip()
for name, spec in SPECS.items():
    d = f"{PROMPTS}/{name}"
    if not os.path.isdir(d):
        sys.exit(f"FATAL: {d} does not exist")
    cfg = {
        "mechanism": name,
        "rates_from": "phase3/data/human_marker_rates.json (long-form human, dev+bench, n=130)",
        "draws": [dict({"rate": R(d[0]), "text": d[1]},
                       **({"else_text": d[2]} if len(d) > 2 else {}))
                  for d in DRAWS],
        "post": {**SHARED_POST, **spec.get("post_extra", {})},
    }
    if "choice" in spec:
        cfg["choice"] = spec["choice"]
    json.dump(cfg, open(f"{d}/config.json", "w"), indent=1)
    # The prompt on disk keeps a __FLOOR__ marker so the shared part is edited
    # in one place; the resolved prompt is what generation actually sends.
    resolved = open(f"{d}/prompt.txt").read().replace("__FLOOR__", floor).strip()
    open(f"{d}/prompt.resolved.txt", "w").write(resolved + "\n")
    print(f"{name:<15} {len(cfg['draws'])} draws, "
          f"{len(cfg['post'])} post-ops"
          f"{', shape choice' if 'choice' in cfg else ''}, "
          f"{len(resolved.split())} words resolved")
print(f"\nexpected draws per document: "
      f"{sum(R(d[0]) for d in DRAWS):.2f}")

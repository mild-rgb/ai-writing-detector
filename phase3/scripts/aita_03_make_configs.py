"""Emit the AITA prompt config from the MEASURED AITA human marker rates.

Same rule as make_configs.py: no rate is typed in by hand. They are read from
phase3/aita/data/aita_human_marker_rates.json, which markers.py wrote by
counting the AITA human class (dev+bench, n=130).

Re-measuring was not optional. AITA writes nothing like ELI5, and reusing the
ELI5 config would have mis-set almost every draw:

    marker                      ELI5     AITA
    first person I/my          56.9%    99.2%
    personal experience         1.5%    43.1%
    question mark              36.1%    83.1%
    mild profanity             13.9%    28.5%
    own hedge                  10.8%    26.9%
    curly quote                 3.9%    31.5%
    TL;DR                       7.7%    13.1%
    markdown emphasis          32.3%     9.2%
    Edit: line                 13.8%     1.5%
    numbered list               7.7%     1.5%

The instruction WORDING is rewritten too, not just the rate. "Ground one claim
in something you have seen or done yourself" is an odd thing to ask of a post
that is nothing but firsthand experience, and "put a question to the reader"
describes an ELI5 aside, where in AITA the 83.1% question rate is mostly the
post asking for its verdict.

Draws whose measured rate is 0.0% [0.0, 2.9] are dropped rather than injected at
zero: a rate n=130 cannot resolve cannot be matched by a draw, and a draw that
never fires is dead weight in the config.

    python3 phase3/scripts/aita_03_make_configs.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RATES = f"{ROOT}/phase3/aita/data/aita_human_marker_rates.json"
PROMPTS = f"{ROOT}/phase3/aita/prompts"

rates = {r["marker"]: r["human_rate"] / 100 for r in json.load(open(RATES))}


# Measured by aita_11_name_pool.py over all 6,343 human documents: the share
# that name a person appearing on the curated gendered name list.
NAMING_RATE = 0.117


def R(name):
    if name == "__naming__":
        return NAMING_RATE
    if name not in rates:
        sys.exit(f"FATAL: no measured rate for '{name}'. Run markers.py first.")
    return round(rates[name], 4)


# (marker, instruction) or (marker, instruction, instruction-when-not-drawn).
# The third element makes the draw two-sided, which is required for any feature
# the models produce spontaneously: a one-sided draw only ever ADDS to whatever
# the model was going to do anyway.
DRAWS = [
    ("first person I/my",
     "Write this as your own story, in the first person.",
     "Tell this without using 'I' or 'my' -- write it about the people involved, not about yourself."),
    # The "do not use a stock phrasing" clause was REPLACED, not deleted. It was
    # pushing the class to "tell me if I" (12% ai, 0% human) and away from the
    # canonical "am I the asshole" (23% human, 10% ai) -- the prohibition made
    # the text LESS human on that axis and planted a lexical tell. It is
    # replaced rather than removed because it is a scar: v8_invert planted
    # "I might be wrong" in 56.3% of documents against a human 0.0%, and asking
    # for the model's own words is what stops a fixed string recurring. The rule
    # is right in general and wrong only here, where the stock phrase is the
    # genre's own name. A bare deletion risks the mirror failure -- models like a
    # canonical phrase and it could overshoot far past the human 23%, trading a
    # 12-point tell for a larger one. Permit, do not mandate; measure both
    # strings; convert to a rate-matched draw at 23% only if canonical clears 35%.
    # Measured at 50.8% against a human 83.1% on dev -- a 32-point gap, the only
    # marker clearly out. The draw was already two-sided at the human rate, so
    # the cause was not the draw: the floor's "do not end on a flourish, stop
    # when you have said what happened" was fighting it, and the models obeyed
    # the prohibition over the instruction. The fix is in _floor.txt, which now
    # exempts the verdict question from that rule. Strengthening the draw alone
    # would have bolted a question onto a post the floor had just told the model
    # to end abruptly, which is a worse tell than the one it closes.
    ("question mark",
     "Somewhere in this, ask whether you were in the wrong -- as an actual question. Phrase it however you would actually put it; the sub's usual phrasing is fine if that is what comes out. It can close the post or land mid-thought.",
     "Do not ask a question anywhere in this post. State the situation and stop."),
    ("personal experience",
     "Include one concrete thing you did or saw that you have not mentioned yet, dropped in mid-thought rather than set up.",
     None),
    ("mild profanity",
     "You are annoyed and it shows. Blunt, dry, mild swearing is fine.",
     "Keep it level. No swearing."),
    ("own hedge (I think/pretty sure)",
     "Hedge one detail in your own words -- you are not certain it happened exactly that way. Do not use a stock disclaimer.",
     None),
    ("exclamation",
     "Let one sentence end on an exclamation mark.",
     "No exclamation marks anywhere."),
    ("TL;DR",
     "End with a one-line 'TL;DR' summarising the situation, after a blank line.",
     "Do not add a summary line."),
    ("markdown emphasis",
     "Italicise or bold one short phrase where the emphasis actually falls, using asterisks the way reddit markdown does.",
     "Do not use any asterisks, bold or italics anywhere in this post."),
    ("Edit: line",
     "End with a line beginning 'Edit:' answering something a commenter would have asked. It must refer to a detail that is really in your post.",
     None),
    ("numbered list",
     "One part of this is easier as a short numbered list. Use one, then go back to prose.",
     "Do not use a numbered list."),
    ("bullet list",
     "One part of this is easier as a few dashed bullet points. Use them, then go back to prose.",
     "Do not use bullet points."),
    # NAMING. Two problems, and the larger one is the RATE, not the choice.
    # Measured on the 1,049-document early read against the human class on the
    # same questions: real posters name a person in 10.7% of posts -- they write
    # "my wife", not "Sarah" -- while gemini named someone in 84.0%, grok 56.0%,
    # glm 51.3%, deepseek 40.9%, qwen 40.7%, nemotron 30.7%. Only gpt, at 12.7%,
    # was near human.
    #
    # So this is TWO-SIDED, not a permit. A one-sided "if you name someone, call
    # them X" would fix which name appears and leave the over-naming untouched,
    # and the over-naming is the bigger tell. The negative branch pins the rate
    # for the models that name too much, and because gpt already sits at the
    # human rate the same draw leaves it where it is -- which is what "do not
    # push the under-namer further off" actually requires.
    #
    # {name} is filled per (id, model) from phase3/aita/data/name_pool.json,
    # weighted by the frequency each name has in the human corpus.
    ("__naming__",
     "Name one person in this and call them {name}. Everyone else is referred to by their relationship to you.",
     "Do not give anyone a name. Refer to everyone by their relationship to you -- my wife, my sister, my boss."),
    ("OP/upvote/karma",
     "Refer to the sub or to commenting the way a regular would, once, in passing.",
     None),
]

POST = {
    "em_dash": R("em dash"),
    "en_dash": R("en dash"),
    "curly": R("curly quote"),
    "ellipsis": R("ellipsis char"),
    "double_space": R("double space after period"),
    "lowercase_para": R("paragraph starts lowercase"),
}

RESOLUTION = 0.029  # the upper Wilson bound on 0/130

draws, dropped = [], []
for spec in DRAWS:
    marker, text = spec[0], spec[1]
    rate = R(marker)
    if rate < RESOLUTION and rate == 0.0:
        dropped.append(marker)
        continue
    d = {"rate": rate, "text": text}
    if len(spec) > 2 and spec[2]:
        d["else_text"] = spec[2]
    draws.append(d)

pool = json.load(open(f"{ROOT}/phase3/aita/data/name_pool.json"))["pool"]

cfg = {
    "mechanism": "aita_floor",
    "name_pool": pool,
    "rates_from": "phase3/aita/data/aita_human_marker_rates.json (AITA human, dev+bench, n=130)",
    "note": "Port of phase3/corpus/floor -- the generator that actually built the "
            "ELI5 corpus -- to r/AmItheAsshole. No structural instruction: the "
            "ELI5 paragraph campaign was retired after v2 because matching human "
            "paragraph structure pushed models into an essayistic register that "
            "Haiku found EASIER to catch (79.2% balanced against 57.3%). "
            "Whitespace is normalised on the detector side instead.",
    "draws": draws,
    "post": POST,
}

d = f"{PROMPTS}/aita_floor"
os.makedirs(d, exist_ok=True)
json.dump(cfg, open(f"{d}/config.json", "w"), indent=1)
# generate.py reads prompt.txt VERBATIM -- it does not expand __FLOOR__. In the
# ELI5 tree phase3/corpus/floor/prompt.txt is fully resolved on disk, which is
# why the corpus run carried the floor, while every adversarial_prompts/pN
# prompt.txt still begins with a literal "__FLOOR__" line. Measured consequence:
# p1_pinned's documents use the floor's own banned words in 10.7% of documents
# against 1.6% for the corpus floor, so those six runs were generated with no
# floor constraints at all. Here the template lives in template.txt and the
# RESOLVED text is written to prompt.txt, so what generate.py reads is what was
# intended.
floor = open(f"{PROMPTS}/_floor.txt").read().strip()
template = open(f"{d}/template.txt").read()
if "__FLOOR__" not in template:
    sys.exit(f"FATAL: {d}/template.txt has no __FLOOR__ marker")
resolved = template.replace("__FLOOR__", floor).strip()
open(f"{d}/prompt.txt", "w").write(resolved + "\n")
assert "__FLOOR__" not in open(f"{d}/prompt.txt").read()

print(f"floor: {len(draws)} draws, {len(POST)} post-ops, "
      f"{len(resolved.split())} words resolved")
if dropped:
    print(f"dropped (measured 0.0%, below what n=130 resolves): {', '.join(dropped)}")
print(f"expected draws fired per document: {sum(x['rate'] for x in draws):.2f}")
for x in draws:
    print(f"   {x['rate']*100:5.1f}%  {'two-sided' if 'else_text' in x else 'one-sided'}  {x['text'][:64]}")

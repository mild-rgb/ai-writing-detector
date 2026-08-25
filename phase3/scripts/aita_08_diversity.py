"""Scenario-diversity audit of the generated openers, per model, against humans.

Three questions, each with the human class on the SAME titles as the yardstick.
An absolute number here means nothing -- 2,900 AITA posts genuinely resemble one
another, because the genre is narrow -- so every AI figure is read against a
human figure computed the same way on the same questions at the same n.

1. TITLE-MATCH. Can an opener be matched back to its own title? If the AI
   inherits its diversity from the 6,343 real titles, its retrieval rate should
   sit near the human rate. A rate far ABOVE human means the opener is echoing
   the title rather than telling a story; far BELOW means the opener has drifted
   off its own prompt and the scenarios are coming from the model instead.

2. SCENARIO REPETITION. Nearest-neighbour cosine within each class, and the
   count of near-duplicate pairs. Compared per model against a human sample of
   the SAME SIZE on the SAME questions, because nearest-neighbour similarity
   rises mechanically with set size and an unmatched comparison would
   manufacture a difference.

3. NAME OVERUSE. A model that reaches for "Sarah" every time has a tell no
   cosine will show, since one name in thirty words barely moves a vector.
   Character names are extracted and their concentration measured against the
   human spread.

    python3 phase3/scripts/aita_08_diversity.py
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEAR_DUP = 0.5

# Capitalised words that are not character names. Sentence-initial tokens are
# dropped separately; this catches the rest.
NOT_NAMES = set("""i i'm i've i'd i'll a an the my me we he she it they them his her our your you
monday tuesday wednesday thursday friday saturday sunday january february march april may june july
august september october november december mom dad mum mother father sister brother wife husband
son daughter aunt uncle cousin grandma grandpa nan gran christmas thanksgiving easter halloween
new year's god english spanish french american british uber walmart target facebook instagram
reddit aita nta yta esh info tl tldr edit update so but and then when after before also
ive ill id im ive youre theyre hes shes weve theyve wouldnt couldnt didnt wasnt isnt dont cant
sunday street day world year years time way thing things people someone anyone everyone
because however although since while during until unless whether though if that this these those
what who where why how which
""".split())
NAME_RX = re.compile(r"\b[A-Z][a-z]{2,}\b")


def openers_of(text, n=30):
    return " ".join(text.split()[:n])


def names_in(text):
    """Capitalised words that are not sentence-initial and not common words."""
    out = []
    # mark tokens that follow a sentence end or start the string
    toks = re.findall(r"\S+", text)
    initial = set()
    nxt = True
    for i, t in enumerate(toks):
        if nxt:
            initial.add(i)
        nxt = bool(re.search(r"[.!?][\"')\]]*$", t))
    for i, t in enumerate(toks):
        # Normalise curly apostrophes to straight BEFORE stripping punctuation.
        # op_curly rewrites apostrophes as U+2019 in a third of documents, and
        # stripping that as punctuation turns "I've" into "Ive" and "I'll" into
        # "Ill", which then look exactly like capitalised names. That put "Ive"
        # at the top of the human name list with 19 occurrences and inflated the
        # human unique-name count, making the AI classes look more concentrated
        # by comparison than they are.
        t = t.replace("\u2019", "'").replace("\u2018", "'")
        w = re.sub(r"[^A-Za-z']", "", t)
        if "'" in w:          # any contraction, not a name
            continue
        if i in initial or not w:
            continue
        if NAME_RX.fullmatch(w) and w.lower() not in NOT_NAMES:
            out.append(w)
    return out


def retrieval(open_texts, title_texts):
    """top-1 / top-5 rate of an opener retrieving its OWN title, plus the
    random baseline for the same n."""
    v = TfidfVectorizer(sublinear_tf=True, min_df=1, stop_words="english")
    v.fit(list(open_texts) + list(title_texts))
    O = v.transform(open_texts)
    T = v.transform(title_texts)
    S = (O @ T.T).toarray()
    n = len(open_texts)
    order = np.argsort(-S, axis=1)
    top1 = float(np.mean(order[:, 0] == np.arange(n)))
    top5 = float(np.mean([(i in order[i, :5]) for i in range(n)]))
    return top1, top5, 1.0 / n


def repetition(texts):
    v = TfidfVectorizer(sublinear_tf=True, min_df=1, stop_words="english")
    X = v.fit_transform(texts)
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, -1.0)
    nn = S.max(axis=1)
    iu = np.triu_indices(len(texts), k=1)
    dups = int((S[iu] > NEAR_DUP).sum())
    return float(nn.mean()), float(np.percentile(nn, 95)), dups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openers", default=f"{ROOT}/phase3/aita/data/openers.jsonl")
    ap.add_argument("--humans", default=f"{ROOT}/phase3/aita/data/aita_human.jsonl")
    ap.add_argument("--words", type=int, default=30,
                    help="how many leading words count as the opener")
    ap.add_argument("--json", dest="json_out")
    a = ap.parse_args()

    # Accepts either the opener file (an `opener` field) or a normal answers
    # file (a `text` field), so the same analysis runs on truncated openers and
    # on full corpus documents without a second code path. Full documents are
    # cut to the same word count, because scenario choice happens in the opening
    # sentences and comparing a 30-word opener against a 400-word document would
    # measure length, not diversity.
    ai = []
    for l in open(a.openers):
        if not l.strip():
            continue
        r = json.loads(l)
        if "opener" not in r:
            r["opener"] = openers_of(r["text"], a.words)
        ai.append(r)
    humans = {json.loads(l)["id"]: json.loads(l) for l in open(a.humans)}
    by = defaultdict(list)
    for r in ai:
        by[r["model"]].append(r)

    print(f"{len(ai)} ai openers over {len(by)} models, "
          f"first {len(ai[0]['opener'].split())} words each\n")

    # ---- 1 & 2, per model, against the human class on the SAME questions ----
    print("1+2. TITLE-MATCH and REPETITION, per model, human control on the "
          "same questions at the same n")
    print(f"{'model':<26}{'n':>5}{'top1':>13}{'top5':>13}"
          f"{'nn cos':>14}{'near-dups':>12}")
    rows = {}
    for m in sorted(by):
        v = by[m]
        ids = [r["id"] for r in v]
        ao = [r["opener"] for r in v]
        ho = [openers_of(humans[i]["human_answer"], a.words) for i in ids]
        ti = [humans[i]["question"] for i in ids]
        a1, a5, base = retrieval(ao, ti)
        h1, h5, _ = retrieval(ho, ti)
        ann, a95, adup = repetition(ao)
        hnn, h95, hdup = repetition(ho)
        rows[m] = dict(n=len(v), ai_top1=a1, hu_top1=h1, ai_top5=a5, hu_top5=h5,
                       base=base, ai_nn=ann, hu_nn=hnn, ai_dup=adup, hu_dup=hdup)
        print(f"{m.split('/')[-1]:<26}{len(v):>5}"
              f"{100*a1:>7.1f}/{100*h1:<5.1f}{100*a5:>7.1f}/{100*h5:<5.1f}"
              f"{ann:>7.3f}/{hnn:<6.3f}{adup:>6d}/{hdup:<5d}")
    print("   (each cell is ai/human; random top-1 baseline is "
          f"{100*rows[list(rows)[0]]['base']:.2f}%)")

    # ---- 3, name concentration ----
    print("\n3. CHARACTER NAMES, per model against the human class on the same "
          "questions")
    print(f"{'model':<26}{'names':>7}{'uniq':>6}{'top name':>22}{'top-5 share':>13}")
    name_rows = {}
    for m in sorted(by) + ["HUMAN (all assigned)"]:
        if m.startswith("HUMAN"):
            texts = [openers_of(humans[r["id"]]["human_answer"], a.words) for r in ai]
        else:
            texts = [r["opener"] for r in by[m]]
        c = Counter(n for t in texts for n in names_in(t))
        tot = sum(c.values())
        if not tot:
            print(f"{m.split('/')[-1]:<26}{0:>7}{0:>6}{'-':>22}{'-':>13}")
            continue
        top = c.most_common(1)[0]
        share5 = sum(v for _, v in c.most_common(5)) / tot
        name_rows[m] = dict(total=tot, uniq=len(c), top=top[0], top_n=top[1],
                            top_share=top[1] / tot, top5_share=share5,
                            most_common=c.most_common(8))
        print(f"{m.split('/')[-1]:<26}{tot:>7}{len(c):>6}"
              f"{top[0] + ' ' + str(top[1]) + ' (' + f'{100*top[1]/tot:.1f}%)':>22}"
              f"{100*share5:>12.1f}%")
    for m, d in name_rows.items():
        print(f"   {m.split('/')[-1]:<26} " +
              ", ".join(f"{n} {k}" for n, k in d["most_common"][:6]))

    if a.json_out:
        json.dump({"per_model": rows, "names": name_rows}, open(a.json_out, "w"),
                  indent=1, default=str)
        print(f"\n-> {a.json_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Model attribution and the era-vs-model distinction, on the ELI5 + AITA corpora.

The human-vs-AI detector is vintage-locked: it catches the seven 2026 generators
and misses older models (see the OOD probe). This script asks a different, better-
posed question the two corpora were built for: given a document, WHICH of the seven
2026 models (or a human) wrote it -- and what does a bag-of-words model reveal about
how that signal is structured.

Findings, all reproduced below (BoW = binary word+punct presence, logistic
regression, the same floor definition used elsewhere):

  1. ATTRIBUTION WORKS. Trained on both domains, BoW names which of 7 models wrote
     a held-out document 91.3% of the time (ELI5 88.2%, AITA 92.9%, chance 12.5%).

  2. IT IS THE TRAINING DOMAIN, NOT AN INHERENT LIMIT. Trained on ELI5 alone, five
     of seven models' fingerprints do NOT transfer to AITA (10-42% recall). Add
     AITA to training and the same five reach 75-100% (deepseek 75.3 is the floor,
     grok 100.0 the ceiling). The cross-domain collapse was a single-domain
     artifact.

  3. HUMAN IS THE CLEANEST CUT. The human class is identified 98.4% on held-out
     text from both domains, with 0.69% of AI leaking into it. Human/AI is coarse
     and domain-invariant; which-model is fine and carries the error.

  4. ERA-LOCKED, NOT MODEL-LOCKED (leave-one-model-out). Hold a 2026 model out of
     training entirely and its text is STILL caught as AI 95.3% of the time on
     average (evasion 0.3% grok to 17.4% gpt) -- it is misattributed to another
     2026 model, not called human. Compare the SAME BoW floor on older models in
     the OOD probe, which missed 96.3% of GPT-4/3.5, 93.8% of LLaMA/OPT and 91.7%
     of llama-chat/MPT. Like for like: this method catches an unseen 2026 model
     19 times in 20 and misses a 2022 model 19 times in 20. So there is a shared
     2026-frontier signature that generalises to unseen SAME-ERA models; the
     failure is a boundary between eras, not between known and unknown models.

NOTE ON SPLITS: ELI5 uses its stamped train/test split. AITA has no stamped split
(its partition lives in a side file), so this uses a seeded 80/20 split by q_id for
CHARACTERISATION only -- not a canonical corpus split. See README provenance note.

    python3 phase3/scripts/attribution.py
"""
import json, re, random, collections
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SEED = 20260825
def toks(t): return re.findall(r"[a-z']+|[^\sa-zA-Z0-9]", t.lower())
def short(m): return m.split("/")[-1]
def bow(): return CountVectorizer(tokenizer=toks, token_pattern=None, binary=True, min_df=5)
def fit(train):
    v = bow(); clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(v.fit_transform([t for t, _ in train]), [g for _, g in train]); return v, clf

# ---------- load ----------
eli5 = [json.loads(l) for l in open(f"{ROOT}/exp01_detector/docs.jsonl")]
aita = [json.loads(l) for l in open(f"{ROOT}/aita/data/corpus_answers.jsonl") if l.strip()]
hmap = {r["q_id"]: r["human_answer"] for r in map(json.loads, open(f"{ROOT}/aita/data/aita_human.jsonl"))}
models = sorted({short(d["generator"]) for d in eli5 if d["label"] == "ai"})

e_ai = lambda sp: [(d["text_norm"], short(d["generator"])) for d in eli5 if d["split"] == sp and d["label"] == "ai"]
e_hu = lambda sp: [d["text_norm"] for d in eli5 if d["split"] == sp and d["label"] == "human"]
aq = sorted({r["q_id"] for r in aita if r.get("text")})
random.Random(SEED).shuffle(aq); atrain = set(aq[:int(0.8 * len(aq))])
a_ai = lambda tr: [(r["text"], short(r["model"])) for r in aita if r.get("text") and (r["q_id"] in atrain) == tr]
a_hu = lambda tr: [hmap[r["q_id"]] for r in aita if r.get("text") and r["q_id"] in hmap and (r["q_id"] in atrain) == tr]

# ---------- sample counts (all available AI docs, every split) ----------
e_by = collections.Counter(short(d["generator"]) for d in eli5 if d["label"] == "ai")
a_by = collections.Counter(m for _, m in a_ai(True) + a_ai(False))
print(f"SAMPLES PER MODEL\n{'model':24}{'ELI5':>7}{'AITA':>7}{'total':>7}")
for m in models: print(f"{m:24}{e_by[m]:>7}{a_by[m]:>7}{e_by[m]+a_by[m]:>7}")
print(f"{'ALL AI':24}{sum(e_by.values()):>7}{sum(a_by.values()):>7}{sum(e_by.values())+sum(a_by.values()):>7}")

def per_class(v, clf, data):
    pred = clf.predict(v.transform([t for t, _ in data])); y = [g for _, g in data]
    return pred, y, accuracy_score(y, pred)

# ---------- (2) single-domain contrast: ELI5-only -> AITA ----------
rng = random.Random(SEED)
tr = e_ai("train") + [(t, "human") for t in rng.sample(e_hu("train"), len(e_ai("train")) // len(models))]
v, clf = fit(tr)
te = a_ai(False) + a_ai(True)
pred, y, _ = per_class(v, clf, te)
print("\nSINGLE-DOMAIN (train ELI5 only) -> AITA, recall per model:")
for m in models:
    idx = [i for i, g in enumerate(y) if g == m]
    print(f"    {m:24} {np.mean([pred[i]==m for i in idx])*100:5.1f}%")

# ---------- (1)+(3) both-domain attributor ----------
ai_tr = e_ai("train") + a_ai(True)
per = len(ai_tr) // len(models)
train = ai_tr + [(t, "human") for t in rng.sample(e_hu("train") + a_hu(True), per)]
v, clf = fit(train)
mk = lambda ai, hu: ai + [(t, "human") for t in rng.sample(hu, max(1, len(ai) // len(models)))]
etest = mk(e_ai("test"), e_hu("test")); atest = mk(a_ai(False), a_hu(False)); comb = etest + atest
print(f"\nBOTH-DOMAIN 8-way (7 models + human), balanced. chance 12.5%")
print(f"    ELI5 holdout {per_class(v,clf,etest)[2]*100:.1f}%   AITA holdout {per_class(v,clf,atest)[2]*100:.1f}%"
      f"   COMBINED {per_class(v,clf,comb)[2]*100:.1f}%")
pred, y, _ = per_class(v, clf, comb)
hy = [i for i, g in enumerate(y) if g == "human"]
leak = sum(1 for i in range(len(y)) if y[i] != "human" and pred[i] == "human")
print(f"    human recall {np.mean([pred[i]=='human' for i in hy])*100:.1f}%   "
      f"AI leaked into human {leak}/{sum(1 for g in y if g!='human')} = "
      f"{leak/sum(1 for g in y if g!='human')*100:.2f}%")
# Per-model recall on the AITA holdout, printed so the "add AITA and they jump"
# claim is citable from this script rather than reconstructed. Directly
# comparable to the SINGLE-DOMAIN block above: same models, same target domain.
apred, ay, _ = per_class(v, clf, atest)
print("    per-model recall on the AITA holdout (compare the single-domain block):")
for m in models:
    idx = [i for i, g in enumerate(ay) if g == m]
    print(f"        {m:24} {np.mean([apred[i]==m for i in idx])*100:5.1f}%")

# ---------- (4) leave-one-model-out ----------
allai = e_ai("train") + e_ai("test") + a_ai(True) + a_ai(False)
allhu = e_hu("train") + e_hu("test") + a_hu(True) + a_hu(False)
per = len(allai) // len(models)
print("\nLEAVE-ONE-MODEL-OUT: train on other 6 + human, test on held-out model")
print(f"    {'held-out':24}{'called HUMAN (evades)':>22}{'caught as AI':>14}")
ev = []
for M in models:
    trn = [(t, m) for t, m in allai if m != M] + [(t, "human") for t in random.Random(SEED).sample(allhu, per)]
    v2, clf2 = fit(trn)
    p = clf2.predict(v2.transform([t for t, m in allai if m == M]))
    h = np.mean(p == "human"); ev.append(h)
    print(f"    {M:24}{h*100:>20.1f}%{(1-h)*100:>13.1f}%")
print(f"    mean evasion (unseen 2026 model called human): {np.mean(ev)*100:.1f}%")
print("    (same BoW floor on OLDER models in the OOD probe missed 96.3% of")
print("     GPT-4/3.5, 93.8% of LLaMA/OPT, 91.7% of llama-chat/MPT -- so this")
print("     method catches an unseen 2026 model ~19/20 and misses a 2022 one ~19/20)")

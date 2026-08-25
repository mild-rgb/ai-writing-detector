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
     a held-out document 90.3% of the time (ELI5 87.6%, AITA 93.1%, chance 12.5%).

  2. IT IS THE TRAINING DOMAIN, NOT AN INHERENT LIMIT. Trained on ELI5 alone, five
     of seven models' fingerprints do NOT transfer to AITA (10-42% recall). Add
     AITA to training and the same five reach 77-100% (deepseek 77.1 is the floor,
     grok and gemini 100.0 the ceiling). The cross-domain collapse was a single-domain
     artifact.

  3. HUMAN IS THE CLEANEST CUT. The human class is identified 97.6% on held-out
     text from both domains, with 0.52% of AI leaking into it. Human/AI is coarse
     and domain-invariant; which-model is fine and carries the error.

  4. ERA-LOCKED, NOT MODEL-LOCKED (leave-one-model-out). Hold a 2026 model out of
     training entirely and its text is STILL caught as AI 95.4% of the time on
     average (evasion 0.1% grok to 17.8% gpt) -- it is misattributed to another
     2026 model, not called human. Compare the SAME BoW floor on older models in
     the OOD probe, which missed 96.3% of GPT-4/3.5, 93.8% of LLaMA/OPT and 91.7%
     of llama-chat/MPT. Like for like: this method catches an unseen 2026 model
     19 times in 20 and misses a 2022 model 19 times in 20. So there is a shared
     2026-frontier signature that generalises to unseen SAME-ERA models; the
     failure is a boundary between eras, not between known and unknown models.

NOTE ON SPLITS: both corpora use their own STAMPED, question-grouped train/test
split, read off the records. AITA's is written by aita_13_stamp_corpus.py over pool
questions only, disjoint from the dev/bench/heldout used to tune the floor prompt.
Text is text_norm on both sides, so the two domains are normalised alike.

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
aita = [json.loads(l) for l in open(f"{ROOT}/aita/data/aita_docs.jsonl")]
models = sorted({short(d["generator"]) for d in eli5 if d["label"] == "ai"})

e_ai = lambda sp: [(d["text_norm"], short(d["generator"])) for d in eli5 if d["split"] == sp and d["label"] == "ai"]
e_hu = lambda sp: [d["text_norm"] for d in eli5 if d["split"] == sp and d["label"] == "human"]
a_ai = lambda sp: [(d["text_norm"], short(d["generator"])) for d in aita if d["split"] == sp and d["label"] == "ai"]
a_hu = lambda sp: [d["text_norm"] for d in aita if d["split"] == sp and d["label"] == "human"]

# ---------- sample counts (all available AI docs, every split) ----------
e_by = collections.Counter(short(d["generator"]) for d in eli5 if d["label"] == "ai")
a_by = collections.Counter(short(d["generator"]) for d in aita if d["label"] == "ai")
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
te = a_ai("train") + a_ai("dev") + a_ai("test")
pred, y, _ = per_class(v, clf, te)
print("\nSINGLE-DOMAIN (train ELI5 only) -> AITA, recall per model:")
for m in models:
    idx = [i for i, g in enumerate(y) if g == m]
    print(f"    {m:24} {np.mean([pred[i]==m for i in idx])*100:5.1f}%")

# ---------- (1)+(3) both-domain attributor ----------
ai_tr = e_ai("train") + a_ai("train")
per = len(ai_tr) // len(models)
train = ai_tr + [(t, "human") for t in rng.sample(e_hu("train") + a_hu("train"), per)]
v, clf = fit(train)
mk = lambda ai, hu: ai + [(t, "human") for t in rng.sample(hu, max(1, len(ai) // len(models)))]
etest = mk(e_ai("test"), e_hu("test")); atest = mk(a_ai("test"), a_hu("test")); comb = etest + atest
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
allai = e_ai("train") + e_ai("test") + a_ai("train") + a_ai("test")
allhu = e_hu("train") + e_hu("test") + a_hu("train") + a_hu("test")
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
print(f"    mean evasion (unseen 2026 model called human): {np.mean(ev)*100:.1f}%"
      f"   -> caught as AI {100-np.mean(ev)*100:.1f}%")
print("    (same BoW floor on OLDER models in the OOD probe missed 96.3% of")
print("     GPT-4/3.5, 93.8% of LLaMA/OPT, 91.7% of llama-chat/MPT -- so this")
print("     method catches an unseen 2026 model ~19/20 and misses a 2022 one ~19/20)")

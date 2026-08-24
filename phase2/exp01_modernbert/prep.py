#!/usr/bin/env python3
"""Build the exp01 dataset: v6-only, question-paired, 3 complete generator triples.

Emits one JSONL of documents plus a splits.json describing 12 experiment cells.
Deterministic: question assignment is seeded on a fixed constant, so re-running
reproduces the same split exactly.
"""
import json, glob, collections, random, re, hashlib, os

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'phase1')
SEED = 20260822
TEST_FRAC = 0.30
EXCL = ('answers_prededup.jsonl', 'iter01_INVALID')

def norm_ws(t):
    """Whitespace normalisation: collapse every run of whitespace to one space."""
    return re.sub(r'\s+', ' ', t).strip()

# ---- gather v6 AI answers -------------------------------------------------
rows = []
for f in glob.glob(f'{ROOT}/study/**/answers*.jsonl', recursive=True) + \
         glob.glob(f'{ROOT}/judge/**/answers*.jsonl', recursive=True):
    if any(e in f for e in EXCL):
        continue
    for l in open(f):
        d = json.loads(l)
        if d.get('prompt_version') == 'v6':
            rows.append(d)

hum = {}
for p in ['data/interim/questions_1000.jsonl', 'data/interim/questions_longform.jsonl']:
    for l in open(f'{ROOT}/{p}'):
        d = json.loads(l)
        hum[d['q_id']] = d

# ---- keep only questions with all three generators and a human answer ------
cnt = collections.Counter(d['q_id'] for d in rows)
core_q = sorted(q for q, c in cnt.items() if c == 3 and q in hum)
core_qs = set(core_q)

docs = []
for d in rows:
    if d['q_id'] not in core_qs:
        continue
    docs.append({'doc_id': f"ai-{d['q_id']}-{d['model'].split('/')[-1]}",
                 'q_id': d['q_id'], 'label': 'ai', 'source': d['model'],
                 'question': d['question'], 'text': d['text'],
                 'text_norm': norm_ws(d['text']), 'words': d['words'],
                 'newlines': d['text'].count('\n')})
for q in core_q:
    h = hum[q]
    docs.append({'doc_id': f'hu-{q}', 'q_id': q, 'label': 'human', 'source': 'HUMAN',
                 'question': h['question'], 'text': h['human_answer'],
                 'text_norm': norm_ws(h['human_answer']), 'words': h['human_words'],
                 'newlines': h['human_answer'].count('\n')})

# ---- question-level split (never document-level: a question's human answer
#      and all three AI answers must land on the same side) -----------------
rng = random.Random(SEED)
shuf = core_q[:]
rng.shuffle(shuf)
n_test = round(len(shuf) * TEST_FRAC)
q_test = sorted(shuf[:n_test])
q_train = sorted(shuf[n_test:])
assert not (set(q_test) & set(q_train))

GENS = sorted({d['source'] for d in docs if d['label'] == 'ai'})

# ---- three tasks ----------------------------------------------------------
# A  4-way: human / grok / deepseek / qwen. Natively balanced -- 294 per class,
#    every question contributing exactly one document to each. No reweighting.
# B  unseen-generator assignment: train 3-way (human + two generators), test the
#    held-out generator. It MUST be assigned to one of the seen classes; which
#    one it picks is a direct measurement of model similarity. This is LOGO
#    reinterpreted for a multiclass label set -- you cannot ask a classifier to
#    name a class it has never seen, but you can ask who it mistakes it for.
# C  binary ai/human, seen vs unseen -- kept for continuity with the binary
#    pre-registration already on record.
qtr, qte = set(q_train), set(q_test)
def ids(pred):  return sorted(d['doc_id'] for d in docs if pred(d))
cells = []

for tf in ('text', 'text_norm'):
    t = 'raw' if tf == 'text' else 'norm'
    cells.append({'task': 'A_4way', 'cell': f'4way|{t}', 'text_field': tf,
                  'label_key': 'source', 'classes': ['HUMAN'] + GENS,
                  'train_ids': ids(lambda d: d['q_id'] in qtr),
                  'test_ids':  ids(lambda d: d['q_id'] in qte)})

for g in GENS:
    seen_cls = ['HUMAN'] + [x for x in GENS if x != g]
    for tf in ('text', 'text_norm'):
        t = 'raw' if tf == 'text' else 'norm'
        cells.append({'task': 'B_unseen_assign', 'cell': f"assign|{g.split('/')[-1]}|{t}",
                      'heldout_generator': g, 'text_field': tf, 'label_key': 'source',
                      'classes': seen_cls,
                      'train_ids': ids(lambda d: d['q_id'] in qtr and d['source'] != g),
                      'test_ids':  ids(lambda d: d['q_id'] in qte and d['source'] == g)})

n_train_ai = 2 * len(q_train)
for g in GENS:
    test_ids = ids(lambda d: d['q_id'] in qte and (d['source'] == g or d['label'] == 'human'))
    for cond in ('unseen', 'seen'):
        pool = ids(lambda d: d['label'] == 'ai' and d['q_id'] in qtr
                   and (cond == 'seen' or d['source'] != g))
        r = random.Random(f"{SEED}|{g}|{cond}")
        train = sorted(r.sample(pool, n_train_ai)) + ids(
            lambda d: d['label'] == 'human' and d['q_id'] in qtr)
        for tf in ('text', 'text_norm'):
            t = 'raw' if tf == 'text' else 'norm'
            cells.append({'task': 'C_binary', 'cell': f"{g.split('/')[-1]}|{cond}|{t}",
                          'heldout_generator': g, 'condition': cond, 'text_field': tf,
                          'label_key': 'label', 'classes': ['human', 'ai'],
                          'train_ids': train, 'test_ids': sorted(test_ids)})

out = os.path.dirname(os.path.abspath(__file__))
with open(f'{out}/docs.jsonl', 'w') as fh:
    for d in docs: fh.write(json.dumps(d) + '\n')
with open(f'{out}/splits.json', 'w') as fh:
    json.dump({'seed': SEED, 'test_frac': TEST_FRAC, 'q_train': q_train, 'q_test': q_test,
               'generators': GENS, 'n_train_ai_per_cell': n_train_ai, 'cells': cells}, fh, indent=1)

sha = hashlib.sha256(open(f'{out}/docs.jsonl','rb').read()).hexdigest()[:16]
print(f"docs.jsonl   {len(docs)} documents  sha256:{sha}")
print(f"  per class: " + "  ".join(f"{c}={sum(1 for d in docs if d['source']==c)}"
                                    for c in ['HUMAN']+GENS))
print(f"questions    {len(core_q)}  ->  train {len(q_train)}  test {len(q_test)}")
import collections as _c
for t, n in _c.Counter(c['task'] for c in cells).items():
    print(f"  {t:18s} {n} cells")
print(f"cells        {len(cells)} total")

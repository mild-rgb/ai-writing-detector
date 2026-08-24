#!/usr/bin/env python3
"""Build the phase 3 detector dataset from phase3/corpus.

Reads `dataset.jsonl` -- c4's finished, authoritative corpus artifact: one row
per document, human and AI already paired 1:1 by question, split already
assigned so no question straddles. Emits `docs.jsonl` (the same documents plus
the normalised text field and derived surface counts), `splits.json` (the
experiment cells) and `dataset_card.json` (hashes and composition).

Deterministic: no RNG at all. The split is read, never re-drawn.

Invariants this file exists to enforce, checked on every build:

1. TWO SOURCES OF SPLIT TRUTH MUST AGREE. `dataset.jsonl` carries a `split`
   field and `train/dev/test.jsonl` carry the membership. They are checked
   against each other. An earlier corpus build had a third source -- a stale
   `split` stamped into `answers.jsonl` at generation time -- which disagreed on
   73 rows and would have moved held-out documents into training. That field has
   since been removed at source; this check is what would catch its return.

2. QUESTION GROUPING. A question's human answer and its AI answer share the
   question and are about the same thing. They must never straddle a split.

3. PAIRING. Exactly one human and one AI document per question, or the class
   balance this corpus was built for does not hold.
"""
import json, os, re, hashlib, argparse, collections, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.normpath(os.path.join(HERE, '..', 'corpus'))

# Documented, non-fatal ceiling for stale-split rows in answers.jsonl. Raising
# this is a deliberate act, not a shrug -- see invariant 1 above.
MAX_STALE_SPLIT_ROWS = 200


def norm_ws(t):
    """Collapse every run of whitespace to a single space.

    The detector-side normalisation. Phase 2 proved the model gets *better* and
    far more stable when the whitespace shortcut is removed, so `norm` is the
    headline condition and `raw` is run alongside only to measure how much of any
    result is whitespace.
    """
    return re.sub(r'\s+', ' ', t).strip()


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def id_hash(ids):
    """Hash of a sorted id list -- pins a cell's membership independently of the
    file hash, so a cell can be verified even if docs.jsonl grows."""
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()[:16]


def build(corpus=CORPUS, out=HERE, require_complete=False):
    dpath = f'{corpus}/dataset.jsonl'
    rows = [json.loads(l) for l in open(dpath)]

    # --- invariant 1: the split field and the split files must agree --------
    split_files = {}
    for name in ('train', 'dev', 'test'):
        fp = f'{corpus}/{name}.jsonl'
        if not os.path.exists(fp):
            raise SystemExit(f'missing {fp}: the split files are the second source '
                             f'of truth and their absence is not a thing to shrug at')
        split_files[name] = {json.loads(l)['doc_id'] for l in open(fp)}
    for name, ids in split_files.items():
        from_field = {r['doc_id'] for r in rows if r['split'] == name}
        if ids != from_field:
            raise SystemExit(
                f'{name}: {name}.jsonl and the `split` field in dataset.jsonl '
                f'disagree on {len(ids ^ from_field)} document(s). Two sources of '
                f'split truth must agree before anything trains.')

    # --- invariant 3: pairing ----------------------------------------------
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r['question_id']].append(r)
    unpaired = [q for q, v in byq.items()
                if len(v) != 2 or {x['label'] for x in v} != {'human', 'ai'}]
    if unpaired:
        raise SystemExit(f'{len(unpaired)} question(s) are not one human + one AI: '
                         f'{unpaired[:5]}')

    # --- invariant 2: question grouping ------------------------------------
    straddle = [q for q, v in byq.items() if len({x['split'] for x in v}) != 1]
    if straddle:
        raise SystemExit(f'{len(straddle)} question(s) straddle a split: {straddle[:5]}')

    if require_complete and (len(rows) < 2 or len(byq) * 2 != len(rows)):
        raise SystemExit(f'corpus looks truncated: {len(rows)} rows over '
                         f'{len(byq)} questions')

    docs = []
    for r in rows:
        gen = 'HUMAN' if r['label'] == 'human' else r['model']
        docs.append({
            'doc_id': r['doc_id'], 'item_id': r['question_id'], 'q_id': r['q_id'],
            'label': r['label'], 'generator': gen, 'split': r['split'],
            'question': r.get('question'), 'text': r['text'],
            'text_norm': norm_ws(r['text']), 'words': r['words'],
            'newlines': r['text'].count('\n'), 'chars': len(r['text']),
            'prompt': r.get('prompt'), 'target_words': r.get('target_words'),
            'gen_seed': r.get('seed')})

    docs.sort(key=lambda d: d['doc_id'])
    empty = [d['doc_id'] for d in docs if not d['text_norm']]
    assert not empty, f'empty documents after normalisation: {empty[:5]}'

    GENS = sorted({d['generator'] for d in docs if d['label'] == 'ai'})
    by = lambda pred: sorted(d['doc_id'] for d in docs if pred(d))
    gen_of = {d['doc_id']: d['generator'] for d in docs}
    q_of = {d['doc_id']: d['q_id'] for d in docs}
    ai_gen_of_item = {d['item_id']: d['generator'] for d in docs if d['label'] == 'ai'}

    cells = []
    # --- cell family 1: the headline binary detector ----------------------
    # All seven generators seen in training. Reported PER GENERATOR on the test
    # split, never as a pooled rate -- detection ranges 7-80% across models and a
    # pooled number is an average of things that are not alike.
    for tf in ('text', 'text_norm'):
        t = 'raw' if tf == 'text' else 'norm'
        cells.append({
            'task': 'binary', 'cell': f'binary|{t}', 'text_field': tf,
            'label_key': 'label', 'classes': ['human', 'ai'],
            'heldout_generator': None,
            'train_ids': by(lambda d: d['split'] == 'train'),
            'val_ids':   by(lambda d: d['split'] == 'dev'),
            'test_ids':  by(lambda d: d['split'] == 'test')})

    # --- cell family 2: leave-one-generator-out ---------------------------
    # The real generalization test. The held-out generator's questions are
    # removed WHOLE -- both the AI answer and its paired human answer -- so the
    # test set is naturally balanced and nothing about those questions was seen.
    # Test is therefore every document of those questions, across all three
    # original splits, which is ~7x more evaluation data than a test-split-only
    # holdout would give.
    for g in GENS:
        short = g.split('/')[-1]
        held_items = {i for i, gg in ai_gen_of_item.items() if gg == g}
        for tf in ('text', 'text_norm'):
            t = 'raw' if tf == 'text' else 'norm'
            cells.append({
                'task': 'logo', 'cell': f'logo|{short}|{t}', 'text_field': tf,
                'label_key': 'label', 'classes': ['human', 'ai'],
                'heldout_generator': g,
                'train_ids': by(lambda d: d['split'] == 'train'
                                and d['item_id'] not in held_items),
                'val_ids':   by(lambda d: d['split'] == 'dev'
                                and d['item_id'] not in held_items),
                'test_ids':  by(lambda d: d['item_id'] in held_items)})

    # --- per-cell integrity: no question may appear on two sides ----------
    for c in cells:
        qtr = {q_of[i] for i in c['train_ids']}
        qva = {q_of[i] for i in c['val_ids']}
        qte = {q_of[i] for i in c['test_ids']}
        assert not (qtr & qte), f"{c['cell']}: TRAIN/TEST QUESTION LEAK"
        assert not (qtr & qva), f"{c['cell']}: TRAIN/VAL QUESTION LEAK"
        assert not (qva & qte), f"{c['cell']}: VAL/TEST QUESTION LEAK"
        if c['task'] == 'logo':
            g = c['heldout_generator']
            assert not any(gen_of[i] == g for i in c['train_ids'] + c['val_ids']), \
                f"{c['cell']}: held-out generator present in train/val"
            assert {gen_of[i] for i in c['test_ids']} == {'HUMAN', g}, \
                f"{c['cell']}: test set is not purely the held-out generator + humans"
        for k in ('train_ids', 'val_ids', 'test_ids'):
            assert len({gen_of[i] for i in c[k]}) >= 2, f"{c['cell']}: {k} single-class"
        c['id_hashes'] = {k: id_hash(c[k]) for k in ('train_ids', 'val_ids', 'test_ids')}
        c['n'] = {k: len(c[k]) for k in ('train_ids', 'val_ids', 'test_ids')}

    outd, outs = f'{out}/docs.jsonl', f'{out}/splits.json'
    with open(outd, 'w') as fh:
        for d in docs:
            fh.write(json.dumps(d, sort_keys=True) + '\n')
    with open(outs, 'w') as fh:
        json.dump({'generators': GENS, 'n_docs': len(docs),
                   'n_questions': len(byq), 'cells': cells}, fh, indent=1,
                  sort_keys=True)

    # --- dataset card -----------------------------------------------------
    wc = collections.defaultdict(list)
    for d in docs:
        wc[d['generator']].append(d['words'])
    # length was recalibrated per generator; VERIFY the claim rather than repeat
    # it. A per-model length shortcut is exactly the kind of thing that quietly
    # becomes "the detector".
    hum_med = statistics.median(wc['HUMAN'])
    card = {
        'docs_sha256': sha(outd), 'splits_sha256': sha(outs),
        'docs_sha256_short': sha(outd)[:16], 'splits_sha256_short': sha(outs)[:16],
        'corpus_dataset_sha256_short': sha(dpath)[:16],
        'corpus_split_files_sha256_short': {n: sha(f'{corpus}/{n}.jsonl')[:16]
                                            for n in ('train', 'dev', 'test')},
        'n_docs': len(docs), 'n_questions': len(byq), 'n_cells': len(cells),
        'complete': True,
        'split_counts': dict(collections.Counter(d['split'] for d in docs)),
        'generator_counts': dict(collections.Counter(
            d['generator'] for d in docs if d['label'] == 'ai')),
        'generator_x_split': {g: dict(collections.Counter(
            d['split'] for d in docs if d['generator'] == g)) for g in GENS},
        'words': {g: {'median': statistics.median(v), 'mean': round(sum(v)/len(v), 1),
                      'max': max(v), 'n': len(v)} for g, v in sorted(wc.items())},
        'words_ratio_to_human_median': {
            g: round(statistics.median(v) / hum_med, 3) for g, v in sorted(wc.items())},
        'newlines_median': {g: statistics.median(
            [d['newlines'] for d in docs if d['generator'] == g])
            for g in ['HUMAN'] + GENS},
        'norm_removes_all_newlines': not any('\n' in d['text_norm'] for d in docs),
    }
    with open(f'{out}/dataset_card.json', 'w') as fh:
        json.dump(card, fh, indent=1, sort_keys=True)
    return docs, cells, card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=CORPUS)
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--require-complete', action='store_true',
                    help='fail unless every question has an answer or a recorded failure')
    a = ap.parse_args()
    docs, cells, card = build(a.corpus, a.out, a.require_complete)

    print(f"docs.jsonl    {card['n_docs']} documents   sha256:{card['docs_sha256_short']}")
    print(f"splits.json   {card['n_cells']} cells       sha256:{card['splits_sha256_short']}")
    print(f"questions     {card['n_questions']}   COMPLETE={card['complete']}")
    print(f"corpus        dataset.jsonl:{card['corpus_dataset_sha256_short']}")
    print(f"norm removes every newline: {card['norm_removes_all_newlines']}")
    print(f"split counts  {card['split_counts']}")
    print('\nper generator (AI docs)      train / dev / test   med words  x human')
    for g, n in sorted(card['generator_counts'].items(), key=lambda x: -x[1]):
        sp = card['generator_x_split'][g]
        print(f"  {g:34s} {n:4d}  {sp.get('train',0):4d}/{sp.get('dev',0):3d}/"
              f"{sp.get('test',0):3d}   {card['words'][g]['median']:7.0f}"
              f"   {card['words_ratio_to_human_median'][g]:.3f}")
    print(f"  {'HUMAN':34s} {sum(1 for d in docs if d['label']=='human'):4d}"
          f"                    {card['words']['HUMAN']['median']:7.0f}   1.000")
    print('\ncells:')
    for c in cells:
        print(f"  {c['cell']:28s} train {c['n']['train_ids']:5d}  val {c['n']['val_ids']:4d}"
              f"  test {c['n']['test_ids']:5d}   {c['id_hashes']['test_ids']}")


if __name__ == '__main__':
    main()

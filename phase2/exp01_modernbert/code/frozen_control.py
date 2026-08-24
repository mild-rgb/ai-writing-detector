"""LFM2.5 frozen a4_generic control on the exp01 test documents.

Zero-shot, unfitted, NOTHING is trained or tuned here. Reproduces the phase 1
`score_fast` path exactly: run the body, then lm_head on the read position only;
RIGHT padding (LFM2's causal conv makes left padding a correctness bug);
margin = logsumexp(logP[yes]) - logsumexp(logP[no]); frozen threshold from
frozen_config.json, not refitted.
"""
import json, os, sys, time, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import auc, balanced_accuracy, stratified_auc

def build_prompts(tok, template, items):
    out = []
    for it in items:
        body = template.replace("{question}", it["question"]).replace("{text}", it["text"])
        out.append(tok.apply_chat_template([{"role": "user", "content": body}],
                                           tokenize=False, add_generation_prompt=True))
    return out

@torch.inference_mode()
def score_fast(model, tok, template, items, pos_ids, neg_ids, tok_budget=2048):
    prompts = build_prompts(tok, template, items)
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    lens = [len(e) for e in enc]
    order = sorted(range(len(enc)), key=lambda i: -lens[i])
    batches, cur = [], []
    for i in order:
        trial = cur + [i]
        if cur and len(trial) * lens[trial[0]] > tok_budget:
            batches.append(cur); cur = [i]
        else:
            cur = trial
    if cur: batches.append(cur)
    margins = [None]*len(items)
    real = sum(lens); t0 = time.time()
    for b in batches:
        mx = max(lens[i] for i in b)
        ids = torch.full((len(b), mx), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(b), mx), dtype=torch.long)
        for r, i in enumerate(b):                      # RIGHT padding, deliberately
            ids[r, :lens[i]] = torch.tensor(enc[i]); att[r, :lens[i]] = 1
        ids, att = ids.cuda(), att.cuda()
        h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
        pos = torch.tensor([lens[i]-1 for i in b], device=h.device)
        h1 = h[torch.arange(len(b), device=h.device), pos]
        lp = torch.log_softmax(model.lm_head(h1).float(), dim=-1)
        m = (torch.logsumexp(lp[:, pos_ids], -1) - torch.logsumexp(lp[:, neg_ids], -1)).tolist()
        for r, i in enumerate(b): margins[i] = m[r]
    dt = time.time()-t0
    return margins, {"n": len(items), "seconds": round(dt, 2),
                     "prefill_tok_s": round(real/dt), "items_s": round(len(items)/dt, 1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', default='/content/exp01/docs.jsonl')
    ap.add_argument('--splits', default='/content/exp01/splits.json')
    ap.add_argument('--config', default='/content/exp01/frozen_config.json')
    ap.add_argument('--out', default='/content/exp01/frozen_control.json')
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    import hashlib
    assert hashlib.sha256(cfg['template'].encode()).hexdigest()[:16] == cfg['prompt_sha'], \
        'prompt_sha mismatch'
    print(f"prompt_sha {cfg['prompt_sha']} verified; threshold {cfg['threshold']:.4f} (frozen)")

    docs = {d['doc_id']: d for d in map(json.loads, open(a.docs))}
    S = json.load(open(a.splits))
    tok = AutoTokenizer.from_pretrained(cfg['model'])
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model'], torch_dtype=torch.bfloat16, attn_implementation=cfg['attn']).cuda().eval()

    # score every test document once per text field
    test_ids = sorted({i for c in S['cells'] for i in c['test_ids']})
    print(f"{len(test_ids)} distinct test documents")
    M = {}
    for field in ('text', 'text_norm'):
        items = [{'question': docs[i]['question'], 'text': docs[i][field]} for i in test_ids]
        m, st = score_fast(model, tok, cfg['template'], items,
                           cfg['pos_tokens'], cfg['neg_tokens'], cfg['tok_budget'])
        print(f"  {field:10s} {st}")
        M[field] = dict(zip(test_ids, m))

    out = {'config': {k: cfg[k] for k in ('model', 'axis', 'prompt_sha', 'threshold',
                                          'tok_budget', 'pad_side', 'dtype')},
           'margins': M, 'by_generator': {}, 'pooled': {}, 'cells': {}}
    thr = cfg['threshold']

    # The unfitted binary reference. With Task C dropped this is the ONLY
    # AI-vs-human AUC on these test questions, and the one comparable to phase 1's
    # 0.771 and to wc -l. It is NOT a substitute for C's fitted number - it is the
    # unfitted control, which is a different quantity.
    q_test = set(S['q_test'])
    hum = [i for i in test_ids if docs[i]['label'] == 'human']
    gens = sorted({docs[i]['source'] for i in test_ids if docs[i]['label'] == 'ai'})
    print(f"\n{'':22s} {'AUC':>7} {'balAcc':>8} {'stratAUC(k=3)':>14}   (n = 88 ai + 88 human)")
    for field in ('text', 'text_norm'):
        tag = 'raw' if field == 'text' else 'norm'
        print(f"  -- {tag}")
        for g in gens:
            ids = [i for i in test_ids if docs[i]['source'] == g] + hum
            sc = [M[field][i] for i in ids]
            lab = [docs[i]['label'] for i in ids]
            r = {'auc': auc(sc, lab), 'balanced_acc': balanced_accuracy(sc, lab, thr),
                 'n': len(ids)}
            if field == 'text':
                sa, kept, _ = stratified_auc(sc, lab, [docs[i]['newlines'] for i in ids], k=3)
                r['strat_auc_k3'] = sa
            out['by_generator'][f'{g}|{tag}'] = r
            print(f"     {g.split('/')[-1]:18s} {r['auc']:>7.3f} {r['balanced_acc']:>8.3f} "
                  f"{r.get('strat_auc_k3', float('nan')):>14.3f}")
        # pooled: all 264 AI against the same 88 humans
        ids = [i for i in test_ids if docs[i]['label'] == 'ai'] + hum
        sc = [M[field][i] for i in ids]; lab = [docs[i]['label'] for i in ids]
        pr = {'auc': auc(sc, lab), 'balanced_acc': balanced_accuracy(sc, lab, thr), 'n': len(ids)}
        out['pooled'][tag] = pr
        print(f"     {'POOLED (264 ai)':18s} {pr['auc']:>7.3f} {pr['balanced_acc']:>8.3f}")

    # per-cell, for whatever cells exist in splits.json (C may be absent)
    for c in S['cells']:
        if c['task'] != 'C_binary':
            continue
        f = c['text_field']; ids = c['test_ids']
        sc = [M[f][i] for i in ids]; lab = [docs[i]['label'] for i in ids]
        out['cells'][c['cell']] = {'auc': auc(sc, lab),
                                   'balanced_acc': balanced_accuracy(sc, lab, thr)}
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f"\nwrote {a.out}")
    print("phase 1 reference: LFM2.5 long3 AUC 0.771 | wc -l 0.803")

if __name__ == '__main__':
    main()

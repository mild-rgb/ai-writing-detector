"""Score a single-document benchmark with a sub-1.7B local model on one GPU.

Runs LiquidAI/LFM2.5-1.2B-Instruct (1.17B params) as a detector and emits one
real-valued margin per document. Written for a Colab L4; needs ~13GB of VRAM at
the default batch budget and no packages beyond torch + transformers.

    python3 scripts/15_run_small.py study/small/items_long2.jsonl out.json

THE SCORING PASS GENERATES ZERO TOKENS. The verdict is read out of the logits at
the last prompt position, so the decode path is never entered and the cost of an
item is exactly its prefill. Two things about the implementation are not
incidental:

* `lm_head` is applied ONLY to the position being read. The stock forward
  projects every position through a 1024x65536 head, computing B*L*65536 logits
  to use B*65536 of them; slicing the hidden states first is worth ~9% and
  removes the memory ceiling that otherwise OOMs above a 32k token batch.

* Batch shape changes the answer. LFM2 runs in bf16 and GEMM reduction order
  varies with batch dimensions, so the same document scores up to ~0.4 nats
  differently depending on what it was batched with. It is deterministic for a
  FIXED shape (eight identical rows in one batch agree exactly), and it moves
  AUC by at most 0.006 across every configuration tested -- but it can move a
  tuned threshold, so `--tok-budget` is part of the frozen config and must match
  between the run that tuned the threshold and the run that applies it.

Padding is on the right. The usual decoder-only habit of left padding is wrong
here: a causal convolution slides a backward-looking window, so pads placed
before the text enter that window.
"""
import argparse
import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"

# The frozen axis. Note what it is NOT: it does not ask the model which label to
# assign. Seven classification-style prompts were tried and all seven ranked the
# real humans as more machine-made than the machines -- including one that
# listed only human cues and one that explicitly warned against the trap. The
# model reads the features correctly and maps them to labels backwards. So it is
# asked a descriptive question and the mapping is done here, in code.
TEMPLATE = (
    "A comment posted in reply to a question on r/explainlikeimfive.\n"
    "===DOC===\nQUESTION: {question}\n\nCOMMENT:\n{text}\n===END===\n"
    "Would this comment be just as good an answer to a slightly different question\n"
    "on the same topic? Answer yes or no."
)
POS_WORDS, NEG_WORDS = ("yes", "Yes"), ("no", "No")   # margin = logP(yes)-logP(no)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", help="jsonl with item/question/text per line")
    ap.add_argument("out", help="where to write {item: [...], margin: [...]}")
    ap.add_argument("--tok-budget", type=int, default=2048)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.items)]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    pos = [tok.encode(w, add_special_tokens=False)[0] for w in POS_WORDS]
    neg = [tok.encode(w, add_special_tokens=False)[0] for w in NEG_WORDS]

    enc = []
    for it in items:
        body = TEMPLATE.replace("{question}", it["question"]).replace("{text}", it["text"])
        p = tok.apply_chat_template([{"role": "user", "content": body}],
                                    tokenize=False, add_generation_prompt=True)
        enc.append(tok(p, add_special_tokens=False)["input_ids"])
    lens = [len(e) for e in enc]

    # Length-sorted batching. Counter-intuitively a SMALL budget is faster here:
    # batches then hold near-identical lengths and pad almost nothing, and the
    # padding saved outweighs the larger kernels a big batch would buy.
    order = sorted(range(len(enc)), key=lambda i: -lens[i])
    batches, cur = [], []
    for i in order:
        trial = cur + [i]
        if cur and len(trial) * lens[trial[0]] > a.tok_budget:
            batches.append(cur); cur = [i]
        else:
            cur = trial
    if cur:
        batches.append(cur)

    margins = [None] * len(items)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        for b in batches:
            mx = max(lens[i] for i in b)
            ids = torch.full((len(b), mx), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(b), mx), dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, :lens[i]] = torch.tensor(enc[i]); att[r, :lens[i]] = 1
            h = model.model(input_ids=ids.cuda(),
                            attention_mask=att.cuda()).last_hidden_state
            at = torch.tensor([lens[i] - 1 for i in b], device=h.device)
            lp = torch.log_softmax(
                model.lm_head(h[torch.arange(len(b), device=h.device), at]).float(), dim=-1)
            m = (torch.logsumexp(lp[:, pos], -1) - torch.logsumexp(lp[:, neg], -1)).tolist()
            for r, i in enumerate(b):
                margins[i] = m[r]
    torch.cuda.synchronize(); dt = time.time() - t0

    total = sum(lens)
    json.dump({"item": [i["item"] for i in items], "margin": margins,
               "model": a.model, "tok_budget": a.tok_budget,
               "prefill_tok_s": total / dt, "decode_tokens": 0, "seconds": dt},
              open(a.out, "w"))
    print(f"{len(items)} items in {dt:.2f}s | {total/dt:,.0f} prefill tok/s | "
          f"{len(items)/dt:.1f} items/s | 0 decode tokens -> {a.out}")


if __name__ == "__main__":
    main()

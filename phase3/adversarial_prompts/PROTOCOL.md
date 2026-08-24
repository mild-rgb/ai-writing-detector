# Six prompts, six mechanisms, one frozen blend

## The objective, stated precisely

Strip the **blatantly obvious surface tells** from AI-generated long-form ELI5
answers, by rate-matching to the measured human base rates rather than by
prohibition, so that a detector is forced onto deep textual patterns instead of
counting line breaks.

**This is not an undetectability exercise.** Sonnet and Pangram are expected to
keep catching this text, and that is the intended outcome. Success is that the
*trivial* baselines stop working.

Concretely, the trivial baselines are:

| baseline | what it is | phase 1 value on long form |
|---|---|---|
| `wc -l` | newline count, one fitted integer | AUC 0.80, balanced accuracy 74.0% |
| paragraph count | blank-line-separated blocks | AUC 0.77 |
| words per paragraph | mean block length | AUC 0.79 |
| word count | already pinned per question since v3 | AUC 0.49 |
| marker rates | 35 mechanical regexes over 5 families | mixed |

On phase 1's `long2`, `wc -l` beats Haiku 4.5 (74.0% against 57.3%) and loses
only to Sonnet. That is the thing being removed.

## Why six, and what makes them distinct

Each prompt is a different **mechanism** aimed at the same measurable target.
They are not six wordings of one idea, and they are not six prompts each
handling a different tell -- that would leave every one of them failing five of
the six criteria. Each must knock down all the baselines; they differ in route.

| prompt | route to structure |
|---|---|
| `p1_pinned` | explicit numeric target drawn from the question's own human answer |
| `p2_mechanical` | no instruction at all; re-wrapped after generation |
| `p3_shape` | a described shape drawn from the human distribution, no numbers |
| `p4_voice` | structure as a consequence of who is speaking |
| `p5_disfluency` | structure as a by-product of unplanned production |
| `p6_scope` | structure as a consequence of answering less of the question |

`p1` and `p2` are a deliberate control pair: the same target through prompting
and through transformation. Phase 1 has both outcomes on record -- an
instruction that took completely (first person, 0% -> 80%) and one that failed
twice and had to be done mechanically (em dashes). Which kind paragraph
convention is has never been measured, and §17 of `phase1/NARRATIVE.md` asks for
the prediction to be stated before running it. Each `mechanism.md` states one.

## The stopping rule

Computed by `phase3/scripts/report_card.py`, with no judge involved. A prompt
passes when, for newline count, paragraph count, mean words per paragraph and
word count:

1. pooled one-sided separability ≤ 0.55
2. every generator's own separability ≤ 0.60
3. the best two-sided rule's balanced accuracy ≤ 55%

and, across the 35 markers:

4. no marker off by more than 15 points at p < 0.05
5. no marker at exactly 0.0% whose human rate exceeds 5%

**Criterion 2 exists because a blend can cancel.** Seven models sitting on
opposite sides of the human median produce a pooled AUC near chance while each
remains perfectly separable. Demonstrated on a synthetic blend before any of
this was generated: pooled AUC 0.555, per-model 0.883 and 0.993, and a two-cut
rule at 86.2% balanced accuracy on the identical documents. Criterion 3 catches
the same failure from the other side, since a rank statistic cannot see a
two-sided rule.

**Criterion 5 is bounded, not absolute.** "Never prohibit" has no stopping point
as a rule: a marker whose human rate is 0.0% [0.0, 2.9] cannot be matched by any
draw, because every nonzero rate is as arbitrary as zero. Markers below what
n=130 can resolve are left to the prompt's prohibition, and the resulting zero
is a fingerprint only in principle -- at the n this project judges, it is
undetectable. Markers above 5% must be matched.

Stop after two consecutive versions with no significant change and record the
floor rather than fitting noise (`phase1/study/DESIGN.md`).

## The loop

Each round, for each prompt:

    generate on dev  ->  report card  +  subagent QA read   ->  pass / fail  ->  revise or freeze
                         (counts)        (reads)

The two checks answer different questions and neither substitutes for the other.

**The report card counts.** Four structural distributions and 35 marker rates
over every document in the round, per generator and pooled. It decides pass or
fail. No judge is involved.

**The QA read reads.** A fresh subagent per batch reads a stratified sample of
the round's documents -- 24 generated across the six generators plus 3 human
documents mixed in unlabelled -- and reports *defects*: mangled formatting,
incoherence, register problems, repetition, anything mechanically inserted, and
any obvious tell that is not in the 35-marker list.

This step exists because the report card passed `p2_mechanical` on every
structural statistic while its documents were torn: a markdown header split off
into its own paragraph with its title merged into the prose behind it. Nothing
in four distributions and 35 regexes can see that. A reader can, immediately.

**The QA read is not a detection pass and must never become one.** The subagent
is asked what is wrong with a document, never who wrote it, and its output never
becomes a rate. Detection judging stays where it is: Haiku x5 plus Sonnet, once,
on the frozen `bench` split. Letting an in-loop read turn into a detection metric
would reintroduce exactly the 13-point run-to-run noise the loop was designed to
keep out of its own stopping rule.

**The human documents are a credibility control, read one way only.** If a QA
pass flags genuine 2011-2019 redditors for the same defect it flags the
generated text for, that flag is a property of the reader rather than of the
output, and it does not motivate a prompt change. It is never scored as a
false-positive rate.

## The blend

Seven generators, one per lab, fixed composition across all six prompts:
grok-4.6, qwen3.8-max, deepseek-v4-pro, nemotron-3.5-lightning,
gemini-3.7-flash, gpt-5.6-luna-pro, glm-5.3.

The roster specified `google/gemini-3.7-flash:batch`; that variant is async-only
and 404s on the synchronous endpoint, so the sync model is used. Same model,
different billing tier. `nemotron-3.5-lightning` needs `reasoning.exclude`, or it
spends its entire token budget reasoning and returns an empty completion.

**Question sets.** The long-form corpus is partitioned once, permanently, in
`phase3/data/longform_partition.json`, excluding the 204 ids phase 1 touched:
`dev` 60 (every iteration), `bench` 70 (the frozen confirmation set), `heldout`
60 (untouched), `pool` 606.

**Rates come from measurement, not from phase 1.** The long-form human class
differs from the short-form class phase 1 rate-matched against: first person
56.9% not 23.3%, `Edit:` 13.8% not 5.1%, markdown emphasis 32.3%, paragraph
median 6, newline median 11. `phase3/data/human_marker_rates.json` is measured
by `markers.py` on dev+bench (n=130) and the prompt configs are generated *from*
that file by `make_configs.py` -- no rate is typed in by hand.

## Judging

140 items per prompt (70 ai, 10 per generator, one model per question; 70
distinct human documents), 28 batches of 5, **five independent passes**: one by
Claude Code subagents and four over the API.

Both routes were tested against each other and against phase 1's published
numbers on `long2` first (`phase3/study/judge_compare/RESULTS.md`). They agree on
balanced accuracy to within 2 points across three runs — but a *re-run of the
same route* moved detection by 13 points and false positives by 9. So:

- **balanced accuracy is the headline**, quoted as a mean with the SD across the
  five passes;
- detection and false-positive rates are reported underneath, never alone;
- phase-1-to-phase-3 comparisons are made on balanced accuracy only.

Sonnet gets one shot per prompt after its Haiku and surface criteria are met,
and stops immediately if it does not move. It is a stretch check, not a second
optimisation loop. Pangram is out of scope pending approval.

## Reproduce

    python3 phase3/scripts/partition_longform.py
    python3 phase3/scripts/markers.py --human-ids dev bench --json phase3/data/human_marker_rates.json
    python3 phase3/scripts/make_configs.py
    python3 phase3/scripts/generate.py phase3/adversarial_prompts/pN_name --split dev
    python3 phase3/scripts/report_card.py phase3/adversarial_prompts/pN_name/answers_dev.jsonl
    python3 phase3/scripts/generate.py phase3/adversarial_prompts/pN_name --split bench
    python3 phase3/scripts/build_bench.py phase3/adversarial_prompts/pN_name --pass 0
    python3 phase3/scripts/score_passes.py phase3/adversarial_prompts/pN_name

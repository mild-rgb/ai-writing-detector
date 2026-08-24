# Are subagent judges and API judges the same instrument?

Phase 1's published detector numbers came from Claude Code **subagent** judges,
one fresh agent per 5-item batch. Phase 3 could judge the same way or over the
**OpenRouter API** (`anthropic/claude-haiku-4.5`), one fresh HTTP request per
batch. If the two routes read the same documents differently, no phase-3 number
is comparable to a phase-1 number.

**Design.** The same 150 items -- phase 1's `long2`, the clean rebuilt long-form
benchmark -- judged three times:

| arm | route | when |
|---|---|---|
| `published` | subagent | phase 1, reported in NARRATIVE.md §14 |
| `subagent` | subagent | this session, fresh agents, identical instruction |
| `api` | OpenRouter | this session, identical instruction text |

The instruction text is byte-identical across the two fresh arms. The batch
files are the same files phase 1 handed its judges.

## Aggregate

| arm | detection | false positives | balanced |
|---|---|---|---|
| published (subagent) | 24.0% [15.8, 34.8] | 9.3% [4.6, 18.0] | 57.3% |
| api | 28.0% [19.1, 39.0] | 10.7% [5.5, 19.7] | **58.7%** |
| subagent (fresh) | 37.3% [27.3, 48.6] | 18.7% [11.5, 28.9] | **59.3%** |

Every interval overlaps every other. **Balanced accuracy spans 2.0 points across
all three arms** -- 57.3, 58.7, 59.3.

## Pairwise, item by item

McNemar on the discordant pairs, which is the correct test for two raters on the
same items; a two-proportion test would discard the pairing.

| pair | agreement | kappa | discordant | McNemar p |
|---|---|---|---|---|
| api vs published | **80.0%** | 0.323 | 17 / 13 | **0.58** |
| subagent vs published | 78.0% | 0.377 | 25 / 8 | **0.0046** |
| api vs subagent | 75.3% | 0.324 | 12 / 25 | 0.047 |

## The result, and it is not the one the test was designed to find

**The route is not the dominant source of variation. The run is.**

A fresh subagent run differs from phase 1's own subagent run **significantly**
(p = 0.0046): it calls "ai" on 28.0% of items against 16.7%, and its detection
and false-positive rates are both about 9 points higher. The API arm does *not*
differ significantly from phase 1's subagent run (p = 0.58) and lands closer to
it on both rates.

So the API route is at least as faithful to the published numbers as re-running
the published route is. Per-item agreement is 75-80% and kappa 0.32-0.38 for
**every** pair, including subagent-against-subagent. Haiku 4.5 is only a
fair-agreement instrument *with itself* on this task.

That is phase 1 §8 again from a different angle: "precision comes from judge
count, not trial count", measured there as a judge SD of 19.6 against a binomial
expectation of 5.0. Here it shows up as a same-route re-run moving a headline
rate by 13 points.

## What this changes

1. **Both routes are usable, and neither is a fixed instrument.** Quoting a
   single-pass detection rate to one decimal place is not supportable by either
   route. Balanced accuracy is the stable statistic here: 2 points of spread
   across three runs and two routes, against 13 points on detection alone.
2. **Replicates are not optional.** The phase-3 design already specified R=5
   Haiku passes per prompt with the SD across passes reported beside the mean.
   This test is the justification: a single pass could have landed anywhere in a
   13-point band.
3. **A phase-1-to-phase-3 comparison must be made on balanced accuracy**, and
   even then with the note that the same route re-run moves the components.

## Reproduce

    python3 phase3/scripts/judge_api.py phase1/study/single/long2/batches \
        phase3/study/judge_compare/api --model anthropic/claude-haiku-4.5
    python3 phase1/scripts/14_score_single.py phase3/study/judge_compare api subagent
    python3 phase3/scripts/compare_judges.py phase3/study/judge_compare api published
    python3 phase3/scripts/compare_judges.py phase3/study/judge_compare subagent published

API arm cost: 92,704 input and 9,831 output tokens, about $0.14.
Subagent arm cost: 30 agents, roughly 780,000 subagent tokens.

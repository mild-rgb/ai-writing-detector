# p2_mechanical — iteration log

Every version is kept. Each row records **one** change and the measurement that
motivated it, because phase 1's v4 changed six lines at once and v7 changed
four, and neither could be attributed to anything afterwards.

The stopping rule is in `../PROTOCOL.md` and is computed by
`report_card.py` on 60 dev questions x 7 generators. No judge is involved in
the decision to iterate.

| version | change | motivated by | pooled newline sep | pooled words sep | markers failing |
|---|---|---|---|---|---|
| v1 | the mechanism as designed (see `mechanism.md`) | prediction on record before any data | pending | pending | pending |

## Harness constants shared by all six prompts

These are not part of any prompt's mechanism and are held identical across all
six, so a difference between prompts is a difference between mechanisms:

- draws and their rates, generated from `phase3/data/human_marker_rates.json`
- rate-matched punctuation post-ops (em dash 2.3%, curly 3.9%, ellipsis 0.8%,
  double space 27.7%, lowercase paragraph 9.2%, en dash 0.0%)
- per-model length calibration and the [0.80, 1.30] rejection band
- `reasoning.exclude` for nemotron-3.5-lightning

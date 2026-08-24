# p2_mechanical — structure as a post-hoc transformation

**Mechanism.** The prompt says nothing whatsoever about paragraphs. The target
structure is imposed *after* generation by re-wrapping the text to the human
answer's own paragraph count, with cut points drawn unevenly. Sentence order and
wording are untouched; only where the blank lines fall changes.

**Why this one exists.** It is the control for p1, and the pairing is the point.
Phase 1 has both outcomes on record -- an instruction that took completely
(first person, 0% -> 80%) and an instruction that failed twice and had to be
done mechanically (em dashes, ~10% against a human 0.4% through two explicit
prohibitions). Running the same structural target through both routes on the
same questions and the same models is the only way to say which kind of
instruction this is.

**This is a transformation, not a prompting success, and it is labelled as one.**
phase1/NARRATIVE.md §4 records the em-dash substitution the same way. If p2
lands its structural target and p1 does not, the honest conclusion is that
paragraph convention is not promptable in these models, not that this prompt is
better writing.

**Prediction on record.** p2 hits the structural target by construction, so its
newline and paragraph separability should fall to near 0.50 -- it is the ceiling
for what matching this feature can buy. Whatever separability survives in p2 is
the part that was never about whitespace. That number is worth more than the
prompt is.

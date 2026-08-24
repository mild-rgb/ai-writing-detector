# p1_pinned — structure as a pinned numeric target

**Mechanism.** Paragraph count and mean paragraph length are pinned per document
to *that question's own human answer*, exactly the way `{n}` already pins word
count in every prompt since v3. Nothing else in the six does this numerically.

**Why this one exists.** Paragraph convention is the largest known defect in the
phase 1 corpus: newline AUC 0.78-0.86 on every set, and on a threshold fitted in
place `wc -l` beats every detector below Sonnet. `prompts/v6.txt` says nothing at
all about paragraphs, so this is not a prohibition driving a feature to zero --
it is an omission from the rate-matching program. phase1/NARRATIVE.md §17 calls
a prompt that fixes it "the single highest-value change left" and asks for the
prediction to be stated before running.

**Prediction on record.** Instruction-following on a numeric structural target
will be partial and model-dependent. Phase 1 has both outcomes: "write in the
first person" moved a rate 0% -> 80%, and two explicit em-dash prohibitions
moved nothing. I expect pooled newline separability to fall from ~0.80 to below
0.65 but not to 0.55, and I expect the residual to be concentrated in whichever
models are worst at following a formatting instruction. p2 is the control that
says whether the residual is the instruction or the models.

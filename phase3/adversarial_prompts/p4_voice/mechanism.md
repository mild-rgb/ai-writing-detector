# p4_voice — structure as a consequence of who is speaking

**Mechanism.** Nothing in the prompt addresses formatting at all. Instead the
model is told, concretely, that it is a person typing a reply in one sitting,
with the voice features drawn at their measured long-form human rates -- first
person 56.9%, own-words hedging 10.8%, mild profanity 13.8%, a question to the
reader 36.2%, a scrap of personal experience 1.5%, a professional credential
2.3%. Paragraph breaks are described as places a person stops to think, not as a
layout.

**Why this one exists.** phase1/NARRATIVE.md §13 found that long ELI5 answers
come disproportionately from people writing inside their own profession, and
that "professional reads as synthetic" is where the false positives come from.
The human class here is not casual writing -- it is expert writing typed
quickly. This mechanism aims at that register directly and lets structure fall
out of it.

**Prediction on record.** This is the mechanism I expect to *fail* the
structural criterion while doing well on markers. Register instructions moved
voice features cleanly in phase 1 (v5's total absolute signal gap 160.8 -> 51.0)
and moved nothing structural, because nothing structural was mentioned. If p4's
newline separability stays near 0.75 while its marker gaps go to zero, that is
the finding: voice and layout are separate axes, and matching one does not touch
the other.

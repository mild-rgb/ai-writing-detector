# p3_shape — structure as a described distribution

**Mechanism.** No numbers. Each document draws one *shape description* from the
measured human distribution -- a single block (0.8%), two or three chunks
(10.0%), the ordinary four-to-eight (67.7%), or a long many-break answer (21.5%)
-- and the prompt describes that shape in words, the way a person would describe
how a comment looks rather than how it measures.

**Why this one exists.** p1 asks for a number and p2 imposes one. This asks
whether the model can realise a *qualitative* target, which is the form most
style instructions actually take. It also tests the thing that makes the pooled
blend statistic honest: the human class is not a point, it is a distribution
with a fat middle and two tails, and a generator that always writes the median
shape is as separable as one that always writes a single block. Matching a mean
is not matching a distribution.

**Prediction on record.** I expect p3 to sit between p1 and p2 on structural
separability, and to beat both on the *variance* of the structural features --
the drawn tails should give it a spread closer to the human one, where p1's
numeric target pulls every document toward its own question's value and p2's
re-wrap reproduces exactly one target per document.

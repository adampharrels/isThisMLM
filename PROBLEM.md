# Problem Formulation

## Task
Detect whether a message is an attempt to recruit the reader into a multi-level marketing (MLM) scheme.

## Formal definition
- **Input x**: the text of a message (or one sender's messages concatenated).
- **Label y ∈ {0, 1}**: 1 = MLM recruitment attempt, 0 = anything else.
- **Model output**: p̂ = P(y = 1 | x), a probability in [0, 1].
- **Decision rule**: flag the message if p̂ ≥ t, where t is a threshold chosen from the relative cost of errors (see below).

This is binary text classification.

## Labeling rule (y = 1)
A message is positive if it invites the recipient to join, sell for, or "partner" with a
business whose income depends on recruiting others or on buying/reselling a company's
product line, typically described vaguely ("an opportunity", "a business venture") with
lifestyle or income claims and a push to a call, link, or "mentor".

Borderline cases, decided consistently:
- A friend selling their own product with no recruitment angle -> 0
- A legitimate job recruiter naming a real role and company -> 0
- Generic spam (prizes, loans, phishing) -> 0 (kept deliberately as a hard negative)
- Customer-only sales pitch for an MLM product with no recruiting angle -> 0

## Loss function
Binary cross-entropy (class-weighted):

    L = -[ w1 * y * log(p̂) + w0 * (1 - y) * log(1 - p̂) ]

- Differentiable and convex for logistic regression, so training is well-behaved.
- Gradient w.r.t. the raw score is simply (p̂ - y).
- Equivalent to maximum-likelihood estimation, so p̂ behaves like a real probability.
- Class weights w1, w0 compensate for MLM being rarer than non-MLM.

## Threshold
t = C_FP / (C_FP + C_FN). Wrongly flagging a friend (C_FP) vs missing a real pitch (C_FN).
If missing a pitch is judged twice as costly, t = 1/3.

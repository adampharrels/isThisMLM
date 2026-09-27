# isThisMLM

Detecting MLM recruitment messages as a binary text classification problem: given a
message or post, estimate the probability that it's an attempt to recruit the reader
into a multi-level marketing scheme, rather than an ordinary personal message.

See [PROBLEM.md](PROBLEM.md) for the full problem formulation (input/output spec,
labelling rule, loss function, and decision threshold).

## Repo structure

```
PROBLEM.md              problem formulation: input/output, labelling rule, loss, threshold
data/raw/
  SMSSpamCollection      UCI SMS Spam Collection (negatives: ham + spam, both label 0)
  mlm_positives.csv      hand-transcribed real MLM recruitment messages/posts (label 1)
notebooks/
  01_data.ipynb          full implementation: data prep, model, evaluation, deployment
```

## Data

- **Positives** (`data/raw/mlm_positives.csv`, n=129): real MLM recruitment messages and
  posts transcribed from r/antiMLM, tagged by `type` (`post`, `dm`, or `script`). Names
  and contact details are anonymized.
- **Negatives** (`data/raw/SMSSpamCollection`, n=5,572 before dedup): the UCI SMS Spam
  Collection, used as label 0 for both `ham` and `spam` messages.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn jupyter matplotlib
```

## Running

Open `notebooks/01_data.ipynb` and run all cells top to bottom (the `.venv` kernel).
It:

1. Loads and combines both data sources, deduplicates, and checks class balance.
2. Trains a TF-IDF + logistic regression classifier under stratified 5-fold
   cross-validation, with class-weighted loss to handle the ~2.4% positive rate.
3. Runs error analysis on the learned coefficients and reports precision, recall, and
   PR-AUC.
4. Selects a deployment threshold from the precision/recall trade-off.
5. Exposes a `predict_mlm(text)` function returning a probability, a flag, and the
   top contributing words for a single new message.
6. Runs two diagnostic checks: a hand-written hard-negative probe (non-MLM messages
   that superficially resemble a pitch) and a generalization check (training on public
   posts only, testing recall on direct messages).

## Results

5-fold cross-validation, TF-IDF + logistic regression, threshold t = 0.6:

| precision | recall | PR-AUC |
| --- | --- | --- |
| 0.90 | 0.83 | 0.90 |

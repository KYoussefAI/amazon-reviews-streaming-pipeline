# Spark Model Comparison Results

Generated at: `2026-05-07T07:38:28`

## Selection Rule

All candidate models are trained on the training split.

Models are compared on the validation split using **Macro F1**.

The test split is used only once for the model selected by validation Macro F1.

## Validation Ranking

| Rank | Model | Accuracy | Macro F1 | Positive F1 | Negative F1 | Neutral F1 | Saveable Pipeline | Supports Probability | Selected |
|---:|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 1 | majority_vote_ensemble | 0.8569 | 0.7003 | 0.9244 | 0.7318 | 0.4449 | False | False | True |
| 2 | one_vs_rest_linear_svc | 0.8818 | 0.6921 | 0.9389 | 0.7313 | 0.4061 | True | False | False |
| 3 | logistic_regression | 0.8121 | 0.6728 | 0.8954 | 0.7015 | 0.4214 | True | True | False |
| 4 | naive_bayes | 0.7761 | 0.6212 | 0.8780 | 0.6329 | 0.3527 | True | True | False |
| 5 | random_forest_light | 0.7816 | 0.2948 | 0.8773 | 0.0070 | 0.0000 | True | True | False |

## Selected Model

Selected model: `majority_vote_ensemble`

Validation Macro F1: `0.7003`

## Final Test Metrics for Selected Model

| Metric | Value |
|---|---:|
| Accuracy | 0.8579 |
| Macro F1 | 0.7023 |
| Positive F1 | 0.9256 |
| Negative F1 | 0.7330 |
| Neutral F1 | 0.4482 |

## Model Notes

- `majority_vote_ensemble`: Batch majority vote across trained model predictions.
- `one_vs_rest_linear_svc`: Linear SVC adapted to multiclass using OneVsRest.
- `logistic_regression`: Strong baseline for sparse TF-IDF text classification.
- `naive_bayes`: Fast classic text-classification baseline using multinomial Naive Bayes.
- `random_forest_light`: Light Random Forest comparison; less ideal for sparse TF-IDF text.

## Engineering Decision

Only a saveable Spark PipelineModel can directly replace the streaming model without changing `predict_stream.py`.
The majority-vote ensemble is evaluated as a batch comparison method first.

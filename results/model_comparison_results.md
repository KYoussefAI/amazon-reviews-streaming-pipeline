# Spark Model Comparison Results

Generated at: `2026-05-06T23:45:19`

## Selection Rule

All candidate models are trained on the training split.

Models are compared on the validation split using **Macro F1**.

The test split is used only once for the model selected by validation Macro F1.

## Validation Ranking

| Rank | Model | Accuracy | Macro F1 | Positive F1 | Negative F1 | Neutral F1 | Saveable Pipeline | Supports Probability | Selected |
|---:|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 1 | majority_vote_ensemble | 0.8563 | 0.6997 | 0.9241 | 0.7304 | 0.4446 | False | False | True |
| 2 | one_vs_rest_linear_svc | 0.8818 | 0.6925 | 0.9387 | 0.7302 | 0.4088 | True | False | False |
| 3 | logistic_regression | 0.8111 | 0.6718 | 0.8948 | 0.7002 | 0.4204 | True | True | False |
| 4 | naive_bayes | 0.7759 | 0.6210 | 0.8778 | 0.6327 | 0.3525 | True | True | False |
| 5 | random_forest_light | 0.7818 | 0.2956 | 0.8774 | 0.0094 | 0.0000 | True | True | False |

## Selected Model

Selected model: `majority_vote_ensemble`

Validation Macro F1: `0.6997`

## Final Test Metrics for Selected Model

| Metric | Value |
|---|---:|
| Accuracy | 0.8573 |
| Macro F1 | 0.7010 |
| Positive F1 | 0.9253 |
| Negative F1 | 0.7308 |
| Neutral F1 | 0.4469 |

## Model Notes

- `majority_vote_ensemble`: Batch majority vote across trained model predictions.
- `one_vs_rest_linear_svc`: Linear SVC adapted to multiclass using OneVsRest.
- `logistic_regression`: Strong baseline for sparse TF-IDF text classification.
- `naive_bayes`: Fast classic text-classification baseline using multinomial Naive Bayes.
- `random_forest_light`: Light Random Forest comparison; less ideal for sparse TF-IDF text.

## Engineering Decision

Only a saveable Spark PipelineModel can directly replace the streaming model without changing `predict_stream.py`.
The majority-vote ensemble is evaluated as a batch comparison method first.

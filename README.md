# Amazon Reviews Real-Time Sentiment Analysis Pipeline

End-to-end Big Data project for real-time sentiment analysis on Amazon review events using Kafka, Apache Spark, Spark ML, and downstream storage/visualization components.

The project is designed as a production-oriented learning system: first build a working pipeline, then progressively improve scalability, model quality, storage, orchestration, and monitoring.

---

## 1. Project Objective

The objective is to build a complete real-time data engineering and machine learning pipeline:

```text
Amazon Reviews Dataset
→ Kafka
→ Spark Structured Streaming
→ Spark ML Sentiment Prediction
→ MongoDB
→ Dashboard
```

The current architecture target is:

```text
Producer → Kafka → Spark → MongoDB → Dashboard
```

The project currently focuses on the Spark ML model training and the transition toward real-time streaming inference.

---

## 2. Dataset

The project uses the **Amazon Fine Food Reviews** dataset.

Expected local path:

```text
data/raw/Reviews.csv
```

Main columns used:

| Column | Description |
|---|---|
| `Text` | Review text |
| `Score` | Rating from 1 to 5 |

Sentiment labels are generated from `Score`:

| Score condition | Label |
|---|---|
| `Score < 3` | `negative` |
| `Score == 3` | `neutral` |
| `Score > 3` | `positive` |

The raw dataset is intentionally excluded from Git because it is large and can be downloaded separately.

---

## 3. Current Project Status

Completed:

- Kafka producer for streaming review events.
- Basic Kafka consumer for validation/debugging.
- Initial Python/sklearn experimentation pipeline.
- Spark-only ML training pipeline.
- Train/validation/test split.
- Class-weighted Logistic Regression.
- TF-IDF feature extraction in Spark.
- Unigram + bigram feature engineering.
- Simulated Annealing hyperparameter tuning.
- Final Spark `PipelineModel` training and local saving.
- Validation and test evaluation.

Current next phase:

```text
Phase 8 — Spark Structured Streaming inference
```

Goal of Phase 8:

```text
Kafka → Spark Structured Streaming → Load saved Spark PipelineModel → Predict sentiment
```

MongoDB integration starts after this streaming inference step is validated.

---

## 4. Repository Structure

Current production-oriented structure:

```text
BIG-DATA-PROJECT/
│
├── data/
│   └── raw/
│       └── Reviews.csv                 # local only, ignored by Git
│
├── docs/
│   ├── spark_sentiment_tuning_report.md
│   └── project_progress_for_teammates.md
│
├── kafka/
│   └── docker-compose.yml
│
├── results/
│   └── spark_sa_tuning_results.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   └── producer.py
│   │
│   ├── experiments/
│   │   ├── preprocessing/
│   │   │   ├── clean.py
│   │   │   ├── dataset.py
│   │   │   ├── label.py
│   │   │   ├── resampling.py
│   │   │   └── vectorizer.py
│   │   │
│   │   └── training/
│   │       └── train.py
│   │
│   └── spark/
│       ├── model/
│       │   └── sentiment_pipeline_model/      # local artifact, ignored by Git
│       │
│       ├── training/
│       │   ├── train_spark_pipeline.py
│       │   └── tune_spark_pipeline_sa.py
│       │
│       └── streaming/
│           ├── consumer.py
│           └── predict_stream.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Folder meaning

| Folder | Purpose |
|---|---|
| `src/ingestion/` | Kafka producer and ingestion logic |
| `src/experiments/` | Early experimentation code using Python/sklearn |
| `src/spark/training/` | Production-oriented Spark ML training and tuning |
| `src/spark/streaming/` | Spark streaming inference code |
| `src/spark/model/` | Locally saved Spark model artifact |
| `results/` | Tuning results and experiment outputs |
| `docs/` | Technical reports and project documentation |

---

## 5. Architecture Overview

### Current offline training architecture

```text
Reviews.csv
→ Spark DataFrame
→ Label generation
→ Train/validation/test split
→ Class weights
→ Spark ML Pipeline
→ Validation tuning
→ Final test evaluation
→ Save Spark PipelineModel
```

### Target online inference architecture

```text
Reviews.csv
→ Kafka Producer
→ Kafka topic: amazon_reviews
→ Spark Structured Streaming
→ Load saved Spark PipelineModel
→ Predict sentiment
→ Console output
→ MongoDB later
```

---

## 6. Kafka Ingestion

Kafka topic:

```text
amazon_reviews
```

Message format:

```json
{
  "text": "review text here",
  "score": 5
}
```

The producer reads reviews from the local CSV dataset and sends them as JSON events to Kafka.

The `score` field is included for debugging and validation. The streaming prediction model must predict sentiment from the review text, not from the score.

---

## 7. Spark ML Training Pipeline

The production-oriented model is implemented fully with Spark ML components.

Pipeline stages:

```text
text
→ RegexTokenizer
→ StopWordsRemover
→ NGram
→ CountVectorizer for unigrams
→ IDF for unigrams
→ CountVectorizer for bigrams
→ IDF for bigrams
→ VectorAssembler
→ StringIndexer
→ LogisticRegression
```

The model uses both:

```text
unigram TF-IDF features
bigram TF-IDF features
```

This was selected because using bigrams alone performed worse, while combining unigrams and bigrams improved the validation metrics significantly.

---

## 8. Labeling and Splitting

Labeling rule:

```python
Score < 3   → negative
Score == 3  → neutral
Score > 3   → positive
```

Split strategy:

```text
Train:      80%
Validation: 10%
Test:       10%
```

The model is trained only on the training set.

The validation set is used for model selection and hyperparameter tuning.

The test set is used once for final evaluation after tuning.

---

## 9. Class Imbalance Handling

The dataset is imbalanced:

```text
positive = dominant class
negative = minority class
neutral  = smallest class
```

To reduce majority-class bias, class weights are computed from the training data:

```text
class_weight = total_training_rows / (number_of_classes × class_count)
```

The observed class weights during the final 100k training run were approximately:

```text
positive -> 0.4348
negative -> 2.1826
neutral  -> 4.1318
```

These weights are passed to Spark Logistic Regression using `weightCol="class_weight"`.

---

## 10. Hyperparameter Tuning

The model was improved through manual tuning and Simulated Annealing.

Tuned parameters:

| Parameter | Meaning |
|---|---|
| `vocab_size` | Maximum vocabulary size for CountVectorizer |
| `min_df` | Minimum document frequency required to keep a term |
| `max_iter` | Maximum number of Logistic Regression optimization iterations |
| `reg_param` | Regularization strength |
| `use_bigrams` | Whether unigram + bigram features are used |

### Simulated Annealing

The tuning script is:

```text
src/spark/training/tune_spark_pipeline_sa.py
```

The tuner:

1. Starts from an initial configuration.
2. Generates a neighboring configuration.
3. Trains a Spark pipeline on the training set.
4. Evaluates it on the validation set.
5. Optimizes primarily by `macro_f1`.
6. Saves results to Markdown and CSV.

`macro_f1` was selected because accuracy alone is misleading on imbalanced data.

---

## 11. Final Selected Model

Final selected configuration:

```python
pipeline = build_pipeline(
    vocab_size=10000,
    min_df=4,
    max_iter=15,
    reg_param=0.000025,
    use_bigrams=True
)
```

This configuration was selected because it achieved the best validation macro F1 while keeping the model smaller and more efficient than larger vocabulary alternatives.

Approximate feature size:

```text
10000 unigram features + 10000 bigram features = about 20000 total features
```

---

## 12. Final Metrics

### Validation metrics

```text
Accuracy    = 0.8159
Macro F1    = 0.6470
Positive F1 = 0.9031
Negative F1 = 0.6419
Neutral F1  = 0.3961
```

### Test metrics

```text
Accuracy    = 0.8130
Macro F1    = 0.6349
Positive F1 = 0.9018
Negative F1 = 0.6558
Neutral F1  = 0.3470
```

Interpretation:

- Accuracy and positive F1 remained stable from validation to test.
- Negative F1 generalized well and improved on the test set.
- Neutral F1 remains the weakest and most unstable class.
- The model is acceptable as a production baseline before moving to streaming inference.

---

## 13. Model Artifact

The trained Spark model is saved locally at:

```text
src/spark/model/sentiment_pipeline_model
```

This folder is a generated artifact and should not be committed to Git.

It can be recreated by running:

```bash
spark-submit src/spark/training/train_spark_pipeline.py
```

---

## 14. Git and Data Rules

The following should not be committed:

```text
data/raw/Reviews.csv
src/spark/model/
src/spark/models/
*.model
__pycache__/
*.pyc
```

Recommended `.gitignore` entries:

```gitignore
# Raw data
data/raw/Reviews.csv
data/raw/*.csv

# Spark trained models
src/spark/model/
src/spark/models/
*.model

# Python cache
__pycache__/
*.pyc

# Spark local artifacts
spark-warehouse/
metastore_db/
derby.log

# Local environments
.venv/
env/
big_data_env/
```

---

## 15. Running the Training Pipeline

From the project root in WSL:

```bash
source ../env/big_data_env/bin/activate
export PYTHONPATH=$PWD
```

Run final training:

```bash
spark-submit src/spark/training/train_spark_pipeline.py
```

Clean output version:

```bash
spark-submit src/spark/training/train_spark_pipeline.py 2>/dev/null | grep -E "==========|Accuracy|Macro F1|Positive F1|Negative F1|Neutral F1|Model saved"
```

Expected output includes:

```text
Model saved to: src/spark/model/sentiment_pipeline_model
```

---

## 16. Running the Simulated Annealing Tuner

From the project root:

```bash
export PYTHONPATH=$PWD
```

Run:

```bash
spark-submit src/spark/training/tune_spark_pipeline_sa.py
```

Clean output version:

```bash
spark-submit src/spark/training/tune_spark_pipeline_sa.py 2>/dev/null | grep -E "INITIAL|SA STEP|Temperature|Current config|Candidate config|Macro F1|Accepted|BEST|Accuracy|Positive F1|Negative F1|Neutral F1|Saved"
```

Results are saved under:

```text
results/spark_sa_tuning_results.md
results/spark_sa_tuning_results.csv
```

---

## 17. Next Phase

Next implementation step:

```text
Spark Structured Streaming inference
```

Required behavior:

```text
1. Start Kafka.
2. Start Spark streaming prediction job.
3. Load saved Spark PipelineModel.
4. Read events from Kafka topic `amazon_reviews`.
5. Parse JSON messages.
6. Apply `model.transform()`.
7. Display predicted sentiment in console.
```

This phase must not train a model.

The saved model should be loaded from:

```text
src/spark/model/sentiment_pipeline_model
```

---

## 18. Future Work

Planned next phases:

| Phase | Goal |
|---|---|
| Phase 8 | Spark Structured Streaming prediction |
| Phase 9 | Store predictions in MongoDB |
| Phase 10 | Build dashboard |
| Phase 11 | Add monitoring/logging |
| Phase 12 | Add Airflow orchestration |
| Later | Test additional models and ensemble ML |

Possible future ML improvements:

- Naive Bayes.
- Linear SVC.
- One-vs-Rest classifiers.
- Ensemble voting.
- Stacking.
- Larger-scale full-dataset retraining.
- More advanced text preprocessing.

---

## 19. Current Production Baseline

The current production baseline is:

```text
Spark ML PipelineModel
TF-IDF features
Unigrams + bigrams
Class-weighted Logistic Regression
Simulated Annealing tuned hyperparameters
```

Final selected parameters:

```text
vocab_size = 10000
min_df = 4
max_iter = 15
reg_param = 0.000025
use_bigrams = True
```

The model is saved locally and ready to be loaded by the Spark streaming inference pipeline.

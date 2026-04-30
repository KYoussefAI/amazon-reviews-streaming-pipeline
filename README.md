# Amazon Reviews Streaming Pipeline

A real-time Big Data sentiment analysis project built around the following architecture:

```text
Amazon Reviews Dataset
→ Kafka
→ Spark
→ MongoDB
→ Dashboard
```

Current production direction:

```text
Producer → Kafka → Spark Structured Streaming → Saved Spark ML Model → Prediction
```

The project is developed phase by phase with two goals:

1. Build a working end-to-end data engineering system.
2. Understand every component deeply before adding complexity.

---

## 1. Project Objective

The objective is to build a complete real-time data pipeline that streams Amazon review events, processes them with Spark, predicts sentiment using a trained Spark ML model, stores the results later in MongoDB, and visualizes them through a dashboard.

The fixed architecture is:

```text
Producer → Kafka → Spark → MongoDB → Dashboard
```

Each component has a specific role:

| Component | Role |
|---|---|
| Producer | Reads Amazon reviews and sends them as JSON events |
| Kafka | Handles real-time ingestion and event streaming |
| Spark | Processes review events and applies ML prediction |
| MongoDB | Stores prediction results in later phases |
| Dashboard | Visualizes real-time and offline analytics in later phases |

---

## 2. Dataset

The project uses the **Amazon Fine Food Reviews** dataset.

Dataset source:

```text
https://www.kaggle.com/snap/amazon-fine-food-reviews
```

Expected local path:

```text
data/raw/Reviews.csv
```

Main fields used:

| Field | Description |
|---|---|
| Text | Review text |
| Score | Rating from 1 to 5 |
| Time | Review timestamp, kept for later analytics |

Sentiment labels are created using this rule:

| Score | Label |
|---|---|
| Score < 3 | negative |
| Score = 3 | neutral |
| Score > 3 | positive |

Important: the raw CSV dataset is large and must **not** be pushed to GitHub.

Recommended `.gitignore` entries:

```gitignore
data/raw/Reviews.csv
data/raw/*.csv
```

---

## 3. Current Status

Completed so far:

```text
Phase 1 — Kafka ingestion
Phase 2 — Python/sklearn preprocessing experiments
Phase 3 — Dataset creation and splitting
Phase 4 — Initial Spark ML training
Phase 5 — Validation metrics
Phase 6 — Testing
Phase 7 — Model selection
Phase 7.5 — Spark-only production training pipeline
Phase 7.6 — Hyperparameter tuning with Simulated Annealing
Phase 7.7 — Final saved Spark PipelineModel
```

Current next phase:

```text
Phase 8 — Real-time Spark streaming prediction
Kafka → Spark Structured Streaming → Load saved model → Predict sentiment
```

MongoDB storage starts later in Phase 9.

---

## 4. Current Repository Structure

Current structure after the refactor:

```text
BIG-DATA-PROJECT/
│
├── data/
│   └── raw/
│       └── Reviews.csv                  # local only, not committed
│
├── docs/
│   ├── spark_sentiment_tuning_report.md
│   └── project_progress_for_teammates.md
│
├── kafka/
│   └── docker-compose.yml
│
├── results/
│   ├── spark_sa_tuning_results.csv
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
│       │   └── sentiment_pipeline_model/ # generated locally, not committed
│       │
│       ├── training/
│       │   ├── train_spark_pipeline.py
│       │   └── tune_spark_pipeline_sa.py
│       │
│       └── streaming/
│           ├── consumer.py
│           └── predict_stream.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

Folder meaning:

| Folder | Meaning |
|---|---|
| `src/ingestion/` | Kafka producer code |
| `src/experiments/` | Earlier Python/sklearn experimentation layer |
| `src/spark/training/` | Production-oriented Spark ML training and tuning |
| `src/spark/model/` | Local saved Spark model artifact |
| `src/spark/streaming/` | Spark streaming prediction code |
| `results/` | Tuning outputs and experiment summaries |
| `docs/` | Project documentation |

---

## 5. Completed Work

### Phase 1 — Kafka Streaming

Built:

- Kafka and Zookeeper setup using Docker
- Kafka topic: `amazon_reviews`
- Producer that streams reviews from `Reviews.csv`
- Basic consumer that reads messages

Kafka event format:

```json
{
  "text": "review text here",
  "score": 5
}
```

Key idea:

Kafka decouples data ingestion from processing. The producer does not need to know how Spark or MongoDB will use the event later.

---

### Phase 2 — Preprocessing Experiments

Built Python preprocessing modules:

```text
src/experiments/preprocessing/clean.py
src/experiments/preprocessing/vectorizer.py
src/experiments/preprocessing/label.py
src/experiments/preprocessing/dataset.py
src/experiments/preprocessing/resampling.py
```

The initial preprocessing pipeline used:

- lowercasing
- regex cleaning
- stopword removal
- lemmatization
- TF-IDF vectorization with sklearn

Important learning:

```text
fit() learns statistics.
transform() applies learned statistics.
```

For TF-IDF, the correct workflow is:

```text
split first
fit TF-IDF on train only
transform validation and test
```

This avoids data leakage.

---

### Phase 3 — Dataset Creation

The dataset was split into:

```text
Train      = 80%
Validation = 10%
Test       = 10%
```

The label rule was:

```text
Score < 3  → negative
Score = 3  → neutral
Score > 3  → positive
```

Important rule:

Validation and test data must not be resampled or used to learn TF-IDF statistics.

---

### Phase 4 — Initial Spark ML Training

The first training approach was hybrid:

```text
Python/sklearn TF-IDF
→ manual conversion to Spark SparseVector
→ Spark Logistic Regression
```

This helped learning, but it was not ideal for production streaming because it mixed sklearn preprocessing with Spark ML training.

Strategic decision:

```text
Move from hybrid sklearn + Spark pipeline
to Spark-only production pipeline.
```

---

## 6. Spark-Only Production Training Pipeline

Current production training file:

```text
src/spark/training/train_spark_pipeline.py
```

The final Spark ML pipeline uses:

```text
text
→ RegexTokenizer
→ StopWordsRemover
→ CountVectorizer
→ IDF
→ NGram
→ VectorAssembler
→ StringIndexer
→ LogisticRegression
```

When `use_bigrams=True`, the model uses both:

```text
unigrams = single words
bigrams  = two-word expressions
```

Example:

```text
review = "not good product"

unigrams:
not
good
product

bigrams:
not good
good product
```

This was important because sentiment often depends on expressions such as:

```text
not good
not worth
waste money
very good
would buy
```

---

## 7. Class Imbalance Handling

The dataset is imbalanced:

```text
positive = dominant class
negative = minority class
neutral  = smallest and most ambiguous class
```

The Spark production model uses class weights.

Formula:

```text
class_weight = total_training_rows / (number_of_classes × class_count)
```

Observed class weights on the 100k sample:

```text
positive -> 0.4348
negative -> 2.1826
neutral  -> 4.1318
```

Meaning:

- positive errors are penalized less
- negative errors are penalized more
- neutral errors are penalized most

This helps the model pay more attention to minority classes.

---

## 8. Hyperparameter Tuning

Tuning was done with two strategies:

1. Manual tests
2. Simulated Annealing tuner

Main tuned parameters:

| Parameter | Meaning |
|---|---|
| `vocab_size` | Maximum vocabulary size kept by CountVectorizer |
| `min_df` | Minimum number of documents a term must appear in |
| `max_iter` | Maximum optimization iterations for Logistic Regression |
| `reg_param` | Regularization strength |
| `use_bigrams` | Whether to use unigrams + bigrams |

Tuning script:

```text
src/spark/training/tune_spark_pipeline_sa.py
```

The tuner reuses the training functions from `train_spark_pipeline.py` instead of duplicating the full training script.

---

## 9. Simulated Annealing Tuning

Simulated Annealing was used because it gives a structured way to explore hyperparameters without testing every possible combination.

Each candidate configuration contains:

```python
{
    "vocab_size": ...,
    "min_df": ...,
    "max_iter": ...,
    "reg_param": ...,
    "use_bigrams": ...
}
```

The tuner:

1. Starts from an initial configuration.
2. Changes one parameter to create a neighbor.
3. Trains a full Spark pipeline on the training set.
4. Evaluates on the validation set.
5. Optimizes `macro_f1`.
6. Saves every result to CSV and Markdown.

Why `macro_f1`?

Because the dataset is imbalanced and accuracy alone would favor the dominant positive class.

---

## 10. Important Tuning Results

### Baseline Spark-only model

The early Spark-only model worked but was biased toward the positive class.

### Class weights

Class weights improved minority-class awareness.

### Bigrams-only

Bigrams-only performed worse because the model lost useful single-word signals.

Example problem:

```text
bigrams-only keeps "not good"
but loses "bad", "great", "terrible", "excellent"
```

Result:

```text
Accuracy    = 0.7638
Macro F1    = 0.5706
Positive F1 = 0.8700
Negative F1 = 0.5388
Neutral F1  = 0.3029
```

### Unigrams + bigrams

This performed much better because the model kept both single words and phrase signals.

Intermediate result:

```text
Accuracy    = 0.8156
Macro F1    = 0.6301
Positive F1 = 0.9053
Negative F1 = 0.6207
Neutral F1  = 0.3644
```

### Final best validation configuration

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

Validation metrics:

```text
Accuracy    = 0.8159
Macro F1    = 0.6470
Positive F1 = 0.9031
Negative F1 = 0.6419
Neutral F1  = 0.3961
```

Test metrics:

```text
Accuracy    = 0.8130
Macro F1    = 0.6349
Positive F1 = 0.9018
Negative F1 = 0.6558
Neutral F1  = 0.3470
```

Conclusion:

The model generalizes reasonably well. Neutral remains the hardest class, but the final model is strong enough to move into streaming inference.

---

## 11. Saved Model

The final Spark `PipelineModel` is saved locally at:

```text
src/spark/model/sentiment_pipeline_model
```

This model should **not** be committed to GitHub because it is a generated artifact.

Recommended `.gitignore` entries:

```gitignore
src/spark/model/
src/spark/models/
*.model
```

The model can be regenerated by running the training script.

---

## 12. How to Run

### 12.1 Activate WSL environment

From Windows PowerShell:

```powershell
wsl
```

Inside WSL, from the project root:

```bash
source ../env/big_data_env/bin/activate
```

Project root example:

```bash
/mnt/c/Users/Me/Desktop/END TO END DATA ENGINEERING PROJECTS/BIG DATA PROJECT
```

---

### 12.2 Train and save the final Spark model

```bash
export PYTHONPATH=$PWD

spark-submit src/spark/training/train_spark_pipeline.py 2>/dev/null | grep -E "==========|Accuracy|Macro F1|Positive F1|Negative F1|Neutral F1|Model saved"
```

Expected important output:

```text
Model saved to: src/spark/model/sentiment_pipeline_model

VALIDATION METRICS
Accuracy:    0.8159
Macro F1:    0.6470
Positive F1: 0.9031
Negative F1: 0.6419
Neutral F1:  0.3961

TEST METRICS
Accuracy:    0.8130
Macro F1:    0.6349
Positive F1: 0.9018
Negative F1: 0.6558
Neutral F1:  0.3470
```

---

### 12.3 Run the SA tuner

```bash
export PYTHONPATH=$PWD

spark-submit src/spark/training/tune_spark_pipeline_sa.py 2>/dev/null | grep -E "INITIAL|SA STEP|Temperature|Current config|Candidate config|Macro F1|Accepted|BEST|Accuracy|Positive F1|Negative F1|Neutral F1|Saved"
```

The tuner writes results to:

```text
results/spark_sa_tuning_results.csv
results/spark_sa_tuning_results.md
```

---

### 12.4 Run Spark streaming prediction later

This is the next phase and is not fully finalized yet.

Expected command:

```bash
export PYTHONPATH=$PWD

spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.4 \
  src/spark/streaming/predict_stream.py
```

---

## 13. Git Rules

Do commit:

```text
src/
docs/
results/*.md
README.md
requirements.txt
.gitignore
kafka/docker-compose.yml
```

Do not commit:

```text
data/raw/Reviews.csv
src/spark/model/
__pycache__/
*.pyc
large generated files
```

Recommended `.gitignore` additions:

```gitignore
# Python
__pycache__/
*.pyc
.venv/
env/

# Data
data/raw/Reviews.csv
data/raw/*.csv

# Spark models
src/spark/model/
src/spark/models/
*.model

# Spark local files
spark-warehouse/
metastore_db/
derby.log

# Results - keep markdown summaries if useful, ignore large CSVs if needed
# results/*.csv
```

---

## 14. Current Commit Title Recommendation

```text
Optimize Spark sentiment pipeline and save final trained model
```

Alternative:

```text
Add Spark ML tuning, bigram features, and saved sentiment model
```

---

## 15. Next Phase

Next phase:

```text
Phase 8 — Spark Structured Streaming prediction
```

Goal:

```text
Kafka topic amazon_reviews
→ Spark reads stream
→ Spark loads saved PipelineModel
→ Spark predicts sentiment
→ predictions printed to console
```

Important rule:

```text
Streaming must not train the model.
It only loads and applies the saved model.
```

MongoDB starts after this in Phase 9.

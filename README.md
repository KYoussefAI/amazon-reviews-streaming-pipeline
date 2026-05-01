# Amazon Reviews Real-Time Sentiment Analysis Pipeline

A production-oriented Big Data project for real-time sentiment analysis on Amazon review events using Kafka, Apache Spark Structured Streaming, Spark ML, and MongoDB.

The project is built phase by phase to simulate a real data engineering workflow: ingestion, distributed processing, machine learning inference, storage, visualization, and orchestration.

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Current Project Status](#2-current-project-status)
3. [Architecture](#3-architecture)
4. [Dataset](#4-dataset)
5. [Repository Structure](#5-repository-structure)
6. [Completed Phases](#6-completed-phases)
7. [Spark ML Training Pipeline](#7-spark-ml-training-pipeline)
8. [Model Tuning Summary](#8-model-tuning-summary)
9. [Final Model Metrics](#9-final-model-metrics)
10. [Phase 8: Streaming Inference](#10-phase-8-streaming-inference)
11. [How to Run the Project](#11-how-to-run-the-project)
12. [Git and Artifact Rules](#12-git-and-artifact-rules)
13. [Next Phase: MongoDB Storage](#13-next-phase-mongodb-storage)
14. [Future Work](#14-future-work)

---

## 1. Project Objective

The objective is to build a complete real-time Big Data and machine learning pipeline:

```text
Amazon Reviews Dataset
→ Kafka
→ Spark Structured Streaming
→ Spark ML Sentiment Prediction
→ MongoDB
→ Dashboard
```

The fixed target architecture is:

```text
Producer → Kafka → Spark → MongoDB → Dashboard
```

The current implementation has completed the Spark streaming inference step and is ready to move into MongoDB storage.

---

## 2. Current Project Status

Completed:

- Kafka producer that streams Amazon review events.
- Kafka topic used for review ingestion.
- Spark-only ML training pipeline.
- Train/validation/test split.
- Class-weighted Logistic Regression.
- TF-IDF feature extraction using Spark ML.
- Unigram + bigram feature engineering.
- Simulated Annealing hyperparameter tuning.
- Final Spark `PipelineModel` training and local saving.
- Spark Structured Streaming inference from Kafka.
- Real-time sentiment predictions printed to the console.

Current completed architecture:

```text
Producer → Kafka → Spark Structured Streaming → Saved Spark ML Model → Console Predictions
```

Next phase:

```text
Phase 9 — Store prediction results in MongoDB
```

---

## 3. Architecture

### Current Working Architecture

```text
Reviews.csv
   ↓
producer.py
   ↓
Kafka topic: amazon_reviews
   ↓
predict_stream.py
   ↓
Saved Spark PipelineModel
   ↓
Real-time sentiment prediction
   ↓
Console output
```

### Target Architecture After Phase 9

```text
Reviews.csv
   ↓
Kafka Producer
   ↓
Kafka
   ↓
Spark Structured Streaming
   ↓
Spark ML Prediction
   ↓
MongoDB
   ↓
Dashboard
```

---

## 4. Dataset

The project uses the **Amazon Fine Food Reviews** dataset.

Expected local path:

```text
data/raw/Reviews.csv
```

Main columns used:

| Column | Description |
|---|---|
| `Text` | Review text |
| `Score` | Review rating from 1 to 5 |

Sentiment labels are derived from `Score`:

| Score condition | Label |
|---|---|
| `Score < 3` | `negative` |
| `Score == 3` | `neutral` |
| `Score > 3` | `positive` |

The raw dataset is excluded from Git because it is large and should remain local.

---

## 5. Repository Structure

Current production-oriented structure:

```text
BIG DATA PROJECT/
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
│   ├── spark_sa_tuning_results.md
│   └── spark_sa_tuning_results.csv
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
│       │   └── sentiment_pipeline_model/       # local artifact, ignored by Git
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

### Folder Roles

| Folder | Purpose |
|---|---|
| `src/ingestion/` | Kafka producer and ingestion logic |
| `src/experiments/` | Early Python/sklearn experimentation code |
| `src/spark/training/` | Production Spark ML training and tuning |
| `src/spark/streaming/` | Spark streaming prediction logic |
| `src/spark/model/` | Local saved Spark model artifact |
| `results/` | Tuning results |
| `docs/` | Technical reports and project documentation |

---

## 6. Completed Phases

### Phase 1 — Kafka Ingestion

Implemented:

- Kafka and Zookeeper using Docker.
- Topic: `amazon_reviews`.
- Producer that streams review events from `Reviews.csv`.
- Basic consumer for inspection/debugging.

Kafka message format:

```json
{
  "text": "review text here",
  "score": 5
}
```

### Phase 2 — Early Text Preprocessing

Implemented in the experimentation layer:

- Lowercasing.
- Regex cleaning.
- Stopword removal.
- Lemmatization.
- sklearn TF-IDF.

This phase helped validate the ML idea, but it was later replaced for production by a Spark-only pipeline.

### Phase 3 — Dataset Creation

Implemented:

- CSV loading.
- Score-to-label conversion.
- Train/validation/test split.
- Stratification in the early experimentation pipeline.

### Phase 4 to 7 — Training, Validation, Testing, Model Selection

Implemented:

- Logistic Regression baseline.
- Confusion matrix.
- Precision, recall, F1-score per class.
- Handling of class imbalance.
- Transition from hybrid sklearn/Spark pipeline to Spark-only production training.

### Phase 8 — Spark Streaming Inference

Implemented and validated:

```text
Producer → Kafka → Spark Structured Streaming → Saved Spark ML Model → Console Predictions
```

---

## 7. Spark ML Training Pipeline

The production model is implemented fully with Spark ML components.

Final training pipeline:

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

This design was selected because:

- Unigrams preserve individual sentiment words.
- Bigrams capture phrase-level sentiment such as `not good`, `not worth`, and `waste money`.
- Bigrams alone performed worse.
- Unigrams + bigrams significantly improved validation metrics.

---

## 8. Model Tuning Summary

Tuning methods used:

- Manual parameter tuning.
- Class weighting for imbalance.
- Simulated Annealing hyperparameter search.
- Feature engineering comparison:
  - unigrams only
  - bigrams only
  - unigrams + bigrams

Main tuned parameters:

| Parameter | Meaning |
|---|---|
| `vocab_size` | Maximum vocabulary size for CountVectorizer |
| `min_df` | Minimum document frequency for keeping a term |
| `max_iter` | Maximum Logistic Regression optimization iterations |
| `reg_param` | Regularization strength |
| `use_bigrams` | Whether bigram features are included |

Primary metric:

```text
Macro F1
```

Reason:

```text
The dataset is imbalanced, so accuracy alone can hide weak minority-class performance.
```

---

## 9. Final Model Metrics

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

### Validation Metrics

```text
Accuracy    = 0.8159
Macro F1    = 0.6470
Positive F1 = 0.9031
Negative F1 = 0.6419
Neutral F1  = 0.3961
```

### Test Metrics

```text
Accuracy    = 0.8130
Macro F1    = 0.6349
Positive F1 = 0.9018
Negative F1 = 0.6558
Neutral F1  = 0.3470
```

Interpretation:

- Positive class is stable.
- Negative class generalizes well.
- Neutral remains the weakest class.
- Overall validation and test results are close enough for a production baseline.

---

## 10. Phase 8: Streaming Inference

Streaming inference is implemented in:

```text
src/spark/streaming/predict_stream.py
```

The script:

1. Creates a Spark session.
2. Loads the saved Spark `PipelineModel`.
3. Reads Kafka messages from topic `amazon_reviews`.
4. Parses JSON messages.
5. Applies the saved model using `model.transform()`.
6. Converts numeric predictions to readable labels.
7. Prints predictions to console.

Output columns:

| Column | Meaning |
|---|---|
| `text_preview` | Short preview of review text |
| `score` | Original score from dataset, used only for checking |
| `prediction` | Numeric Spark prediction |
| `predicted_label` | Human-readable sentiment label |
| `probability` | Probability vector |

Important:

```text
The streaming script does not train.
It only loads the saved model and predicts.
```

---

## 11. How to Run the Project

### 11.1 Enter Project Directory in WSL

```bash
cd "/mnt/c/Users/Me/Desktop/END TO END DATA ENGINEERING PROJECTS/BIG DATA PROJECT"
source ../env/big_data_env/bin/activate
export PYTHONPATH=$PWD
```

### 11.2 Start Kafka

In terminal 1:

```bash
cd kafka
docker compose up
```

Keep this terminal running.

### 11.3 Run Spark Streaming Prediction

In terminal 2, from project root:

```bash
source ../env/big_data_env/bin/activate
export PYTHONPATH=$PWD

spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.4 \
  src/spark/streaming/predict_stream.py
```

Keep this terminal running.

### 11.4 Run Producer

In terminal 3, from project root:

```bash
source ../env/big_data_env/bin/activate
export PYTHONPATH=$PWD

python src/ingestion/producer.py
```

Expected result:

```text
Spark console output shows streaming batches with predicted sentiment labels.
```

---

## 12. Git and Artifact Rules

Do not commit:

```text
data/raw/Reviews.csv
src/spark/model/
src/spark/models/
*.model
__pycache__/
*.pyc
spark-warehouse/
metastore_db/
derby.log
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

Commit source code, documentation, and lightweight results summaries.

Do not commit the saved model artifact. It can be regenerated by running:

```bash
spark-submit src/spark/training/train_spark_pipeline.py
```

---

## 13. Next Phase: MongoDB Storage

Phase 9 objective:

```text
Spark streaming predictions → MongoDB
```

Target architecture:

```text
Producer → Kafka → Spark Structured Streaming → Sentiment Prediction → MongoDB
```

Phase 9 should be implemented incrementally:

1. Add MongoDB container.
2. Test MongoDB connection.
3. Write one simple document.
4. Add a MongoDB writer for prediction rows.
5. Integrate MongoDB write into Spark Structured Streaming.
6. Keep console output temporarily for debugging.
7. Validate stored predictions in MongoDB.

Phase 9 must remain efficient and should not redesign completed phases.

---

## 14. Future Work

Planned future improvements:

- Store predictions in MongoDB.
- Build dashboard on MongoDB data.
- Add monitoring and logging.
- Add Airflow orchestration.
- Train on the full dataset if needed.
- Test additional ML models:
  - Naive Bayes
  - Linear SVC
  - One-vs-Rest
  - Voting ensemble
  - Stacking ensemble

Current production baseline:

```text
Spark ML PipelineModel
TF-IDF
Unigrams + bigrams
Class-weighted Logistic Regression
Simulated Annealing tuned hyperparameters
```

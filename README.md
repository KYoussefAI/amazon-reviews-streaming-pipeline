# Amazon Reviews Real-Time Sentiment Analysis Pipeline

A production-oriented Big Data project for real-time sentiment analysis on Amazon review events using Kafka, Apache Spark Structured Streaming, Spark ML, MongoDB, and a future dashboard layer.

The project is built phase by phase to simulate a real data engineering workflow: ingestion, distributed processing, machine learning inference, persistent storage, visualization, and orchestration.

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
11. [Phase 9: MongoDB Storage](#11-phase-9-mongodb-storage)
12. [How to Run the Project](#12-how-to-run-the-project)
13. [MongoDB Validation Commands](#13-mongodb-validation-commands)
14. [Git and Artifact Rules](#14-git-and-artifact-rules)
15. [Next Step: Full-Data Training](#15-next-step-full-data-training)
16. [Future Work](#16-future-work)

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

This architecture must not be changed. Each phase adds one engineering layer while preserving the previous validated layers.

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
- MongoDB container added to Docker Compose.
- Python MongoDB connection tested with `pymongo`.
- Spark streaming predictions written to MongoDB using `foreachBatch`.
- MongoDB prediction documents validated with `mongosh`.

Current completed architecture:

```text
Producer → Kafka → Spark Structured Streaming → Saved Spark ML Model → MongoDB
```

Current phase status:

```text
Phase 9 — MongoDB Storage: completed and validated
```

Next engineering step:

```text
Full-data model training using all available Reviews.csv rows
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
MongoDB: amazon_reviews_db.sentiment_predictions
```

### Target Architecture After Dashboard Phase

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

### Engineering Role of Each Component

| Component | Role |
|---|---|
| `producer.py` | Simulates real-time review events from `Reviews.csv` |
| Kafka | Decouples ingestion from processing and stores the event stream temporarily |
| Spark Structured Streaming | Reads Kafka events and processes them as micro-batches |
| Spark ML `PipelineModel` | Applies the saved sentiment model without retraining |
| MongoDB | Stores prediction results permanently as documents |
| Dashboard | Future layer that reads MongoDB data for visualization |

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

Important:

```text
The raw dataset is excluded from Git because it is large and must remain local.
```

Recommended dataset ignore rules:

```gitignore
data/raw/Reviews.csv
data/raw/*.csv
```

---

## 5. Repository Structure

Current production-oriented structure:

```text
BIG DATA PROJECT/
│
├── data/
│   └── raw/
│       └── Reviews.csv                         # local only, ignored by Git
│
├── docs/
│   ├── spark_sentiment_tuning_report.md
│   └── project_progress_for_teammates.md
│
├── kafka/
│   └── docker-compose.yml                      # Kafka, Zookeeper, MongoDB
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
│   ├── spark/
│   │   ├── model/
│   │   │   └── sentiment_pipeline_model/       # local artifact, ignored by Git
│   │   │
│   │   ├── training/
│   │   │   ├── train_spark_pipeline.py
│   │   │   └── tune_spark_pipeline_sa.py
│   │   │
│   │   └── streaming/
│   │       ├── consumer.py
│   │       └── predict_stream.py
│   │
│   └── storage/
│       ├── mongodb_writer.py
│       └── test_mongodb_connection.py
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
| `src/storage/` | MongoDB storage and diagnostic scripts |
| `src/spark/model/` | Local saved Spark model artifact |
| `results/` | Tuning results |
| `docs/` | Technical reports and project documentation |
| `kafka/` | Docker Compose services for Kafka, Zookeeper, and MongoDB |

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

### Phase 9 — MongoDB Storage

Implemented and validated:

```text
Producer → Kafka → Spark Structured Streaming → Saved Spark ML Model → MongoDB
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

Primary tuning metric:

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

Important:

```text
These metrics are from the tuned 100k-sample training run.
Full-data training is the next planned model refresh step.
```

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
5. Adds schema-compatible dummy columns for the saved training pipeline.
6. Applies the saved model using `model.transform()`.
7. Converts numeric predictions to readable labels.
8. Prepares prediction rows for MongoDB storage.

Output columns:

| Column | Meaning |
|---|---|
| `text_preview` | Short preview of review text |
| `text` | Full review text |
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

## 11. Phase 9: MongoDB Storage

Phase 9 adds a persistent storage layer after Spark streaming prediction.

Updated architecture:

```text
Producer → Kafka → Spark Structured Streaming → Spark ML Prediction → MongoDB
```

MongoDB is started as a Docker container using the existing Docker Compose setup.

MongoDB target:

```text
URI: mongodb://localhost:27017
Database: amazon_reviews_db
Collection: sentiment_predictions
```

### Main Phase 9 Files

| File | Role |
|---|---|
| `kafka/docker-compose.yml` | Runs MongoDB container with Kafka/Zookeeper stack |
| `src/storage/test_mongodb_connection.py` | Diagnostic script to test Python → MongoDB connection |
| `src/storage/mongodb_writer.py` | Reusable MongoDB writer used by Spark streaming |
| `src/spark/streaming/predict_stream.py` | Calls MongoDB writer through `foreachBatch` |

### MongoDB Document Schema

Each Spark prediction is stored as a MongoDB document:

```json
{
  "text_preview": "short review preview",
  "text": "full review text",
  "score": 5,
  "prediction": 0.0,
  "predicted_label": "positive",
  "probability": [0.98, 0.01, 0.01],
  "batch_id": 12,
  "processed_at": "2026-05-01T16:33:49Z",
  "source": "spark_structured_streaming"
}
```

### Why `foreachBatch` Is Used

Spark Structured Streaming processes data in micro-batches.

Instead of inserting each row one by one, the project writes each micro-batch to MongoDB:

```text
Spark micro-batch DataFrame
→ foreachBatch(write_predictions_to_mongodb)
→ convert rows to Python dictionaries
→ insert_many()
→ MongoDB collection
```

This keeps the storage logic separate from the streaming prediction logic and makes the system easier to debug.

### Phase 9 Validation Result

MongoDB validation after testing showed:

```text
Total documents in sentiment_predictions: 801
Spark streaming documents:               799
Manual/Python test documents:             2
```

Prediction distribution from Spark streaming test:

```text
positive = 610
negative = 128
neutral  = 61
```

This validates that the end-to-end storage pipeline works.

---

## 12. How to Run the Project

### 12.1 Enter Project Directory in WSL

```bash
cd "/mnt/c/Users/Me/Desktop/END TO END DATA ENGINEERING PROJECTS/BIG DATA PROJECT"
source env/big_data_env/bin/activate
export PYTHONPATH=$PWD
```

If the environment path is different on your machine, activate the correct virtual environment before running Python or Spark commands.

### 12.2 Start Kafka, Zookeeper, and MongoDB

In terminal 1:

```bash
cd kafka
docker compose up -d
docker compose ps
```

Expected services:

```text
zookeeper
kafka
mongodb
```

### 12.3 Test MongoDB Connection from Python

From project root:

```bash
python src/storage/test_mongodb_connection.py
```

Expected output:

```text
========== MONGODB PYTHON TEST ==========
Inserted document id: ...
```

### 12.4 Run Spark Streaming Prediction to MongoDB

In terminal 2, from project root:

```bash
source env/big_data_env/bin/activate
export PYTHONPATH=$PWD

spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.4 \
  src/spark/streaming/predict_stream.py
```

Expected Spark output:

```text
========== STREAMING PREDICTIONS TO MONGODB STARTED ==========
Batch X: inserted Y documents into MongoDB.
```

### 12.5 Run Producer

In terminal 3, from project root:

```bash
source env/big_data_env/bin/activate
export PYTHONPATH=$PWD

python src/ingestion/producer.py
```

Expected result:

```text
Producer sends review events to Kafka.
Spark consumes them, predicts sentiment, and writes results to MongoDB.
```

---

## 13. MongoDB Validation Commands

Open MongoDB shell:

```bash
docker exec -it mongodb mongosh
```

Use the project database:

```javascript
use amazon_reviews_db
```

Count all documents:

```javascript
db.sentiment_predictions.countDocuments()
```

Count only Spark streaming documents:

```javascript
db.sentiment_predictions.countDocuments({
  source: "spark_structured_streaming"
})
```

Show prediction distribution:

```javascript
db.sentiment_predictions.aggregate([
  { $match: { source: "spark_structured_streaming" } },
  { $group: { _id: "$predicted_label", count: { $sum: 1 } } },
  { $sort: { count: -1 } }
])
```

Show latest Spark predictions:

```javascript
db.sentiment_predictions.find({
  source: "spark_structured_streaming"
}, {
  text_preview: 1,
  score: 1,
  predicted_label: 1,
  probability: 1,
  processed_at: 1
}).sort({
  processed_at: -1
}).limit(10).pretty()
```

---

## 14. Git and Artifact Rules

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
exports/mongodb/
backups/mongodb/
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

# MongoDB exports/backups
exports/mongodb/
backups/mongodb/

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

Commit:

```text
source code
docker-compose.yml
README.md
docs/*.md
lightweight results summaries
```

Do not commit the saved Spark model artifact. It can be regenerated by running:

```bash
spark-submit src/spark/training/train_spark_pipeline.py
```

---

## 15. Next Step: Full-Data Training

The current final model was tuned on a 100k sample.

Current training script setting:

```python
SAMPLE_SIZE = 100000
```

To train on all available rows from `data/raw/Reviews.csv`, update:

```python
SAMPLE_SIZE = None
```

in:

```text
src/spark/training/train_spark_pipeline.py
```

Then run:

```bash
spark-submit src/spark/training/train_spark_pipeline.py
```

Expected effect:

```text
Spark trains the final PipelineModel using the full local dataset.
The saved model is written to src/spark/model/sentiment_pipeline_model.
predict_stream.py automatically loads the refreshed model from the same path.
```

Recommended order:

```text
1. Commit Phase 9 MongoDB storage.
2. Change SAMPLE_SIZE to None.
3. Run full-data training.
4. Save the refreshed model.
5. Re-test streaming prediction to MongoDB.
6. Start dashboard phase.
```

Engineering reason:

```text
Storage integration and full-data model training are separate risks.
Phase 9 should remain stable and committed before scaling training.
```

---

## 16. Future Work

Planned future improvements:

- Train the final model on the full dataset.
- Build a dashboard on MongoDB data.
- Add monitoring and logging.
- Add Airflow orchestration.
- Add export/import commands for MongoDB snapshots.
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
Kafka → Spark Structured Streaming → MongoDB storage
```

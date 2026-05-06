# Amazon Reviews Real-Time Sentiment Command Center

A production-oriented Big Data project that streams Amazon review events through Kafka, applies Spark ML sentiment prediction with Spark Structured Streaming, stores enriched prediction results in MongoDB, visualizes analytics in a Streamlit dashboard, and includes an optional Airflow orchestration layer for batch preparation tasks.

This project was built phase by phase as a practical data engineering training lab: ingestion, streaming, distributed processing, machine learning inference, storage, dashboarding, orchestration, validation, and documentation.

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Current Status](#2-current-status)
3. [Requirement Coverage](#3-requirement-coverage)
4. [Architecture](#4-architecture)
5. [Tech Stack](#5-tech-stack)
6. [Dataset](#6-dataset)
7. [Repository Structure](#7-repository-structure)
8. [Spark ML Training Pipeline](#8-spark-ml-training-pipeline)
9. [Final Model Metrics](#9-final-model-metrics)
10. [Streaming Test Export](#10-streaming-test-export)
11. [Kafka Producer](#11-kafka-producer)
12. [Spark Structured Streaming Inference](#12-spark-structured-streaming-inference)
13. [MongoDB Storage](#13-mongodb-storage)
14. [Streamlit Dashboard](#14-streamlit-dashboard)
15. [Airflow Orchestration](#15-airflow-orchestration)
16. [Screenshots](#16-screenshots)
17. [How to Run](#17-how-to-run)
18. [Validation Commands](#18-validation-commands)
19. [Git and Artifact Rules](#19-git-and-artifact-rules)
20. [Future Improvements](#20-future-improvements)

---

## 1. Project Objective

The objective is to build a complete Big Data sentiment analysis pipeline for Amazon product reviews:

```text
Amazon Reviews Dataset
→ Test Split Export
→ Kafka Producer
→ Kafka Topic
→ Spark Structured Streaming
→ Spark ML Prediction
→ MongoDB
→ Streamlit Dashboard
→ PDF Report
```

The project simulates a real data engineering workflow where review events are ingested continuously, processed by a distributed streaming engine, enriched with machine learning predictions, stored in a NoSQL database, and monitored through dashboards and reports.

---

## 2. Current Status

Current completed runtime architecture:

```text
Producer → Kafka → Spark Structured Streaming → Spark ML Prediction → MongoDB → Streamlit Dashboard
```

Optional orchestration layer:

```text
Airflow → validate project → validate dataset → export test split → train model → validate saved model
```

Completed:

- Kafka and Zookeeper running with Docker Compose.
- MongoDB running as a Docker service.
- Spark-only ML training pipeline.
- TF-IDF feature engineering using unigrams and bigrams.
- Class-weighted Logistic Regression for imbalanced sentiment classes.
- Simulated Annealing hyperparameter tuning.
- Final Spark `PipelineModel` saved locally.
- Test split exported for streaming simulation.
- ProductId, UserId, review date, summary, text, true label, and source split preserved end-to-end.
- Kafka producer streaming enriched Amazon review events.
- Spark Structured Streaming inference from Kafka.
- Prediction documents written to MongoDB.
- Streamlit dashboard connected to MongoDB.
- Dashboard KPIs, filters, confidence analytics, score analytics, batch analytics, ProductId analytics, prediction-by-date analytics, latest events table, and PDF report export.
- Airflow DAG added and validated for batch orchestration.
- GitHub-ready README and screenshots.

---

## 3. Requirement Coverage

| Project Requirement | Status | Implementation |
|---|---:|---|
| Real-time review exploration with Kafka | Completed | `src/ingestion/producer.py`, Kafka topic `amazon_reviews` |
| Data preparation: vectorization and TF-IDF | Completed | Spark ML pipeline with `RegexTokenizer`, `StopWordsRemover`, `CountVectorizer`, `IDF` |
| Dataset partitioning and label creation | Completed | `train_spark_pipeline.py`, `export_test_split_for_streaming.py` |
| Train model on 80% of Reviews.csv | Completed | Spark training script |
| Validate and tune on 10% | Completed | Validation metrics + Simulated Annealing tuner |
| Test on 10% | Completed | Final test metrics + streaming export |
| Choose best model | Completed | Spark TF-IDF + Logistic Regression selected |
| Save best model | Completed | `src/spark/model/sentiment_pipeline_model` |
| Online prediction using test data | Completed | Kafka producer + Spark Structured Streaming |
| Offline dashboard from MongoDB predictions | Completed | Streamlit dashboard |
| Prediction results by date | Completed | Dashboard + PDF report |
| ProductId `B001E4KFG0` scoring | Completed | Dedicated ProductId dashboard section |
| MongoDB prediction archive | Completed | `amazon_reviews_db.sentiment_predictions` |
| Docker services | Completed | Kafka, Zookeeper, MongoDB |
| GitHub upload | Completed / ready | README, screenshots, scripts |
| Airflow orchestration | Added as portfolio improvement | Batch orchestration only |

---

## 4. Architecture

### 4.1 High-Level Runtime Architecture

```text
                ┌──────────────────────────┐
                │  Amazon Reviews CSV      │
                │  data/raw/Reviews.csv    │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ Test Split Export        │
                │ export_test_split...py   │
                └────────────┬─────────────┘
                             │ JSONL
                             ▼
                ┌──────────────────────────┐
                │ Kafka Producer           │
                │ src/ingestion/producer.py│
                └────────────┬─────────────┘
                             │ JSON events
                             ▼
                ┌──────────────────────────┐
                │ Kafka Topic              │
                │ amazon_reviews           │
                └────────────┬─────────────┘
                             │ stream read
                             ▼
                ┌──────────────────────────┐
                │ Spark Structured         │
                │ Streaming                │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ Spark ML PipelineModel   │
                │ TF-IDF + LogisticReg     │
                └────────────┬─────────────┘
                             │ predictions
                             ▼
                ┌──────────────────────────┐
                │ MongoDB                  │
                │ sentiment_predictions    │
                └────────────┬─────────────┘
                             │ query
                             ▼
                ┌──────────────────────────┐
                │ Streamlit Dashboard      │
                │ Command Center + Report  │
                └──────────────────────────┘
```

### 4.2 Airflow Batch Orchestration Architecture

```text
Airflow DAG: amazon_reviews_batch_orchestration

start
→ check_project_structure
→ check_raw_dataset_exists
→ export_test_split_for_streaming
→ validate_streaming_export
→ train_spark_model
→ validate_saved_model
→ print_next_runtime_commands
→ end
```

Airflow is used for batch preparation and validation. It does **not** replace the live Kafka/Spark streaming services.

---

## 5. Tech Stack

| Layer | Tool |
|---|---|
| Programming | Python |
| Message Broker | Apache Kafka |
| Kafka Coordination | Zookeeper |
| Containerized Services | Docker Compose |
| Distributed Processing | Apache Spark / PySpark |
| Streaming Processing | Spark Structured Streaming |
| Machine Learning | Spark MLlib |
| Feature Engineering | RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, NGram, VectorAssembler |
| Model | Class-weighted Logistic Regression |
| Storage | MongoDB |
| Dashboard | Streamlit + Plotly |
| Reporting | ReportLab PDF export |
| Orchestration | Apache Airflow |
| Environment | WSL/Linux recommended |
| Version Control | Git + GitHub |

---

## 6. Dataset

Dataset:

```text
Amazon Fine Food Reviews
```

Expected local path:

```text
data/raw/Reviews.csv
```

Important raw columns:

| Column | Role |
|---|---|
| `Id` | Review identifier |
| `ProductId` | Product identifier |
| `UserId` | User identifier |
| `Score` | Amazon rating from 1 to 5 |
| `Time` | Unix timestamp |
| `Summary` | Short review summary |
| `Text` | Full review text |

Sentiment target rule:

| Score condition | Target label |
|---|---|
| `Score < 3` | `negative` |
| `Score == 3` | `neutral` |
| `Score > 3` | `positive` |

The raw CSV is not committed to Git because it is large.

---

## 7. Repository Structure

Current project structure:

```text
BIG DATA PROJECT/
│
├── airflow/
│   ├── README_AIRFLOW.md
│   ├── dags/
│   │   └── amazon_reviews_batch_pipeline.py
│   ├── requirements-airflow.txt
│   └── scripts/
│       ├── setup_airflow.sh
│       ├── start_airflow.sh
│       ├── stop_airflow.sh
│       └── test_dag.sh
│
├── data/
│   ├── raw/
│   │   └── Reviews.csv
│   └── processed/
│       └── test_reviews.jsonl
│
├── docs/
│   ├── PHASE5.md
│   ├── spark_sentiment_tuning_report.md
│   └── screenshots/
│
├── kafka/
│   ├── docker-compose.yml
│   └── topics.md
│
├── results/
│   ├── spark_sa_tuning_results.csv
│   └── spark_sa_tuning_results.md
│
├── src/
│   ├── __init__.py
│   ├── dashboard/
│   │   └── app.py
│   ├── experiments/
│   │   ├── preprocessing/
│   │   └── training/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── producer.py
│   ├── spark/
│   │   ├── model/
│   │   │   └── sentiment_pipeline_model/
│   │   ├── streaming/
│   │   │   ├── __init__.py
│   │   │   ├── consumer.py
│   │   │   └── predict_stream.py
│   │   └── training/
│   │       ├── __init__.py
│   │       ├── export_test_split_for_streaming.py
│   │       ├── train_spark_pipeline.py
│   │       └── tune_spark_pipeline_sa.py
│   └── storage/
│       ├── __init__.py
│       ├── mongodb_writer.py
│       └── test_mongodb_connection.py
│
├── project_context.md
├── README.md
├── requirements.txt
├── structure.txt
└── useful_commands.txt
```

Local-only generated artifacts:

```text
data/raw/
data/processed/
src/spark/model/
src/spark/models/
spark-warehouse/
metastore_db/
airflow/airflow_home/
airflow/logs/
exports/
backups/
.venv/
__pycache__/
```

---

## 8. Spark ML Training Pipeline

Production training file:

```text
src/spark/training/train_spark_pipeline.py
```

Spark ML pipeline:

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
→ Class-weighted Logistic Regression
```

Final selected hyperparameters:

```python
vocab_size = 10000
min_df = 4
max_iter = 15
reg_param = 0.000025
use_bigrams = True
```

Saved model path:

```text
src/spark/model/sentiment_pipeline_model
```

The saved model folder is generated locally and should not be committed.

---

## 9. Final Model Metrics

The final full-data training run used valid Amazon review rows after cleaning and filtering.

### 9.1 Full Dataset Class Distribution

| Label | Count |
|---|---:|
| positive | 442,319 |
| negative | 83,623 |
| neutral | 42,502 |

Total rows:

```text
568,444
```

### 9.2 Split Sizes

| Split | Rows |
|---|---:|
| Train | 455,010 |
| Validation | 56,630 |
| Test | 56,804 |

### 9.3 Class Weights

| Label | Weight |
|---|---:|
| positive | 0.4287 |
| negative | 2.2639 |
| neutral | 4.4364 |

### 9.4 Validation Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8116 |
| Macro F1 | 0.6720 |
| Positive F1 | 0.8952 |
| Negative F1 | 0.7003 |
| Neutral F1 | 0.4206 |

### 9.5 Test Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8133 |
| Macro F1 | 0.6733 |
| Positive F1 | 0.8982 |
| Negative F1 | 0.6953 |
| Neutral F1 | 0.4266 |

Macro F1 is tracked because the dataset is imbalanced. Accuracy alone would over-reward the dominant positive class.

---

## 10. Streaming Test Export

Export file:

```text
src/spark/training/export_test_split_for_streaming.py
```

Output:

```text
data/processed/test_reviews.jsonl
```

The export preserves the fields required for downstream Kafka, MongoDB, and dashboard analysis:

```json
{
  "product_id": "B001E4KFG0",
  "user_id": "A3SGXH7AUHU8GW",
  "review_time": "2011-04-27 01:00:00",
  "review_date": "2011-04-27",
  "score": 5,
  "summary": "Good Quality Dog Food",
  "text": "I have bought several of the Vitality canned dog food products...",
  "label": "positive",
  "source_split": "product_demo",
  "source_row_index": 0
}
```

The required ProductId `B001E4KFG0` is preserved for the dashboard requirement. If it does not naturally appear in the test split, the export adds it as a controlled `product_demo` row so the ProductId analysis remains available.

---

## 11. Kafka Producer

Producer file:

```text
src/ingestion/producer.py
```

Input:

```text
data/processed/test_reviews.jsonl
```

Kafka topic:

```text
amazon_reviews
```

The producer streams enriched JSON messages containing:

```text
product_id
user_id
review_time
review_date
score
summary
text
label
source_split
source_row_index
```

This ensures metadata is not lost before Spark Structured Streaming.

---

## 12. Spark Structured Streaming Inference

Streaming file:

```text
src/spark/streaming/predict_stream.py
```

Responsibilities:

1. Read JSON messages from Kafka.
2. Parse enriched message schema.
3. Load the saved Spark `PipelineModel`.
4. Apply sentiment prediction to each review.
5. Preserve original metadata.
6. Write enriched predictions to MongoDB using `foreachBatch`.

Streaming input:

```text
Kafka topic: amazon_reviews
```

Streaming output:

```text
MongoDB collection: amazon_reviews_db.sentiment_predictions
```

Important: the streaming job does not train a model. It only loads the saved Spark model and applies inference.

---

## 13. MongoDB Storage

Mongo writer file:

```text
src/storage/mongodb_writer.py
```

Database:

```text
amazon_reviews_db
```

Collection:

```text
sentiment_predictions
```

Stored document schema:

```json
{
  "product_id": "B001E4KFG0",
  "user_id": "A3SGXH7AUHU8GW",
  "review_time": "2011-04-27 01:00:00",
  "review_date": "2011-04-27",
  "score": 5,
  "summary": "Good Quality Dog Food",
  "text_preview": "I have bought several of the Vitality canned dog food products...",
  "text": "Full review text",
  "true_label": "positive",
  "source_split": "product_demo",
  "source_row_index": 0,
  "prediction": 0.0,
  "predicted_label": "positive",
  "probability": [0.8496, 0.0027, 0.1476],
  "batch_id": 113,
  "processed_at": "2026-05-06T16:21:26Z",
  "source": "spark_structured_streaming"
}
```

---

## 14. Streamlit Dashboard

Dashboard file:

```text
src/dashboard/app.py
```

Dashboard title:

```text
Amazon Reviews Sentiment Command Center
```

Implemented features:

- MongoDB connection status.
- Total predictions.
- Latest batch ID.
- Average confidence.
- Positive, negative, and neutral prediction counts.
- Low-confidence prediction count.
- Predicted sentiment filter.
- Score filter.
- Batch ID filter.
- Professional ProductId filter:
  - type ProductId manually,
  - or select ProductId from available values.
- Sentiment distribution.
- Confidence distribution.
- Records per micro-batch.
- Batch summary table.
- Amazon score distribution.
- Score by predicted sentiment.
- Prediction results by review date.
- Streaming source split distribution.
- Dedicated ProductId analysis for `B001E4KFG0`.
- Context-aware ProductId metrics:
  - if one review exists, show exact event values,
  - if multiple reviews exist, show aggregate product metrics.
- Model confidence by class.
- Risk monitoring:
  - predictions below 60% confidence,
  - suspicious score/sentiment combinations.
- Latest streaming events table.
- PDF report export.

---

## 15. Airflow Orchestration

Airflow is included as an optional orchestration layer for the batch side of the project.

Airflow folder:

```text
airflow/
```

Main DAG:

```text
airflow/dags/amazon_reviews_batch_pipeline.py
```

DAG name:

```text
amazon_reviews_batch_orchestration
```

### 15.1 What Airflow Orchestrates

```text
start_batch_orchestration
→ check_project_structure
→ check_raw_dataset_exists
→ export_test_split_for_streaming
→ validate_streaming_export
→ train_spark_model
→ validate_saved_model
→ print_next_runtime_commands
→ end_batch_orchestration
```

### 15.2 What Airflow Does Not Orchestrate

Airflow does not run the infinite streaming services directly:

```text
bd-kafka
bd-spark
bd-producer
bd-streamlit
```

Those commands remain runtime services.

This separation is intentional:

| Layer | Tool |
|---|---|
| Batch preparation | Airflow |
| Message streaming | Kafka |
| Real-time inference | Spark Structured Streaming |
| Storage | MongoDB |
| Dashboard | Streamlit |

### 15.3 Airflow Setup

From the project root:

```bash
chmod +x airflow/scripts/*.sh
./airflow/scripts/setup_airflow.sh
```

Start Airflow:

```bash
./airflow/scripts/start_airflow.sh
```

Open:

```text
http://localhost:8080
```

Default local login:

```text
username: admin
password: admin
```

Stop Airflow:

```bash
./airflow/scripts/stop_airflow.sh
```

Test DAG import:

```bash
./airflow/scripts/test_dag.sh
```

Expected result:

```text
DAG test passed.
```

Note: local Airflow may show warnings about SQLite and SequentialExecutor. These warnings are acceptable for local development and portfolio demonstration. A production Airflow deployment should use PostgreSQL or MySQL as metadata DB and a production executor.

---

## 16. Screenshots

Recommended screenshots to keep in `docs/screenshots/`.

### 16.1 Core Pipeline Screenshots

| File | Purpose |
|---|---|
| `01_project_structure.png` | Shows clean project structure |
| `02_docker_services_running.png` | Shows Kafka, Zookeeper, and MongoDB running |
| `03_full_training_metrics.png` | Shows training, validation, and test metrics |
| `04_spark_streaming_to_mongodb.png` | Shows Spark streaming batches inserted into MongoDB |
| `05_producer_streaming_reviews.png` | Shows Kafka producer sending reviews |
| `06_mongodb_latest_predictions.png` | Shows MongoDB stored predictions |

### 16.2 Dashboard Screenshots

| File | Purpose |
|---|---|
| `13_dashboard_overview_kpis.png` | Shows dashboard KPIs and prediction summary |
| `14_dashboard_prediction_results_by_date.png` | Shows prediction results by review date |
| `15_dashboard_productid_filter.png` | Shows ProductId filter |
| `16_dashboard_productid_b001e4kfg0_analysis.png` | Shows required ProductId analysis |
| `17_dashboard_enriched_latest_events.png` | Shows metadata-preserving latest events table |
| `18_dashboard_pdf_report_export.png` | Shows PDF report export section |

### 16.3 Airflow Screenshots

| File | Purpose |
|---|---|
| `19_airflow_login_page.png` | Shows Airflow UI available locally |
| `20_airflow_dag_list_detected.png` | Shows DAG detected by Airflow |
| `21_airflow_dag_graph_view.png` | Shows DAG task structure |
| `22_airflow_dag_run_progress.png` | Shows DAG execution in progress |
| `23_airflow_dag_success_run.png` | Shows all DAG tasks completed successfully |

---

## 17. How to Run

The project uses short aliases stored in the WSL shell configuration.

### 17.1 Enter Project Environment

```bash
bigdata
```

Expected:

```text
(.venv) youssef@Youssef:/mnt/c/Users/Me/Desktop/END TO END DATA ENGINEERING PROJECTS/BIG DATA PROJECT$
```

### 17.2 Start Docker Services

Terminal 1:

```bash
bd-kafka
```

This starts Kafka, Zookeeper, and MongoDB.

### 17.3 Export Test Split

```bash
bd-test-export
```

This creates:

```text
data/processed/test_reviews.jsonl
```

### 17.4 Run Spark Streaming

Terminal 2:

```bash
bd-spark
```

Expected logs:

```text
LOADING SAVED PIPELINE MODEL
READING FROM KAFKA
STREAMING PREDICTIONS TO MONGODB STARTED
Batch X: inserted Y documents into MongoDB.
```

### 17.5 Run Producer

Terminal 3:

```bash
bd-producer
```

Expected logs:

```text
Sent row 1/56814 | product_id=B001E4KFG0 | review_date=2011-04-27 | score=5 | label=positive | source_split=product_demo
```

### 17.6 Run Dashboard

Terminal 4:

```bash
bd-streamlit
```

Then open the Streamlit URL shown in the terminal.

### 17.7 Run Airflow

```bash
./airflow/scripts/start_airflow.sh
```

Open:

```text
http://localhost:8080
```

Run the DAG manually:

```text
amazon_reviews_batch_orchestration
```

---

## 18. Validation Commands

### 18.1 Docker Services

```bash
cd kafka
docker compose ps
```

Expected services:

```text
kafka
zookeeper
mongodb
```

### 18.2 MongoDB Count

```bash
bd-mongo
```

Inside `mongosh`:

```javascript
use amazon_reviews_db

db.sentiment_predictions.countDocuments({
  source: "spark_structured_streaming"
})
```

### 18.3 Latest Predictions

```javascript
db.sentiment_predictions.find(
  { source: "spark_structured_streaming" },
  {
    _id: 0,
    product_id: 1,
    review_date: 1,
    score: 1,
    true_label: 1,
    predicted_label: 1,
    source_split: 1,
    batch_id: 1
  }
).sort({ processed_at: -1 }).limit(5).pretty()
```

### 18.4 Required ProductId Validation

```javascript
db.sentiment_predictions.find(
  { product_id: "B001E4KFG0" },
  {
    _id: 0,
    product_id: 1,
    review_date: 1,
    score: 1,
    true_label: 1,
    predicted_label: 1,
    source_split: 1,
    batch_id: 1
  }
).pretty()
```

Expected:

```text
product_id: B001E4KFG0
review_date: 2011-04-27
score: 5
true_label: positive
predicted_label: positive
source_split: product_demo
```

### 18.5 Airflow DAG Validation

```bash
./airflow/scripts/test_dag.sh
```

Expected:

```text
DAG test passed.
```

---

## 19. Git and Artifact Rules

Do not commit local data or generated heavy artifacts.

Recommended `.gitignore` coverage:

```gitignore
# Python
__pycache__/
*.pyc
.venv/
venv/
env/

# Raw and processed data
data/raw/
data/processed/

# Spark generated folders
src/spark/model/
src/spark/models/
spark-warehouse/
metastore_db/

# Airflow runtime
airflow/airflow_home/
airflow/logs/
*.pid

# Exports and backups
exports/
backups/
```

Commit source code, documentation, configuration, and screenshots:

```bash
git add README.md src/ kafka/ airflow/ docs/screenshots/ results/ requirements.txt .gitignore
git commit -m "Finalize Big Data streaming pipeline with Airflow orchestration"
git push origin main
```

---

## 20. Future Improvements

Planned future improvements:

1. Add a Flask or Django web dashboard for closer alignment with the professor’s suggested web technologies.
2. Add model comparison with multiple Spark ML models:
   - Logistic Regression,
   - Naive Bayes,
   - One-vs-Rest Linear SVC if practical.
3. Add automated data quality checks with Great Expectations or custom Spark checks.
4. Add MongoDB indexes for faster ProductId and date queries.
5. Add Dockerized Airflow using PostgreSQL metadata DB.
6. Add CI checks for Python formatting and import validation.
7. Add dashboard authentication for production use.
8. Add model drift monitoring and periodic retraining strategy.

---

## Final Notes

This project demonstrates the repeated professional Big Data pattern:

```text
Data Source
→ Ingestion
→ Message Broker
→ Distributed Processing
→ Machine Learning Inference
→ Storage
→ Dashboard
→ Orchestration
→ Documentation
```

The most important learning outcome is not only the Amazon Reviews use case, but the repeatable architecture pattern that can be reused across future Big Data projects.
.

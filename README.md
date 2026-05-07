# Amazon Reviews Big Data Sentiment Intelligence Platform

A complete end-to-end Big Data engineering project for processing Amazon product reviews, training Spark ML sentiment models, streaming review events through Kafka, generating real-time predictions with Spark Structured Streaming, storing enriched prediction results in MongoDB, orchestrating batch tasks with Apache Airflow, and visualizing the final results through a professional Flask / JavaScript web dashboard.

This project was built as a practical Big Data training pipeline. The goal is not only to classify reviews, but to demonstrate a repeatable professional architecture that can be reused with other datasets and business use cases.

---

## 1. Project Objective

The objective of this project is to build a production-style Big Data pipeline that can:

1. ingest Amazon review data,
2. clean and label the data for sentiment analysis,
3. train several Spark ML models,
4. compare model performance,
5. export a test split for simulated streaming,
6. send review events to Kafka,
7. consume Kafka events with Spark Structured Streaming,
8. predict sentiment using the selected production model,
9. store prediction results in MongoDB,
10. expose the results through a Flask / JavaScript web dashboard,
11. orchestrate batch preparation and training tasks with Airflow,
12. document the full project as a portfolio-ready data engineering system.

---

## 2. Final Architecture

```text
Amazon Reviews CSV
        |
        v
Batch Data Preparation
        |
        v
Spark ML Training and Model Comparison
        |
        v
Best Model Selection
        |
        v
Export Test Split for Streaming
        |
        v
Kafka Producer
        |
        v
Kafka Topic: amazon_reviews
        |
        v
Spark Structured Streaming
        |
        v
Best Single Spark ML Model: one_vs_rest_linear_svc
        |
        v
MongoDB: amazon_reviews_db.sentiment_predictions
        |
        v
Flask / JavaScript Web Dashboard
        |
        v
PDF Report + Portfolio Screenshots
```

---

## 3. Technology Stack

| Layer | Tool |
|---|---|
| Programming language | Python |
| Distributed processing | Apache Spark / PySpark |
| Machine learning | Spark MLlib |
| Streaming broker | Apache Kafka |
| Streaming processing | Spark Structured Streaming |
| Database | MongoDB |
| Web dashboard | Flask, HTML, CSS, JavaScript |
| Orchestration | Apache Airflow |
| Container services | Docker / Docker Compose |
| Version control | Git / GitHub |
| Development environment | WSL / Linux virtual environment |

---

## 4. Repository Structure

```text
.
├── README.md
├── project_context.md
├── requirements.txt
├── structure.txt
├── useful_commands.txt
│
├── airflow/
│   ├── dags/
│   │   └── amazon_reviews_batch_pipeline.py
│   └── scripts/
│       └── test_dag.sh
│
├── data/
│   ├── raw/
│   │   └── Reviews.csv                 # ignored by Git
│   └── processed/                      # ignored by Git
│
├── docs/
│   ├── PHASE5.md
│   ├── spark_sentiment_tuning_report.md
│   └── screenshots/
│       ├── flask_dashboard_overview.png
│       ├── flask_dashboard_overview_with_main_analytics.png
│       ├── flask_main_analytics_and_review_date.png
│       ├── flask_model_quality.png
│       ├── flask_streaming_operations.png
│       ├── flask_required_product_analysis.png
│       ├── flask_risk_monitoring.png
│       └── flask_latest_prediction_events.png
│
├── kafka/
│   ├── docker-compose.yml
│   └── topics.md
│
├── results/
│   ├── spark_sa_tuning_results.csv
│   └── spark_sa_tuning_results.md
│
├── scripts/
│   └── run_web_dashboard.sh
│
└── src/
    ├── __init__.py
    │
    ├── experiments/
    │   ├── __init__.py
    │   ├── preprocessing/
    │   │   ├── __init__.py
    │   │   ├── clean.py
    │   │   ├── dataset.py
    │   │   ├── label.py
    │   │   ├── resampling.py
    │   │   └── vectorizer.py
    │   └── training/
    │       ├── __init__.py
    │       └── train.py
    │
    ├── ingestion/
    │   ├── __init__.py
    │   └── producer.py
    │
    ├── spark/
    │   ├── model/
    │   │   └── ensemble_models/
    │   │       ├── logistic_regression/
    │   │       ├── naive_bayes/
    │   │       ├── one_vs_rest_linear_svc/
    │   │       └── ensemble_metadata.md
    │   │
    │   ├── streaming/
    │   │   ├── __init__.py
    │   │   ├── consumer.py
    │   │   └── predict_stream.py
    │   │
    │   └── training/
    │       ├── __init__.py
    │       ├── export_test_split_for_streaming.py
    │       ├── train_spark_pipeline.py
    │       ├── tune_spark_pipeline_sa.py
    │       └── models/
    │           ├── logistic_regression.py
    │           ├── naive_bayes.py
    │           ├── one_vs_rest_linear_svc.py
    │           └── model_comparison.py
    │
    ├── storage/
    │   ├── __init__.py
    │   ├── mongodb_writer.py
    │   └── test_mongodb_connection.py
    │
    └── web/
        ├── __init__.py
        ├── app.py
        ├── README_WEB.md
        ├── requirements-web.txt
        ├── templates/
        │   └── dashboard.html
        └── static/
            ├── css/
            │   └── style.css
            └── js/
                └── dashboard.js
```

---

## 5. Data Source

The project uses the Amazon Reviews dataset stored locally as:

```text
data/raw/Reviews.csv
```

The raw dataset is intentionally excluded from Git because it is large and should not be committed to GitHub.

`.gitignore` excludes:

```text
data/raw/Reviews.csv
data/raw/*.csv
data/processed/
```

This keeps the repository lightweight and professional.

---

## 6. Sentiment Labeling Strategy

The original dataset contains a numeric Amazon review score from 1 to 5.

The sentiment label is created as follows:

| Score | Sentiment |
|---|---|
| 1 or 2 | negative |
| 3 | neutral |
| 4 or 5 | positive |

This creates a three-class classification problem:

```text
positive
negative
neutral
```

This mapping is used consistently during training, validation, testing, streaming, and dashboard analysis.

---

## 7. Batch Training Pipeline

The training pipeline is implemented with Spark MLlib.

The main training pipeline performs:

```text
Load Reviews.csv
→ Select useful columns
→ Clean / prepare text
→ Create sentiment label
→ Split dataset into train / validation / test
→ Build text feature pipeline
→ Train Spark ML model
→ Evaluate validation metrics
→ Evaluate test metrics
→ Save trained model
```

The text pipeline uses:

```text
RegexTokenizer
→ StopWordsRemover
→ CountVectorizer
→ IDF
→ Spark ML classifier
```

This creates TF-IDF features from the review text and trains sentiment classification models.

---

## 8. Train / Validation / Test Split

The dataset is split into three parts:

```text
training set
validation set
test set
```

The validation set is used for model selection and tuning.

The test set is kept separate to evaluate the final model and to simulate streaming input.

The exported streaming test file is generated by:

```text
src/spark/training/export_test_split_for_streaming.py
```

It creates:

```text
data/processed/test_reviews.jsonl
```

This JSONL file is used by the Kafka producer to simulate real incoming review events.

---

## 9. Feature Engineering

The Spark feature engineering pipeline uses TF-IDF.

### Tokenization

`RegexTokenizer` splits raw review text into tokens.

Example:

```text
"This coffee tastes great!"
```

becomes:

```text
["this", "coffee", "tastes", "great"]
```

### Stop Word Removal

Common words such as `the`, `is`, `and`, `to` are removed because they often add little predictive value.

### CountVectorizer

`CountVectorizer` converts tokens into a sparse numerical vector based on a learned vocabulary.

Important parameters used during tuning included:

```text
vocabSize
minDF
```

### IDF

`IDF` reduces the importance of very common words and increases the importance of more discriminative words.

### Final Features

The final model input is:

```text
features
label_index
class_weight
```

---

## 10. Class Imbalance Handling

The Amazon Reviews dataset is naturally imbalanced because positive reviews are much more frequent than neutral or negative reviews.

To reduce the impact of class imbalance, class weights are computed and added to the training data.

Example class weights from a full training run:

```text
positive -> 0.4287
neutral  -> 4.4364
negative -> 2.2639
```

This gives more importance to minority classes during model training.

---

## 11. Model Training and Comparison

The project was expanded from a single Logistic Regression model to a multi-model Spark ML comparison system.

The trained models include:

```text
Logistic Regression
Naive Bayes
One-vs-Rest Linear SVC
Majority Vote Ensemble
```

The training folder was refactored so that each model has its own file, while a separate comparison/orchestration script handles training and evaluation.

This structure is more professional because:

```text
each model is isolated
model code is easier to maintain
model comparison is centralized
new models can be added cleanly
```

---

## 12. Model Selection

The final model strategy separates batch performance from streaming practicality.

### Best Batch Model

The best overall model from batch comparison is:

```text
majority_vote_ensemble
```

The ensemble combines:

```text
logistic_regression
naive_bayes
one_vs_rest_linear_svc
```

It performs majority voting over the predictions of the three models.

This model is useful for offline scoring, experimentation, and advanced model comparison.

### Best Streaming Production Model

The production streaming model is:

```text
one_vs_rest_linear_svc
```

This model was selected because it is the best single Spark PipelineModel from validation results and is lighter than the ensemble during streaming.

Final decision:

```text
Batch best model: majority_vote_ensemble
Streaming production model: one_vs_rest_linear_svc
```

This is a realistic production trade-off:

```text
ensemble = stronger offline performance but heavier inference
single best model = better streaming latency and simpler runtime
```

---

## 13. Important Note About Confidence

`OneVsRest LinearSVC` does not output calibrated probabilities by default.

Therefore, for the final streaming model:

```text
probability = []
confidence = null
confidence_status = "not_available"
confidence_available = false
```

The Flask dashboard correctly displays:

```text
Average Confidence: N/A
Confidence Coverage: 0.0%
```

This is intentional and correct. The dashboard does not invent false confidence values.

---

## 14. Kafka Layer

Kafka is used as the streaming message broker.

The Kafka service is defined in:

```text
kafka/docker-compose.yml
```

The main Kafka topic is:

```text
amazon_reviews
```

The role of Kafka is to decouple ingestion from processing:

```text
Producer sends review events
Kafka stores them in a topic
Spark Structured Streaming consumes them continuously
```

This makes the architecture more realistic and scalable.

---

## 15. Kafka Producer

The producer is implemented in:

```text
src/ingestion/producer.py
```

Its role is to simulate real-time incoming review events by reading from:

```text
data/processed/test_reviews.jsonl
```

and sending each review as a JSON message to Kafka.

Each Kafka message contains fields such as:

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

The producer prints sent rows so the user can monitor the stream in real time.

---

## 16. Spark Structured Streaming

The streaming prediction job is implemented in:

```text
src/spark/streaming/predict_stream.py
```

The current production version loads:

```text
src/spark/model/ensemble_models/one_vs_rest_linear_svc
```

Then it performs:

```text
Read Kafka topic amazon_reviews
→ Parse JSON messages
→ Build Spark ML input schema
→ Apply saved Spark PipelineModel
→ Generate predicted_label
→ Add model_type = one_vs_rest_linear_svc
→ Write predictions to MongoDB
```

Expected runtime output:

```text
LOADING BEST SINGLE SPARK MODEL
Model name: one_vs_rest_linear_svc
READING FROM KAFKA
STREAMING BEST SINGLE MODEL PREDICTIONS TO MONGODB STARTED
Batch 207: inserted 45 documents into MongoDB. model_type=['one_vs_rest_linear_svc'] | confidence_available=0/45
```

This confirms that the streaming pipeline is running correctly.

---

## 17. Why This Is Still Streaming

Spark Structured Streaming often works internally using micro-batches.

This means Spark does not necessarily process one event at a time. Instead, it continuously checks Kafka and processes newly arrived records in small groups.

Example:

```text
Batch 210: inserted 39 documents
Batch 211: inserted 38 documents
Batch 212: inserted 44 documents
```

This is still streaming because:

```text
the job stays alive
Kafka is continuously monitored
new records are processed as they arrive
offsets are managed by Spark
results are continuously written to MongoDB
```

This is called:

```text
micro-batch streaming
```

---

## 18. MongoDB Storage Layer

MongoDB is used as the serving database for prediction results.

Database:

```text
amazon_reviews_db
```

Collection:

```text
sentiment_predictions
```

MongoDB stores enriched prediction documents containing:

```text
schema_version
product_id
user_id
review_time
review_date
score
summary
text_preview
text
true_label
source_split
source_row_index
prediction
predicted_label
model_type
model_name
probability
confidence
confidence_status
confidence_available
batch_id
processed_at
source
```

The MongoDB writer is implemented in:

```text
src/storage/mongodb_writer.py
```

It also creates useful indexes for dashboard queries:

```text
processed_at
batch_id
product_id
predicted_label
true_label
score
model_type
source
source_split
```

This makes the Flask dashboard faster and more reliable.

---

## 19. Flask / JavaScript Web Dashboard

The final official dashboard is implemented with Flask, HTML, CSS, and JavaScript.

The dashboard files are located in:

```text
src/web/
```

The Flask backend is:

```text
src/web/app.py
```

The frontend template is:

```text
src/web/templates/dashboard.html
```

The CSS styling is:

```text
src/web/static/css/style.css
```

The JavaScript dashboard logic is:

```text
src/web/static/js/dashboard.js
```

The dashboard reads directly from MongoDB through Flask API endpoints.

This replaces the previous Streamlit prototype as the official project interface.

---

## 20. Flask API Endpoints

The Flask application exposes several API endpoints:

| Endpoint | Purpose |
|---|---|
| `/` | Main dashboard page |
| `/api/health` | MongoDB connection health |
| `/api/options` | ProductId, batch, model, and source filter options |
| `/api/summary` | Main KPIs |
| `/api/charts` | Chart data |
| `/api/risk` | Suspicious and risk samples |
| `/api/latest` | Latest prediction events |
| `/api/product/<product_id>` | Product-level analysis |
| `/api/report.pdf` | Downloadable PDF report |

This gives the dashboard a clean backend/frontend separation.

---

## 21. Dashboard Features

The Flask dashboard includes the following sections.

### Pipeline Status

Shows:

```text
MongoDB status
Total predictions
Latest batch ID
Streamed accuracy
Positive predictions
Negative predictions
Neutral predictions
Average confidence
Confidence coverage
Low confidence count
Labeled samples
Latest processed timestamp
```

### Main Analytics

Shows:

```text
Sentiment distribution
Model type distribution
Amazon score distribution
Score by predicted sentiment
Prediction results by review date
Streaming source split distribution
```

### Model Quality Analytics

Shows:

```text
Confidence distribution
Average confidence by sentiment
Average confidence by model
True vs predicted confusion matrix
```

For the current `one_vs_rest_linear_svc` model, confidence charts correctly show no confidence data.

### Streaming Operations

Shows:

```text
Records per micro-batch
Total micro-batches shown
Total records shown
Average records per batch
Minimum records in batch
Maximum records in batch
Latest batch ID
```

### Required ProductId Analysis

The dashboard includes a dedicated analysis section for:

```text
B001E4KFG0
```

This section shows:

```text
product prediction count
review score
true label
predicted label
confidence
review date
model type
batch ID
product sentiment distribution
product score distribution
```

### Risk Monitoring

Shows:

```text
suspicious prediction samples
low-confidence samples
```

Suspicious samples include cases such as:

```text
score >= 4 predicted as negative
score <= 2 predicted as positive
score = 5 predicted as neutral
score = 1 predicted as neutral
```

### Latest Prediction Events

Shows the most recent MongoDB prediction documents written by Spark Streaming.

---

## 22. Official Dashboard Screenshots

The final project uses the Flask / JavaScript dashboard screenshots below.

### Dashboard Overview

![Flask Dashboard Overview](docs/screenshots/flask_dashboard_overview.png)

### Main Analytics and Review Date Analysis

![Flask Main Analytics](docs/screenshots/flask_main_analytics_and_review_date.png)

### Model Quality Analytics

![Flask Model Quality Analytics](docs/screenshots/flask_model_quality.png)

### Streaming Operations

![Flask Streaming Operations](docs/screenshots/flask_streaming_operations.png)

### Required ProductId Analysis

![Required ProductId Analysis](docs/screenshots/flask_required_product_analysis.png)

### Risk Monitoring

![Risk Monitoring](docs/screenshots/flask_risk_monitoring.png)

### Latest Prediction Events

![Latest Prediction Events](docs/screenshots/flask_latest_prediction_events.png)

---

## 23. Apache Airflow Orchestration

Airflow is used to orchestrate the batch side of the project.

The DAG is implemented in:

```text
airflow/dags/amazon_reviews_batch_pipeline.py
```

The DAG name is:

```text
amazon_reviews_batch_orchestration
```

It includes tasks such as:

```text
start_batch_orchestration
check_project_structure
check_raw_dataset_exists
export_test_split_for_streaming
validate_streaming_export
train_spark_model
validate_saved_model
print_next_runtime_commands
end_batch_orchestration
```

The DAG is designed for the batch preparation and training side, not for long-running streaming jobs.

This is intentional because Spark Streaming jobs are long-running services, while Airflow is better suited for scheduled or manually triggered batch workflows.

---

## 24. Why Airflow Does Not Run Streaming Directly

Airflow is used here to orchestrate bounded tasks:

```text
validate project structure
check dataset
export streaming data
train model
validate saved model
print runtime commands
```

The streaming runtime is handled separately because these services are long-running:

```text
Kafka
Spark Structured Streaming
MongoDB
Flask dashboard
```

A streaming job is expected to keep running, while Airflow tasks are expected to start, complete, and report success or failure.

Professional interpretation:

```text
Airflow prepares and validates the pipeline.
Kafka and Spark Streaming run the real-time system.
Flask visualizes the results.
```

---

## 25. Runtime Commands

The project uses WSL/Linux commands.

### Start Kafka and MongoDB

```bash
cd kafka
docker compose up -d
```

Check services:

```bash
docker compose ps
```

Expected services:

```text
kafka
zookeeper
mongodb
```

### Train Spark Models

```bash
spark-submit src/spark/training/train_spark_pipeline.py
```

For cleaner logs:

```bash
spark-submit src/spark/training/train_spark_pipeline.py 2>/dev/null | grep -A 20 -E "==========|Accuracy|Macro F1|Positive F1|Negative F1|Neutral F1|Model saved"
```

### Export Test Split for Streaming

```bash
python src/spark/training/export_test_split_for_streaming.py
```

### Start Spark Streaming

```bash
spark-submit src/spark/streaming/predict_stream.py
```

### Start Kafka Producer

```bash
python src/ingestion/producer.py
```

### Open MongoDB Shell

```bash
docker exec -it mongodb mongosh
```

Then:

```javascript
use amazon_reviews_db

db.sentiment_predictions.find(
  {},
  {
    _id: 0,
    product_id: 1,
    score: 1,
    true_label: 1,
    predicted_label: 1,
    model_type: 1,
    confidence: 1,
    confidence_status: 1,
    confidence_available: 1,
    batch_id: 1,
    source: 1
  }
).sort({ processed_at: -1 }).limit(5).pretty()
```

### Start Flask Dashboard

```bash
python src/web/app.py
```

Or:

```bash
./scripts/run_web_dashboard.sh
```

Open:

```text
http://localhost:5000
```

---

## 26. Useful Aliases

The project can be run more easily with aliases.

Example aliases:

```bash
alias bigdata='cd "/mnt/c/Users/Me/Desktop/END TO END DATA ENGINEERING PROJECTS/BIG DATA PROJECT" && source .venv/bin/activate'
alias bd-kafka='cd "/mnt/c/Users/Me/Desktop/END TO END DATA ENGINEERING PROJECTS/BIG DATA PROJECT/kafka" && docker compose up'
alias bd-spark='cd "/mnt/c/Users/Me/Desktop/END TO END DATA ENGINEERING PROJECTS/BIG DATA PROJECT" && source .venv/bin/activate && spark-submit src/spark/streaming/predict_stream.py'
alias bd-producer='cd "/mnt/c/Users/Me/Desktop/END TO END DATA ENGINEERING PROJECTS/BIG DATA PROJECT" && source .venv/bin/activate && python src/ingestion/producer.py'
alias bd-mongo='docker exec -it mongodb mongosh'
alias bd-web='cd "/mnt/c/Users/Me/Desktop/END TO END DATA ENGINEERING PROJECTS/BIG DATA PROJECT" && source .venv/bin/activate && export PYTHONPATH=$PWD && python src/web/app.py'
```

---

## 27. Airflow Commands

Start Airflow from the project environment:

```bash
airflow standalone
```

Open:

```text
http://localhost:8080
```

Test DAG import:

```bash
./airflow/scripts/test_dag.sh
```

The DAG should appear as:

```text
amazon_reviews_batch_orchestration
```

When the DAG succeeds, the graph should show all tasks in green.

---

## 28. Example MongoDB Document

A clean final document written by the streaming pipeline looks like:

```javascript
{
  schema_version: "v2",
  product_id: "B001E4KFG0",
  review_date: "2011-04-27",
  score: 5,
  true_label: "positive",
  prediction: 0.0,
  predicted_label: "positive",
  model_type: "one_vs_rest_linear_svc",
  model_name: "one_vs_rest_linear_svc",
  probability: [],
  confidence: null,
  confidence_status: "not_available",
  confidence_available: false,
  source_split: "test",
  batch_id: 208,
  source: "spark_structured_streaming",
  processed_at: ISODate(...)
}
```

This document format is dashboard-friendly because it contains both the prediction result and useful metadata.

---

## 29. Project Phases Completed

### Phase 1 — Project Architecture

Completed.

Defined an end-to-end Big Data architecture using:

```text
source
ingestion
Kafka
Spark
Spark ML
MongoDB
dashboard
Airflow
documentation
```

### Phase 2 — Dataset Selection

Completed.

Selected Amazon Reviews dataset and defined sentiment labels from review scores.

### Phase 3 — Batch Preprocessing

Completed.

Prepared text, labels, and train/validation/test splits.

### Phase 4 — Spark ML Baseline

Completed.

Built baseline TF-IDF + Logistic Regression model.

### Phase 5 — Model Tuning

Completed.

Tuned parameters such as:

```text
vocabSize
minDF
maxIter
regParam
bigrams
class weights
```

### Phase 6 — Model Comparison

Completed.

Added multiple model files and comparison workflow:

```text
Logistic Regression
Naive Bayes
One-vs-Rest Linear SVC
Majority Vote Ensemble
```

### Phase 7 — Kafka Ingestion

Completed.

Built Kafka producer and Kafka topic setup.

### Phase 8 — Spark Structured Streaming

Completed.

Built streaming prediction job using the best single model:

```text
one_vs_rest_linear_svc
```

### Phase 9 — MongoDB Storage

Completed.

Built MongoDB writer with model metadata, schema versioning, confidence metadata, and dashboard indexes.

### Phase 10 — Flask Web Dashboard

Completed.

Built professional Flask / JavaScript dashboard with:

```text
KPIs
charts
filters
risk monitoring
product analysis
PDF export
```

### Phase 11 — Airflow Orchestration

Completed.

Built Airflow DAG for batch orchestration and model preparation tasks.

### Phase 12 — GitHub Cleanup

Completed in progress.

Large files are ignored, screenshots are organized, and the README describes the final architecture.

---

## 30. What Was Improved During the Project

The project evolved significantly through multiple professional refactors.

### Initial Version

The initial version used:

```text
single Spark Logistic Regression model
basic streaming
Streamlit dashboard prototype
```

### Final Version

The final version uses:

```text
multiple Spark ML models
model comparison
majority vote ensemble for batch evaluation
one_vs_rest_linear_svc for production streaming
MongoDB metadata schema
Flask / JavaScript professional web dashboard
Airflow batch orchestration
clean screenshots and documentation
```

---

## 31. Important Engineering Decisions

### Decision 1 — Use Spark MLlib

Spark MLlib was used because the project focuses on Big Data engineering and distributed processing.

### Decision 2 — Use Kafka

Kafka was used to simulate a real streaming ingestion layer and decouple producers from consumers.

### Decision 3 — Use MongoDB

MongoDB was selected as a flexible document database for storing prediction results and metadata.

### Decision 4 — Use Flask Instead of Streamlit for Final Interface

Streamlit was useful as a prototype, but the final project uses Flask / JavaScript because it better matches the project requirement for a web interface.

### Decision 5 — Use One-vs-Rest Linear SVC for Streaming

The ensemble model is stronger offline but heavier for streaming.

`one_vs_rest_linear_svc` is used as the final streaming model because it is the best single Spark PipelineModel and provides better runtime simplicity.

### Decision 6 — Display Confidence as N/A

The dashboard displays confidence as `N/A` when the active model does not provide calibrated probabilities. This is more honest and professional than inventing a fake confidence score.

---

## 32. How to Run the Full Project

### Step 1 — Start the Environment

```bash
bigdata
```

### Step 2 — Start Kafka and MongoDB

```bash
cd kafka
docker compose up -d
```

### Step 3 — Train or Validate Models

```bash
spark-submit src/spark/training/train_spark_pipeline.py
```

### Step 4 — Export Streaming Data

```bash
python src/spark/training/export_test_split_for_streaming.py
```

### Step 5 — Start Spark Streaming

```bash
spark-submit src/spark/streaming/predict_stream.py
```

### Step 6 — Start Kafka Producer

```bash
python src/ingestion/producer.py
```

### Step 7 — Start Flask Web Dashboard

```bash
python src/web/app.py
```

Open:

```text
http://localhost:5000
```

---

## 33. Validation Checklist

Before considering the project ready, verify:

```text
Kafka and MongoDB containers are running
Spark model exists in src/spark/model/ensemble_models/one_vs_rest_linear_svc
Producer sends rows to Kafka
Spark inserts rows into MongoDB
MongoDB documents contain model_type = one_vs_rest_linear_svc
confidence_available = false for LinearSVC
Flask dashboard shows connected status
Flask dashboard shows model type correctly
Airflow DAG imports successfully
README screenshots are Flask screenshots, not Streamlit screenshots
```

---

## 34. Current Final State

The current final runtime state is:

```text
Official dashboard: Flask / JavaScript
Streaming model: one_vs_rest_linear_svc
Batch best model: majority_vote_ensemble
Storage: MongoDB
Orchestration: Airflow for batch preparation/training
Streaming: Kafka + Spark Structured Streaming
```

The project is now portfolio-ready as a complete Big Data engineering pipeline.

---

## 35. Future Improvements

The following improvements can be added later:

```text
Dockerize the Flask dashboard
Add authentication to the web dashboard
Add PostgreSQL or data warehouse layer for analytical reporting
Add Prometheus / Grafana monitoring
Add model drift monitoring
Add scheduled Airflow retraining
Add REST API for manual review scoring
Add CI/CD GitHub Actions
Add unit tests for data transformations
Add integration tests for Kafka and MongoDB
Deploy the web dashboard to a server
```

These are future improvements and are not claimed as completed features.

---

## 36. Final Project Summary

This project demonstrates a complete Big Data pipeline:

```text
Data Source
→ Ingestion
→ Kafka Streaming Layer
→ Spark Distributed Processing
→ Spark ML Training
→ Model Comparison
→ Spark Structured Streaming Prediction
→ MongoDB Storage
→ Flask Web Dashboard
→ Airflow Orchestration
→ Documentation and Portfolio Delivery
```

It shows both engineering and machine learning thinking:

```text
data pipeline design
streaming architecture
distributed ML training
model selection
trade-off between accuracy and streaming latency
dashboard design
orchestration
documentation
GitHub cleanup
```

The most important final decision is:

```text
Use the ensemble model for offline model comparison,
but use one_vs_rest_linear_svc as the production streaming model.
```

This gives the project a realistic production story:

```text
best possible model for analysis
best practical model for streaming
honest dashboard metrics
clean architecture
professional documentation
```

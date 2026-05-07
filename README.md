# Amazon Reviews Streaming Sentiment Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?logo=apachespark&logoColor=white)
![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-231F20?logo=apachekafka&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?logo=mongodb&logoColor=white)
![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?logo=apacheairflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

An end-to-end Big Data engineering pipeline that trains multiple Spark ML sentiment classification models on Amazon product reviews, streams predictions in real time through Kafka and Spark Structured Streaming, persists enriched results in MongoDB, and visualizes live pipeline output through a professional Flask/JavaScript web dashboard — all orchestrated with Apache Airflow.

This project is built as a portfolio-grade, production-style Big Data system demonstrating a complete data engineering workflow: from raw data ingestion to real-time inference and interactive visualization.

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Repository Structure](#4-repository-structure)
5. [Data Source](#5-data-source)
6. [Sentiment Labeling Strategy](#6-sentiment-labeling-strategy)
7. [Feature Engineering Pipeline](#7-feature-engineering-pipeline)
8. [Class Imbalance Handling](#8-class-imbalance-handling)
9. [Model Training and Comparison](#9-model-training-and-comparison)
10. [Model Selection Rationale](#10-model-selection-rationale)
11. [Confidence Reporting](#11-confidence-reporting)
12. [Kafka Streaming Layer](#12-kafka-streaming-layer)
13. [Spark Structured Streaming](#13-spark-structured-streaming)
14. [MongoDB Storage Layer](#14-mongodb-storage-layer)
15. [Flask Web Dashboard](#15-flask-web-dashboard)
16. [Flask API Endpoints](#16-flask-api-endpoints)
17. [Dashboard Features](#17-dashboard-features)
18. [Dashboard Screenshots](#18-dashboard-screenshots)
19. [Apache Airflow Orchestration](#19-apache-airflow-orchestration)
20. [Getting Started](#20-getting-started)
21. [Runtime Commands Reference](#21-runtime-commands-reference)
22. [Example MongoDB Document](#22-example-mongodb-document)
23. [Key Engineering Decisions](#23-key-engineering-decisions)
24. [Future Improvements](#24-future-improvements)

---

## 1. Project Objective

The goal of this project is to build a production-style Big Data pipeline that demonstrates a complete, repeatable data engineering architecture applicable to any review classification or streaming NLP use case.

The pipeline covers every stage of the engineering lifecycle:

- Ingest and clean Amazon review data using PySpark
- Label reviews for three-class sentiment classification
- Train and compare multiple Spark MLlib models
- Export a held-out test split for simulated real-time streaming
- Produce review events to a Kafka topic
- Consume events with Spark Structured Streaming and run live inference
- Persist enriched prediction documents in MongoDB
- Expose pipeline results through a Flask/JavaScript web dashboard
- Orchestrate batch preparation and training with Apache Airflow

---

## 2. Architecture

```
Amazon Reviews CSV
        │
        ▼
┌─────────────────────────┐
│  Batch Preprocessing    │  (PySpark)
│  Clean · Label · Split  │
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│     Spark MLlib Training             │
│                                      │
│  ├── Logistic Regression             │
│  ├── Naive Bayes                     │
│  ├── One-vs-Rest Linear SVC          │
│  └── Majority Vote Ensemble  ───────►│ Best Batch Model
└──────────┬───────────────────────────┘
           │
           ▼
  Export Test Split
  data/processed/test_reviews.jsonl
           │
           ▼
  ┌─────────────────┐
  │  Kafka Producer │
  └────────┬────────┘
           │
           ▼
  Topic: amazon_reviews
           │
           ▼
  ┌───────────────────────────┐
  │  Spark Structured         │
  │  Streaming                │
  │  (one_vs_rest_linear_svc) │──► Best Streaming Model
  └───────────┬───────────────┘
              │
              ▼
  MongoDB: amazon_reviews_db
           .sentiment_predictions
              │
              ▼
  ┌──────────────────────────────┐
  │  Flask / JavaScript          │
  │  Web Dashboard               │
  │  (live reads from MongoDB)   │
  └──────────────────────────────┘
              │
              ▼
      Airflow DAG orchestrates
      the full batch side above
```

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Distributed Processing | Apache Spark / PySpark |
| Machine Learning | Spark MLlib |
| Streaming Broker | Apache Kafka |
| Stream Processing | Spark Structured Streaming |
| Database | MongoDB |
| Web Dashboard | Flask, HTML, CSS, JavaScript |
| Orchestration | Apache Airflow |
| Containerization | Docker / Docker Compose |
| Version Control | Git / GitHub |
| Runtime Environment | WSL / Linux |

---

## 4. Repository Structure

```
.
├── README.md
├── requirements.txt
│
├── airflow/
│   ├── dags/
│   │   └── amazon_reviews_batch_pipeline.py   # Airflow DAG
│   └── scripts/
│       └── test_dag.sh
│
├── data/
│   ├── raw/
│   │   └── Reviews.csv                        # git-ignored — download separately
│   └── processed/                             # git-ignored — generated at runtime
│
├── docs/
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
    ├── ingestion/
    │   └── producer.py                        # Kafka producer
    │
    ├── spark/
    │   ├── model/
    │   │   └── ensemble_models/
    │   │       ├── logistic_regression/
    │   │       ├── naive_bayes/
    │   │       ├── one_vs_rest_linear_svc/    # Production streaming model
    │   │       └── ensemble_metadata.md
    │   │
    │   ├── streaming/
    │   │   ├── consumer.py
    │   │   └── predict_stream.py              # Spark Structured Streaming job
    │   │
    │   └── training/
    │       ├── train_spark_pipeline.py
    │       ├── tune_spark_pipeline_sa.py
    │       ├── export_test_split_for_streaming.py
    │       └── models/
    │           ├── logistic_regression.py
    │           ├── naive_bayes.py
    │           ├── one_vs_rest_linear_svc.py
    │           └── model_comparison.py
    │
    ├── storage/
    │   ├── mongodb_writer.py
    │   └── test_mongodb_connection.py
    │
    └── web/
        ├── app.py                             # Flask backend
        ├── templates/
        │   └── dashboard.html
        └── static/
            ├── css/style.css
            └── js/dashboard.js
```

The `data/raw/` and `data/processed/` directories are excluded from Git. The repository stays lightweight; all data artifacts are generated locally at runtime.

---

## 5. Data Source

The project uses the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset, stored locally as:

```
data/raw/Reviews.csv
```

The raw dataset is excluded from version control via `.gitignore`:

```
data/raw/Reviews.csv
data/raw/*.csv
data/processed/
```

Download the dataset from Kaggle and place it at `data/raw/Reviews.csv` before running the pipeline.

---

## 6. Sentiment Labeling Strategy

The dataset contains a numeric Amazon review score from 1 to 5. These scores are mapped to a three-class sentiment label as follows:

| Score | Sentiment |
|---|---|
| 1 or 2 | `negative` |
| 3 | `neutral` |
| 4 or 5 | `positive` |

This mapping is applied consistently at every stage: training, validation, testing, streaming, and the dashboard. All model evaluation metrics, MongoDB documents, and dashboard visualizations use this label definition.

---

## 7. Feature Engineering Pipeline

The Spark ML feature pipeline converts raw review text into numerical features using a TF-IDF approach.

### Tokenization

`RegexTokenizer` splits the raw review text into lowercase tokens, discarding punctuation and special characters.

```
"This coffee tastes great!"  →  ["this", "coffee", "tastes", "great"]
```

### Stop Word Removal

`StopWordsRemover` filters out high-frequency, low-signal words such as `the`, `is`, `and`, and `to`. These terms appear across all sentiment classes and add noise without improving discriminative power.

### CountVectorizer

`CountVectorizer` learns a vocabulary from the training corpus and converts each document's token list into a sparse term-frequency vector. Key tuning parameters include `vocabSize` (maximum vocabulary size) and `minDF` (minimum document frequency threshold).

### IDF Weighting

`IDF` (Inverse Document Frequency) down-weights terms that appear frequently across all documents and up-weights terms that are more discriminative. Combined with term frequency from `CountVectorizer`, this produces standard TF-IDF feature vectors.

### Final Model Input

Each training sample is represented by:

| Field | Description |
|---|---|
| `features` | TF-IDF sparse vector |
| `label_index` | Encoded sentiment class (0, 1, 2) |
| `class_weight` | Per-sample weight for imbalance correction |

---

## 8. Class Imbalance Handling

The Amazon Reviews dataset is strongly skewed toward positive reviews. Without correction, models tend to overfit the majority class and underperform on neutral and negative reviews.

Per-class weights are computed from the training distribution and injected as sample weights during model fitting. Example weights from a full training run:

| Class | Weight |
|---|---|
| `positive` | 0.4287 |
| `negative` | 2.2639 |
| `neutral` | 4.4364 |

Higher weights are assigned to minority classes (`neutral`, `negative`), penalizing misclassifications of these classes more strongly during optimization.

---

## 9. Model Training and Comparison

The project trains four models using Spark MLlib. Each model is defined in its own file under `src/spark/training/models/`, while `model_comparison.py` handles the joint training and evaluation workflow.

| Model | File |
|---|---|
| Logistic Regression | `logistic_regression.py` |
| Naive Bayes | `naive_bayes.py` |
| One-vs-Rest Linear SVC | `one_vs_rest_linear_svc.py` |
| Majority Vote Ensemble | `model_comparison.py` |

The **Majority Vote Ensemble** combines the predictions of all three single models by majority vote. It does not require a separate training step — it aggregates the outputs of the three trained `PipelineModel` objects at inference time.

Each model is evaluated on the held-out validation set using:

- Overall accuracy
- Macro F1 score
- Per-class F1 scores (positive, negative, neutral)

Results are saved to `results/spark_sa_tuning_results.csv` for comparison.

### Training Command

```bash
spark-submit src/spark/training/train_spark_pipeline.py
```

For cleaner console output, filter Spark logs:

```bash
spark-submit src/spark/training/train_spark_pipeline.py 2>/dev/null \
  | grep -E "==========|Accuracy|Macro F1|Positive F1|Negative F1|Neutral F1|Model saved"
```

---

## 10. Model Selection Rationale

The project deliberately uses two different models for two different contexts.

| Context | Selected Model | Reason |
|---|---|---|
| Offline / batch evaluation | `majority_vote_ensemble` | Highest accuracy — aggregates all three classifiers via majority voting |
| Production streaming | `one_vs_rest_linear_svc` | Best single `PipelineModel`; lower inference overhead for continuous micro-batch processing |

The ensemble cannot be saved as a single Spark `PipelineModel` — it requires all three models to be loaded and their predictions combined at runtime. This adds memory and coordination overhead that is acceptable for offline scoring but introduces unnecessary complexity in a long-running streaming job.

`one_vs_rest_linear_svc` is the top-performing individual model and is serialized as a standard `PipelineModel`, making it straightforward to load and apply within Spark Structured Streaming.

---

## 11. Confidence Reporting

`OneVsRest LinearSVC` does not output calibrated class probabilities by default. As a result, the streaming prediction documents store:

```json
{
  "probability": [],
  "confidence": null,
  "confidence_status": "not_available",
  "confidence_available": false
}
```

The Flask dashboard reflects this honestly:

```
Average Confidence:   N/A
Confidence Coverage:  0.0%
```

No fabricated confidence values are generated. The dashboard's Model Quality section correctly shows empty confidence charts when the active streaming model does not support probability output.

---

## 12. Kafka Streaming Layer

Kafka is used as the message broker between the ingestion layer and the Spark processing layer. This decouples producers from consumers, making the architecture more scalable and realistic.

**Docker Compose definition:** `kafka/docker-compose.yml`
**Topic name:** `amazon_reviews`

The producer (`src/ingestion/producer.py`) reads from the exported test split at `data/processed/test_reviews.jsonl` and publishes one review at a time as a JSON message to the Kafka topic. Each message contains:

| Field | Description |
|---|---|
| `product_id` | Amazon product identifier |
| `user_id` | Reviewer identifier |
| `review_time` | Unix timestamp |
| `review_date` | Human-readable date |
| `score` | Original Amazon score (1–5) |
| `summary` | Review headline |
| `text` | Full review body |
| `label` | Ground-truth sentiment label |
| `source_split` | Dataset split (`test`) |
| `source_row_index` | Row index in original dataset |

The producer prints each sent row to the console for real-time monitoring.

---

## 13. Spark Structured Streaming

The streaming prediction job is implemented in `src/spark/streaming/predict_stream.py`.

It loads the saved `one_vs_rest_linear_svc` PipelineModel and runs the following pipeline continuously:

```
Read Kafka topic: amazon_reviews
    │
    ▼
Parse JSON messages into a Spark DataFrame
    │
    ▼
Apply saved Spark PipelineModel (TF-IDF + LinearSVC)
    │
    ▼
Generate predicted_label per review
    │
    ▼
Enrich document with model metadata
    │
    ▼
Write to MongoDB: sentiment_predictions
```

### Micro-Batch Processing

Spark Structured Streaming operates in micro-batch mode. Rather than processing one event at a time, Spark continuously polls Kafka and processes newly arrived records in small groups. This is still streaming because:

- The Spark job runs indefinitely
- Kafka offsets are tracked and managed automatically
- New records are processed as soon as they arrive
- Results are written to MongoDB continuously

**Example runtime output:**

```
LOADING BEST SINGLE SPARK MODEL
Model name: one_vs_rest_linear_svc
READING FROM KAFKA
STREAMING BEST SINGLE MODEL PREDICTIONS TO MONGODB STARTED
Batch 207: inserted 45 documents — model_type=['one_vs_rest_linear_svc'] | confidence_available=0/45
Batch 208: inserted 39 documents — model_type=['one_vs_rest_linear_svc'] | confidence_available=0/39
Batch 209: inserted 44 documents — model_type=['one_vs_rest_linear_svc'] | confidence_available=0/44
```

---

## 14. MongoDB Storage Layer

MongoDB stores all enriched prediction results produced by the Spark Streaming job.

**Database:** `amazon_reviews_db`
**Collection:** `sentiment_predictions`
**Writer:** `src/storage/mongodb_writer.py`

Each document contains the following fields:

| Field | Description |
|---|---|
| `schema_version` | Document schema version (`v2`) |
| `product_id` | Amazon product identifier |
| `user_id` | Reviewer identifier |
| `review_time` | Unix timestamp |
| `review_date` | Review date string |
| `score` | Original Amazon score |
| `summary` | Review headline |
| `text_preview` | Truncated review text |
| `text` | Full review body |
| `true_label` | Ground-truth sentiment |
| `source_split` | Dataset split (`test`) |
| `source_row_index` | Source row index |
| `prediction` | Raw model output (numeric) |
| `predicted_label` | Decoded sentiment label |
| `model_type` | Model identifier |
| `model_name` | Model display name |
| `probability` | Class probabilities (empty for LinearSVC) |
| `confidence` | Max class probability (`null` for LinearSVC) |
| `confidence_status` | `available` or `not_available` |
| `confidence_available` | Boolean flag |
| `batch_id` | Spark micro-batch ID |
| `processed_at` | UTC timestamp of insertion |
| `source` | Source identifier (`spark_structured_streaming`) |

The writer also creates the following indexes to support fast dashboard queries:

```
processed_at · batch_id · product_id · predicted_label
true_label · score · model_type · source · source_split
```

---

## 15. Flask Web Dashboard

The web dashboard is implemented with Flask as the backend and plain HTML, CSS, and JavaScript on the frontend. It reads live data directly from MongoDB through a set of REST API endpoints.

**Key files:**

| File | Purpose |
|---|---|
| `src/web/app.py` | Flask application and API routes |
| `src/web/templates/dashboard.html` | Main dashboard template |
| `src/web/static/css/style.css` | Styling |
| `src/web/static/js/dashboard.js` | Frontend chart and filter logic |

The dashboard replaced an earlier Streamlit prototype. The Flask/JavaScript approach provides better control over layout, interactivity, and PDF export functionality.

---

## 16. Flask API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Renders the main dashboard page |
| `/api/health` | GET | MongoDB connection status |
| `/api/summary` | GET | KPI metrics (totals, accuracy, label counts) |
| `/api/charts` | GET | Data for all dashboard charts |
| `/api/risk` | GET | Suspicious and low-confidence samples |
| `/api/latest` | GET | Most recent prediction documents |
| `/api/options` | GET | Filter options: product IDs, batches, models, sources |
| `/api/product/<product_id>` | GET | Per-product sentiment and score analysis |
| `/api/report.pdf` | GET | Downloadable PDF pipeline report |

---

## 17. Dashboard Features

### Pipeline Status Panel

Displays a real-time summary of the running pipeline:

- MongoDB connection status
- Total prediction count
- Latest batch ID and processed timestamp
- Streamed accuracy (predicted vs. true label)
- Breakdown by sentiment class (positive / negative / neutral)
- Average confidence, confidence coverage, and low-confidence count

### Main Analytics

- Sentiment distribution chart
- Model type distribution
- Amazon score distribution (1–5)
- Score grouped by predicted sentiment
- Predictions over review date
- Streaming source split distribution

### Model Quality Analytics

- Confidence score distribution
- Average confidence by sentiment class
- Average confidence by model type
- True-label vs. predicted-label confusion matrix

> For the current `one_vs_rest_linear_svc` streaming model, confidence charts are intentionally empty as this model does not output calibrated probabilities.

### Streaming Operations

- Records processed per micro-batch (bar chart)
- Total micro-batches and total records processed
- Average, minimum, and maximum records per batch
- Latest batch ID

### Product-Level Analysis

A dedicated section for per-product analysis (e.g., product `B001E4KFG0`), showing:

- Prediction count for that product
- Per-review: score, true label, predicted label, confidence, date, model type, batch ID
- Product-level sentiment distribution chart
- Product-level score distribution chart

### Risk Monitoring

Surfaces potentially problematic predictions for manual review:

| Risk Type | Definition |
|---|---|
| Suspicious high-score | Score ≥ 4 predicted as `negative` |
| Suspicious low-score | Score ≤ 2 predicted as `positive` |
| Suspicious 5-star | Score = 5 predicted as `neutral` |
| Suspicious 1-star | Score = 1 predicted as `neutral` |
| Low-confidence samples | Confidence below a defined threshold |

### Latest Prediction Events

A live table showing the most recently inserted MongoDB documents from the Spark Streaming job.

---

## 18. Dashboard Screenshots

### Dashboard Overview
![Dashboard Overview](docs/screenshots/flask_dashboard_overview.png)

### Overview with Main Analytics
![Dashboard Overview with Analytics](docs/screenshots/flask_dashboard_overview_with_main_analytics.png)

### Main Analytics and Review Date Analysis
![Main Analytics](docs/screenshots/flask_main_analytics_and_review_date.png)

### Model Quality Analytics
![Model Quality](docs/screenshots/flask_model_quality.png)

### Streaming Operations
![Streaming Operations](docs/screenshots/flask_streaming_operations.png)

### Product-Level Analysis
![Product Analysis](docs/screenshots/flask_required_product_analysis.png)

### Risk Monitoring
![Risk Monitoring](docs/screenshots/flask_risk_monitoring.png)

### Latest Prediction Events
![Latest Events](docs/screenshots/flask_latest_prediction_events.png)

---

## 19. Apache Airflow Orchestration

Airflow orchestrates the **batch side** of the pipeline — the bounded, sequential tasks that prepare data and train models. Long-running streaming services (Kafka, Spark Streaming, MongoDB, Flask) are started separately.

**DAG file:** `airflow/dags/amazon_reviews_batch_pipeline.py`
**DAG name:** `amazon_reviews_batch_orchestration`

### DAG Task Sequence

```
start_batch_orchestration
        │
        ▼
check_project_structure
        │
        ▼
check_raw_dataset_exists
        │
        ▼
export_test_split_for_streaming
        │
        ▼
validate_streaming_export
        │
        ▼
train_spark_model
        │
        ▼
validate_saved_model
        │
        ▼
print_next_runtime_commands
        │
        ▼
end_batch_orchestration
```

### Responsibility Split

| Component | Role |
|---|---|
| Airflow DAG | Validates, prepares, and trains — bounded batch tasks |
| Kafka | Continuous event streaming |
| Spark Structured Streaming | Continuous model inference |
| MongoDB | Persistent prediction storage |
| Flask Dashboard | Live visualization |

Airflow is not used to manage long-running streaming jobs because those services are designed to stay alive indefinitely. Airflow tasks are expected to start, complete, and report a clear success or failure state.

---

## 20. Getting Started

### Prerequisites

- Python 3.10+
- Java 11+ (required by Spark)
- Apache Spark installed and on `PATH`
- Docker and Docker Compose
- Apache Airflow (optional — for DAG orchestration only)

### Installation

```bash
git clone https://github.com/KYoussefAI/amazon-reviews-streaming-pipeline.git
cd amazon-reviews-streaming-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

Download the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset from Kaggle and place it at:

```
data/raw/Reviews.csv
```

### Step-by-Step Startup

**Step 1 — Start infrastructure (Kafka, Zookeeper, MongoDB)**

```bash
cd kafka
docker compose up -d
docker compose ps   # verify kafka, zookeeper, and mongodb are running
```

**Step 2 — Train Spark ML models**

```bash
spark-submit src/spark/training/train_spark_pipeline.py
```

**Step 3 — Export the streaming test split**

```bash
python src/spark/training/export_test_split_for_streaming.py
# generates: data/processed/test_reviews.jsonl
```

**Step 4 — Start Spark Structured Streaming**

```bash
spark-submit src/spark/streaming/predict_stream.py
```

**Step 5 — Start the Kafka producer** *(separate terminal)*

```bash
python src/ingestion/producer.py
```

**Step 6 — Start the Flask dashboard** *(separate terminal)*

```bash
python src/web/app.py
# or: ./scripts/run_web_dashboard.sh
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

**Step 7 — (Optional) Run Airflow for batch orchestration**

```bash
airflow standalone             # opens http://localhost:8080
./airflow/scripts/test_dag.sh  # verify DAG import
```

Trigger the DAG `amazon_reviews_batch_orchestration` from the Airflow UI. All tasks should complete in green.

---

## 21. Runtime Commands Reference

### Inspect MongoDB predictions

```bash
docker exec -it mongodb mongosh
```

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
    confidence_available: 1,
    batch_id: 1
  }
).sort({ processed_at: -1 }).limit(5).pretty()
```

### Tune Spark pipeline hyperparameters

```bash
spark-submit src/spark/training/tune_spark_pipeline_sa.py
```

### Verify Kafka topics

```bash
docker exec -it kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --list
```

### Verify Airflow DAG import

```bash
./airflow/scripts/test_dag.sh
```

---

## 22. Example MongoDB Document

A complete prediction document written by the streaming pipeline:

```json
{
  "schema_version": "v2",
  "product_id": "B001E4KFG0",
  "review_date": "2011-04-27",
  "score": 5,
  "true_label": "positive",
  "prediction": 0.0,
  "predicted_label": "positive",
  "model_type": "one_vs_rest_linear_svc",
  "model_name": "one_vs_rest_linear_svc",
  "probability": [],
  "confidence": null,
  "confidence_status": "not_available",
  "confidence_available": false,
  "source_split": "test",
  "batch_id": 208,
  "source": "spark_structured_streaming",
  "processed_at": "ISODate(...)"
}
```

This document schema is designed for dashboard compatibility: it contains both the prediction result and all metadata needed for filtering, charting, and risk analysis.

---

## 23. Key Engineering Decisions

**Spark MLlib over scikit-learn** — The project focuses on distributed Big Data processing. Using Spark MLlib ensures the training pipeline scales horizontally and integrates natively with Spark Structured Streaming for consistent feature transformation at inference time.

**Kafka as streaming broker** — Kafka decouples the ingestion layer from the processing layer. The producer can be stopped, restarted, or replaced without affecting the streaming consumer. This matches real production streaming architectures.

**MongoDB for prediction storage** — MongoDB's flexible document model accommodates the evolving prediction schema (e.g., added confidence metadata in v2) without requiring schema migrations. Indexed fields keep dashboard queries fast.

**Flask over Streamlit for the final interface** — Streamlit was used as an initial prototype for rapid iteration. The final interface uses Flask with plain HTML/CSS/JavaScript to provide full control over layout, API design, filter behavior, and PDF export — requirements that Streamlit does not support cleanly.

**One-vs-Rest LinearSVC for streaming** — The Majority Vote Ensemble achieves the highest offline accuracy but requires loading three separate PipelineModels and combining their outputs at runtime. For a continuously running streaming job, this adds unnecessary overhead. `one_vs_rest_linear_svc` is the best single model and serializes as a standard Spark PipelineModel, making it straightforward to deploy in the streaming context.

**Honest N/A confidence** — `OneVsRest LinearSVC` does not expose calibrated class probabilities. Rather than assigning a surrogate score, the system explicitly marks confidence as `not_available`. The dashboard reflects this accurately. Fabricating a confidence metric would misrepresent the model's outputs.

---

## 24. Future Improvements

| Improvement | Description |
|---|---|
| Dockerize the dashboard | Single-command deployment of the Flask app |
| Authentication | Login layer for the web dashboard |
| Prometheus + Grafana | Real-time pipeline monitoring and alerting |
| Model drift detection | Automated monitoring of accuracy degradation over time |
| Scheduled retraining | Airflow DAG to retrain models on fresh data |
| REST API for on-demand scoring | Submit a review text and receive a real-time prediction |
| PostgreSQL / data warehouse | Long-term analytical storage beyond MongoDB |
| CI/CD with GitHub Actions | Automated testing and linting on every commit |
| Unit and integration tests | Coverage for data transformations, Kafka messages, and MongoDB writes |
| Cloud deployment | Deploy Kafka, Spark, MongoDB, and Flask to a cloud provider |

---

## License

This project is open source and available under the [MIT License](LICENSE).

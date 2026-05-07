# Amazon Reviews Streaming Sentiment Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?logo=apachespark&logoColor=white)
![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-231F20?logo=apachekafka&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?logo=mongodb&logoColor=white)
![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?logo=apacheairflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

An end-to-end Big Data engineering pipeline that trains Spark ML sentiment models on Amazon product reviews, streams predictions in real time through Kafka and Spark Structured Streaming, stores enriched results in MongoDB, and visualizes everything through a Flask/JavaScript web dashboard — all orchestrated with Apache Airflow.

---

## Architecture

```
Amazon Reviews CSV
        │
        ▼
Batch Preprocessing (PySpark)
        │
        ▼
Spark MLlib Training & Model Comparison
  ├── Logistic Regression
  ├── Naive Bayes
  ├── One-vs-Rest Linear SVC
  └── Majority Vote Ensemble  ──► Best Batch Model
        │
        ▼
Export Test Split → data/processed/test_reviews.jsonl
        │
        ▼
Kafka Producer  ──►  Topic: amazon_reviews
                              │
                              ▼
                  Spark Structured Streaming
                  (one_vs_rest_linear_svc)    ──► Best Streaming Model
                              │
                              ▼
                    MongoDB: sentiment_predictions
                              │
                              ▼
                  Flask / JavaScript Dashboard
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Distributed Processing | Apache Spark / PySpark |
| Machine Learning | Spark MLlib |
| Streaming Broker | Apache Kafka |
| Stream Processing | Spark Structured Streaming |
| Database | MongoDB |
| Web Dashboard | Flask, HTML/CSS, JavaScript |
| Orchestration | Apache Airflow |
| Containerization | Docker / Docker Compose |

---

## Features

- **Multi-model training** — Logistic Regression, Naive Bayes, One-vs-Rest Linear SVC, and a Majority Vote Ensemble are trained and compared side-by-side using Spark MLlib.
- **TF-IDF feature pipeline** — Text is processed through tokenization, stop-word removal, CountVectorizer, and IDF before model input.
- **Class imbalance handling** — Per-class weights are computed and injected during training to give minority classes (neutral, negative) appropriate influence.
- **Production model split** — The ensemble is used for offline evaluation; the lighter `one_vs_rest_linear_svc` is deployed for low-latency streaming.
- **Real-time streaming** — Kafka decouples ingestion from prediction. Spark Structured Streaming consumes micro-batches continuously and writes enriched documents to MongoDB.
- **Honest confidence reporting** — Since LinearSVC does not output calibrated probabilities, the dashboard correctly shows `N/A` rather than fabricating a confidence score.
- **Interactive dashboard** — KPIs, sentiment distributions, score analysis, risk monitoring, product-level drilldowns, and PDF export, all backed by live MongoDB queries.
- **Airflow orchestration** — A DAG handles the full batch side: structure validation, dataset checks, data export, model training, and model validation.

---

## Dashboard Screenshots

### Overview & Pipeline Status
![Dashboard Overview](docs/screenshots/flask_dashboard_overview.png)

### Main Analytics
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

## Repository Structure

```
.
├── airflow/
│   ├── dags/
│   │   └── amazon_reviews_batch_pipeline.py
│   └── scripts/
│       └── test_dag.sh
│
├── data/
│   ├── raw/           # Reviews.csv  (git-ignored — download separately)
│   └── processed/     # Generated artifacts (git-ignored)
│
├── docs/
│   ├── spark_sentiment_tuning_report.md
│   └── screenshots/
│
├── kafka/
│   └── docker-compose.yml
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
    │   └── producer.py                         # Kafka producer
    ├── spark/
    │   ├── training/
    │   │   ├── train_spark_pipeline.py
    │   │   ├── tune_spark_pipeline_sa.py
    │   │   ├── export_test_split_for_streaming.py
    │   │   └── models/                         # One file per model
    │   ├── streaming/
    │   │   └── predict_stream.py               # Spark Structured Streaming job
    │   └── model/
    │       └── ensemble_models/                # Saved Spark PipelineModels
    ├── storage/
    │   └── mongodb_writer.py
    └── web/
        ├── app.py                              # Flask backend
        ├── templates/dashboard.html
        └── static/
```

---

## Sentiment Labeling

Amazon review scores are mapped to a three-class label:

| Score | Sentiment |
|---|---|
| 1 – 2 | `negative` |
| 3 | `neutral` |
| 4 – 5 | `positive` |

This mapping is applied consistently across training, validation, testing, streaming, and the dashboard.

---

## Model Selection Rationale

| Context | Model | Reason |
|---|---|---|
| Offline / batch | `majority_vote_ensemble` | Highest accuracy — combines all three classifiers via majority voting |
| Production streaming | `one_vs_rest_linear_svc` | Best single `PipelineModel`; lower inference overhead suits continuous micro-batch processing |

---

## Getting Started

### Prerequisites

- Python 3.10+, Java 11+, Apache Spark, Docker, Apache Airflow
- Download the [Amazon Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and place it at `data/raw/Reviews.csv`

### 1 — Start infrastructure

```bash
cd kafka
docker compose up -d      # starts Kafka, Zookeeper, and MongoDB
```

### 2 — Train models

```bash
spark-submit src/spark/training/train_spark_pipeline.py
```

For cleaner output:

```bash
spark-submit src/spark/training/train_spark_pipeline.py 2>/dev/null \
  | grep -E "==========|Accuracy|Macro F1|Model saved"
```

### 3 — Export streaming data

```bash
python src/spark/training/export_test_split_for_streaming.py
# output: data/processed/test_reviews.jsonl
```

### 4 — Start Spark Structured Streaming

```bash
spark-submit src/spark/streaming/predict_stream.py
```

### 5 — Start Kafka producer

```bash
python src/ingestion/producer.py
```

### 6 — Open the dashboard

```bash
python src/web/app.py
# open http://localhost:5000
```

### Airflow (optional — batch orchestration)

```bash
airflow standalone            # opens http://localhost:8080
./airflow/scripts/test_dag.sh # verify DAG import
```

The DAG `amazon_reviews_batch_orchestration` runs steps 2–3 above in sequence with validation checkpoints.

---

## Flask API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard HTML |
| `GET /api/health` | MongoDB connection status |
| `GET /api/summary` | KPI metrics |
| `GET /api/charts` | Chart data |
| `GET /api/risk` | Suspicious and low-confidence samples |
| `GET /api/latest` | Latest prediction events |
| `GET /api/options` | Filter options (product, batch, model) |
| `GET /api/product/<id>` | Per-product sentiment analysis |
| `GET /api/report.pdf` | Downloadable PDF report |

---

## MongoDB Document Schema

Each prediction event written by the streaming job includes:

```json
{
  "schema_version": "v2",
  "product_id": "B001E4KFG0",
  "review_date": "2011-04-27",
  "score": 5,
  "true_label": "positive",
  "predicted_label": "positive",
  "model_type": "one_vs_rest_linear_svc",
  "confidence": null,
  "confidence_status": "not_available",
  "confidence_available": false,
  "batch_id": 208,
  "source": "spark_structured_streaming",
  "processed_at": "ISODate(...)"
}
```

---

## Future Improvements

- Dockerize the Flask dashboard for one-command deployment
- Add Prometheus / Grafana for pipeline monitoring
- Add model drift detection and scheduled Airflow retraining
- Expose a REST API for on-demand review scoring
- Add PostgreSQL / data warehouse layer for long-term analytics
- Add CI/CD with GitHub Actions
- Add unit and integration tests

---

## License

This project is open source and available under the [MIT License](LICENSE).

# Airflow Orchestration Layer — Amazon Reviews Big Data Project

## Purpose

This folder adds an optional Airflow orchestration layer to the Amazon Reviews Big Data project.

Airflow is used for the **batch preparation part** of the project:

```text
Check project structure
→ Check raw dataset
→ Export test split for streaming
→ Validate test_reviews.jsonl
→ Train Spark ML model
→ Validate saved model
```

Airflow does **not** replace Kafka or Spark Structured Streaming.

The real-time pipeline still runs separately:

```text
Kafka + MongoDB
→ Spark Structured Streaming
→ Kafka producer
→ Streamlit dashboard
```

## Where to paste this folder

Copy the whole `airflow/` folder into your project root:

```text
BIG DATA PROJECT/
├── airflow/
│   ├── dags/
│   │   └── amazon_reviews_batch_pipeline.py
│   ├── scripts/
│   │   ├── setup_airflow.sh
│   │   ├── start_airflow.sh
│   │   ├── stop_airflow.sh
│   │   └── test_dag.sh
│   ├── requirements-airflow.txt
│   └── README_AIRFLOW.md
```

Your final project should look like:

```text
BIG DATA PROJECT/
├── src/
├── kafka/
├── docs/
├── results/
├── airflow/
├── README.md
└── requirements.txt
```

## Setup

From the project root:

```bash
chmod +x airflow/scripts/*.sh
./airflow/scripts/setup_airflow.sh
```

This installs Airflow inside your current `.venv`.

## Start Airflow

From the project root:

```bash
./airflow/scripts/start_airflow.sh
```

Then open:

```text
http://localhost:8080
```

Default login:

```text
username: admin
password: admin
```

## Run the DAG

In the Airflow UI, enable and manually trigger:

```text
amazon_reviews_batch_orchestration
```

## Important environment variables

The DAG uses these defaults:

```bash
AMAZON_REVIEWS_PROJECT_ROOT="/mnt/c/Users/Me/Desktop/END TO END DATA ENGINEERING PROJECTS/BIG DATA PROJECT"
AMAZON_REVIEWS_VENV_PATH="$AMAZON_REVIEWS_PROJECT_ROOT/.venv"
```

If your project path changes, update these in:

```text
airflow/scripts/start_airflow.sh
airflow/scripts/test_dag.sh
```

or export them manually before starting Airflow.

## What Airflow orchestrates

| Task | Role |
|---|---|
| `check_project_structure` | Verifies required project files exist |
| `check_raw_dataset_exists` | Verifies `data/raw/Reviews.csv` exists |
| `export_test_split_for_streaming` | Runs `src/spark/training/export_test_split_for_streaming.py` |
| `validate_streaming_export` | Checks `test_reviews.jsonl`, `source_split`, and `B001E4KFG0` |
| `train_spark_model` | Runs `spark-submit src/spark/training/train_spark_pipeline.py` |
| `validate_saved_model` | Checks the saved Spark `PipelineModel` |
| `print_next_runtime_commands` | Prints next manual commands for the streaming pipeline |

## What Airflow does not orchestrate

Airflow should not run infinite streaming services directly:

```text
bd-kafka
bd-spark
bd-producer
bd-streamlit
```

Those are runtime services. Keep them separate for this portfolio version.

## Recommended README wording

Add this to your main README future/advanced section:

```text
An optional Airflow orchestration layer is included for batch preparation tasks such as test split export, model training, and validation. The real-time Kafka/Spark streaming services remain separate runtime services.
```

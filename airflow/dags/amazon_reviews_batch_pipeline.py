"""
Airflow DAG for the Amazon Reviews Big Data project.

Role:
    This DAG orchestrates the batch part of the project:
    - validate project files
    - validate raw dataset
    - export the test split used by Kafka streaming
    - validate the exported JSONL file
    - optionally train the Spark ML model
    - validate that the saved Spark PipelineModel exists

Important:
    This DAG does NOT replace Kafka or Spark Structured Streaming.
    Streaming services are still launched separately:
        bd-kafka
        bd-spark
        bd-producer
        bd-web

Recommended execution:
    Run this DAG manually from the Airflow UI when you want to prepare
    or refresh the batch artifacts used by the streaming pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator


DEFAULT_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

PROJECT_ROOT = os.environ.get("AMAZON_REVIEWS_PROJECT_ROOT", DEFAULT_PROJECT_ROOT)

VENV_PATH = os.environ.get(
    "AMAZON_REVIEWS_VENV_PATH",
    f"{PROJECT_ROOT}/.venv",
)

PYTHON_BIN = f"{VENV_PATH}/bin/python"
SPARK_SUBMIT_BIN = os.environ.get("SPARK_SUBMIT_BIN", "spark-submit")

RAW_DATA_PATH = f"{PROJECT_ROOT}/data/raw/Reviews.csv"
STREAMING_EXPORT_PATH = f"{PROJECT_ROOT}/data/processed/test_reviews.jsonl"
LEGACY_MODEL_PATH = f"{PROJECT_ROOT}/src/spark/model/sentiment_pipeline_model"
STREAMING_MODEL_PATH = (
    f"{PROJECT_ROOT}/src/spark/model/ensemble_models/one_vs_rest_linear_svc"
)

DEFAULT_BASH_PREFIX = f"""
set -e
cd "{PROJECT_ROOT}"
source "{VENV_PATH}/bin/activate"
export PYTHONPATH="{PROJECT_ROOT}:$PYTHONPATH"
"""


default_args = {
    "owner": "youssef",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="amazon_reviews_batch_orchestration",
    description="Batch orchestration for Amazon Reviews Kafka/Spark/MongoDB sentiment pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["big-data", "spark", "kafka", "mongodb", "portfolio"],
) as dag:

    start = EmptyOperator(
        task_id="start_batch_orchestration"
    )

    check_project_structure = BashOperator(
        task_id="check_project_structure",
        bash_command=f"""
        {DEFAULT_BASH_PREFIX}

        echo "Checking project structure..."

        test -f "src/spark/training/train_spark_pipeline.py"
        test -f "src/spark/training/export_test_split_for_streaming.py"
        test -f "src/ingestion/producer.py"
        test -f "src/spark/streaming/predict_stream.py"
        test -f "src/storage/mongodb_writer.py"
        test -f "src/web/app.py"
        test -f "kafka/docker-compose.yml"

        echo "Project structure check passed."
        """,
    )

    check_raw_dataset_exists = BashOperator(
        task_id="check_raw_dataset_exists",
        bash_command=f"""
        {DEFAULT_BASH_PREFIX}

        echo "Checking raw dataset..."
        test -f "{RAW_DATA_PATH}"

        echo "Raw dataset found:"
        ls -lh "{RAW_DATA_PATH}"
        """,
    )

    export_test_split_for_streaming = BashOperator(
        task_id="export_test_split_for_streaming",
        bash_command=f"""
        {DEFAULT_BASH_PREFIX}

        echo "Exporting test split for streaming..."
        "{PYTHON_BIN}" src/spark/training/export_test_split_for_streaming.py
        """,
        execution_timeout=timedelta(minutes=45),
    )

    validate_streaming_export = BashOperator(
        task_id="validate_streaming_export",
        bash_command=f"""
        {DEFAULT_BASH_PREFIX}

        echo "Validating streaming export file..."

        test -f "{STREAMING_EXPORT_PATH}"

        line_count=$(wc -l < "{STREAMING_EXPORT_PATH}")
        echo "Streaming export line count: $line_count"

        if [ "$line_count" -lt 1000 ]; then
            echo "ERROR: Streaming export seems too small."
            exit 1
        fi

        echo "Checking required ProductId B001E4KFG0..."
        grep -q '"product_id":"B001E4KFG0"' "{STREAMING_EXPORT_PATH}"

        echo "Checking source_split field..."
        grep -q '"source_split"' "{STREAMING_EXPORT_PATH}"

        echo "Streaming export validation passed."
        head -n 3 "{STREAMING_EXPORT_PATH}"
        """,
    )

    train_spark_model = BashOperator(
        task_id="train_spark_model",
        bash_command=f"""
        {DEFAULT_BASH_PREFIX}

        echo "Training Spark ML model..."
        "{SPARK_SUBMIT_BIN}" src/spark/training/train_spark_pipeline.py
        """,
        execution_timeout=timedelta(hours=2),
    )

    train_ensemble_models = BashOperator(
        task_id="train_ensemble_models",
        bash_command=f"""
        {DEFAULT_BASH_PREFIX}

        echo "Training ensemble-compatible streaming models..."
        "{SPARK_SUBMIT_BIN}" src/spark/training/train_ensemble_models.py
        """,
        execution_timeout=timedelta(hours=2),
    )

    validate_saved_models = BashOperator(
        task_id="validate_saved_models",
        bash_command=f"""
        {DEFAULT_BASH_PREFIX}

        echo "Validating saved Spark model artifacts..."

        test -d "{LEGACY_MODEL_PATH}"
        test -f "{LEGACY_MODEL_PATH}/metadata/part-00000" || test -f "{LEGACY_MODEL_PATH}/metadata/_SUCCESS"

        test -d "{STREAMING_MODEL_PATH}"
        test -f "{STREAMING_MODEL_PATH}/metadata/part-00000" || test -f "{STREAMING_MODEL_PATH}/metadata/_SUCCESS"

        echo "Saved batch model found:"
        find "{LEGACY_MODEL_PATH}" -maxdepth 2 -type f | head -n 20

        echo "Saved streaming model found:"
        find "{STREAMING_MODEL_PATH}" -maxdepth 2 -type f | head -n 20
        """,
    )

    print_next_runtime_commands = BashOperator(
        task_id="print_next_runtime_commands",
        bash_command=f"""
        {DEFAULT_BASH_PREFIX}

        echo "Batch orchestration completed."
        echo ""
        echo "Next runtime commands:"
        echo "Terminal 1: bd-kafka"
        echo "Terminal 2: bd-spark"
        echo "Terminal 3: bd-producer"
        echo "Terminal 4: bd-web"
        """,
    )

    end = EmptyOperator(
        task_id="end_batch_orchestration"
    )

    start >> check_project_structure >> check_raw_dataset_exists
    check_raw_dataset_exists >> export_test_split_for_streaming >> validate_streaming_export
    validate_streaming_export >> train_spark_model >> train_ensemble_models >> validate_saved_models
    validate_saved_models >> print_next_runtime_commands >> end

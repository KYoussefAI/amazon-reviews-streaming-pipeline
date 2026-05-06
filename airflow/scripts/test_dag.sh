#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

export AIRFLOW_HOME="$PROJECT_ROOT/airflow/airflow_home"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False

export AMAZON_REVIEWS_PROJECT_ROOT="$PROJECT_ROOT"
export AMAZON_REVIEWS_VENV_PATH="$PROJECT_ROOT/.venv"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Listing DAGs..."
airflow dags list | grep amazon_reviews_batch_orchestration

echo "Testing DAG import..."
python -m py_compile airflow/dags/amazon_reviews_batch_pipeline.py

echo "DAG test passed."

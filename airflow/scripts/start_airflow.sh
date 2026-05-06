#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

export AIRFLOW_HOME="$PROJECT_ROOT/airflow/airflow_home"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True

export AMAZON_REVIEWS_PROJECT_ROOT="$PROJECT_ROOT"
export AMAZON_REVIEWS_VENV_PATH="$PROJECT_ROOT/.venv"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$AIRFLOW_HOME/logs"

echo "Starting Airflow scheduler..."
airflow scheduler > "$AIRFLOW_HOME/logs/scheduler.log" 2>&1 &
SCHEDULER_PID=$!

echo "$SCHEDULER_PID" > "$AIRFLOW_HOME/scheduler.pid"

echo "Starting Airflow webserver on http://localhost:8080 ..."
airflow webserver --port 8080

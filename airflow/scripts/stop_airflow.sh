#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AIRFLOW_HOME="$PROJECT_ROOT/airflow/airflow_home"

if [ -f "$AIRFLOW_HOME/scheduler.pid" ]; then
  SCHEDULER_PID="$(cat "$AIRFLOW_HOME/scheduler.pid")"
  echo "Stopping Airflow scheduler PID: $SCHEDULER_PID"
  kill "$SCHEDULER_PID" || true
  rm -f "$AIRFLOW_HOME/scheduler.pid"
fi

echo "Stopping possible Airflow webserver processes..."
pkill -f "airflow webserver" || true

echo "Airflow stopped."

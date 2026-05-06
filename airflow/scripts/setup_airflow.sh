#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  echo "ERROR: .venv not found in project root."
  echo "Create/activate your Python 3.10 virtual environment first."
  exit 1
fi

source .venv/bin/activate

export AIRFLOW_HOME="$PROJECT_ROOT/airflow/airflow_home"
export AIRFLOW_VERSION="2.10.5"
export PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

echo "Project root: $PROJECT_ROOT"
echo "Airflow home: $AIRFLOW_HOME"
echo "Python version: $PYTHON_VERSION"
echo "Installing Airflow..."

pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

echo "Initializing Airflow database..."
airflow db migrate

echo "Creating admin user..."
airflow users create \
  --username admin \
  --firstname Youssef \
  --lastname Admin \
  --role Admin \
  --email admin@example.com \
  --password admin || true

echo "Airflow setup complete."
echo "Start Airflow with:"
echo "./airflow/scripts/start_airflow.sh"

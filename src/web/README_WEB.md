# Professional Flask Web Dashboard — Amazon Reviews Big Data Project

## Purpose

This package provides the project's professional Flask / JavaScript web interface.

It is designed to align with the project PDF requirement for a web solution using:

```text
Django / Flask / JavaScript
```

The dashboard reads MongoDB prediction documents produced by:

```text
Producer → Kafka → Spark Structured Streaming → Spark ML model → MongoDB
```

## Folder Structure

Paste this package at the project root:

```text
src/web/
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

scripts/
└── run_web_dashboard.sh
```

## Install Requirements

From the project root:

```bash
bigdata
pip install -r src/web/requirements-web.txt
```

## Run

```bash
python src/web/app.py
```

or:

```bash
chmod +x scripts/run_web_dashboard.sh
./scripts/run_web_dashboard.sh
```

Open:

```text
http://localhost:5000
```

## Implemented Dashboard Sections

### 1. Pipeline Status

- MongoDB connection status
- Total predictions
- Latest batch ID
- Streamed accuracy
- Latest processed timestamp

### 2. Prediction Summary

- Positive predictions
- Negative predictions
- Neutral predictions
- Average confidence
- Confidence coverage
- Low-confidence count

### 3. Main Analytics

- Sentiment distribution
- Model type distribution
- Amazon score distribution
- Score by predicted sentiment

### 4. Date and Source Analytics

- Prediction results by review date
- Source split distribution

### 5. Model Quality Analytics

- Confidence distribution
- Average confidence by sentiment
- Average confidence by model
- True vs predicted confusion matrix

### 6. Streaming Operations

- Records per micro-batch
- Batch summary table

### 7. Required ProductId Analysis

Dedicated section for:

```text
B001E4KFG0
```

Includes exact event metrics if one prediction exists, and aggregate product metrics if multiple predictions exist.

### 8. Risk Monitoring

- Suspicious score/sentiment combinations
- Low-confidence prediction samples

### 9. Latest Prediction Events

Full table of enriched MongoDB prediction documents.

### 10. PDF Report

Generates a downloadable PDF report from the active filters.

## Confidence Note

If the active model is:

```text
one_vs_rest_linear_svc
```

confidence may be displayed as:

```text
N/A
```

This is correct because LinearSVC does not output calibrated probabilities by default.

## Suggested Alias

Add this to `~/.bashrc`:

```bash
alias bd-web='cd "$BIGDATA_PROJECT" && source "$BIGDATA_ENV/bin/activate" && export PYTHONPATH=$PWD && python src/web/app.py'
```

Reload:

```bash
source ~/.bashrc
```

Run:

```bash
bd-web
```

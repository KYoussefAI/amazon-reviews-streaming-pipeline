# Amazon Reviews Streaming Pipeline

A Big Data project that builds a real-time sentiment analysis pipeline using Kafka, Spark, MongoDB, and dashboards.

The project is developed phase by phase with a focus on understanding, correctness, and production readiness.

---

## Project Objective

Build a complete real-time data pipeline:

```text
Amazon Reviews Dataset
→ Kafka
→ Spark
→ MongoDB
→ Dashboard
```

The goal is not only to build the pipeline, but also to understand how each component works in a real-world data engineering system.

---

## Fixed Architecture

```text
Producer → Kafka → Spark → MongoDB → Dashboard
```

Each component has a clear responsibility:

| Component | Role                                                    |
| --------- | ------------------------------------------------------- |
| Producer  | Reads Amazon reviews and sends them as events           |
| Kafka     | Handles real-time ingestion and event streaming         |
| Spark     | Processes data and applies machine learning predictions |
| MongoDB   | Stores predictions and processed results                |
| Dashboard | Visualizes real-time and offline analytics              |

---

## Dataset

The project uses the Amazon Fine Food Reviews dataset.

Main fields used:

| Field | Description        |
| ----- | ------------------ |
| Text  | Review text        |
| Score | Rating from 1 to 5 |
| Time  | Review timestamp   |

Sentiment labels are created using this rule:

| Score     | Label    |
| --------- | -------- |
| Score < 3 | negative |
| Score = 3 | neutral  |
| Score > 3 | positive |

---

## Current Project Status

This repository has completed Phases 1 to 7 and is now focused only on Phase 8.

```text
Current chat scope: Phase 8 only
Phase 8 goal: Kafka → Spark Streaming → Prediction
```

---

# Completed Work So Far

## Phase 1 — Kafka Streaming

### What was built

* Kafka and Zookeeper setup using Docker
* Kafka topic: `amazon_reviews`
* Producer that streams Amazon reviews from `Reviews.csv`
* Basic consumer that reads and displays messages

### Producer behavior

The producer reads review rows and sends JSON messages to Kafka:

```json
{
  "text": "review text here",
  "score": 5
}
```

A delay is used to simulate real-time streaming.

### Key concepts learned

* Kafka is an event streaming system
* Producers send events continuously
* Kafka decouples ingestion from processing
* Topics organize streams of events

---

## Phase 2 — Text Preprocessing

### What was built

A text cleaning pipeline that performs:

* Lowercasing
* Regex cleaning
* Stopwords removal
* Lemmatization

### Main function

```python
clean_text(text) -> cleaned string
```

### Problem solved

Raw review text contains noise such as punctuation, numbers, uppercase characters, and useless words. Cleaning makes the text easier to use for machine learning.

---

## Phase 2 Part 2 — TF-IDF Vectorization

### What was built

A TF-IDF vectorizer using `sklearn`.

### Functions

```python
fit_transform(texts)
transform(texts)
```

### Important lesson

TF-IDF must be fitted only on the training data.

Correct approach:

```text
Split data first
Fit TF-IDF on train only
Transform validation and test
```

Incorrect approach:

```text
Fit TF-IDF before splitting
```

Why this is wrong:

* It causes data leakage
* Test data influences training vocabulary and IDF values
* Evaluation becomes biased

---

## Phase 3 — Dataset Creation

### What was built

A dataset preparation module that:

1. Loads `Reviews.csv`
2. Selects useful columns
3. Cleans text
4. Converts scores to labels
5. Splits the dataset into train, validation, and test sets

### Split strategy

```text
Train: 80%
Validation: 10%
Test: 10%
```

Stratification is used to preserve class distribution across splits.

### Output

```python
x_train, x_val, x_test, y_train, y_val, y_test
```

---

## Phase 4 — Training

### What was built

A Spark ML training pipeline using Logistic Regression.

### Training steps

1. Load the prepared dataset
2. Apply TF-IDF correctly
3. Convert scipy sparse vectors to Spark SparseVectors
4. Create Spark DataFrames
5. Convert string labels with `StringIndexer`
6. Train Spark Logistic Regression

### Model used

```text
Spark ML LogisticRegression
```

### Important technical point

The initial pipeline was hybrid:

```text
Python / sklearn TF-IDF → Spark Logistic Regression
```

This worked for experimentation but is not ideal for streaming production.

---

## Phase 5 — Validation

### Metrics implemented

* Accuracy
* Confusion matrix
* Precision per class
* Recall per class
* F1-score per class

### Main finding

The model was biased toward the positive class because the dataset is imbalanced.

Class imbalance problem:

```text
positive → dominant
negative → smaller
neutral → very rare
```

---

## Phase 5 Part 2 — Resampling Experiments

Several strategies were tested.

### No resampling

Problem:

* Strong positive-class bias

### Full oversampling

Problem:

* Overfitting minority classes
* Majority class performance decreased

### Undersampling

Problem:

* Loss of useful data
* Lower overall accuracy

### Partial oversampling around 70–75%

Result:

* Best trade-off
* Improved minority-class performance
* Preserved majority-class performance

Final decision:

```text
Use partial oversampling only on training data
```

Validation and test data must remain untouched.

---

## Phase 6 — Testing

### Important rule

The test set was used only once after tuning.

### Final test behavior

The final model achieved stable validation and test results.

Approximate final result:

```text
Test accuracy ≈ 0.796
```

### Key conclusion

Validation and test performance were close.

This means:

```text
The model generalizes reasonably well
No strong overfitting was detected
```

---

## Phase 7 — Model Selection

### Selected model

```text
Logistic Regression
```

### Selected feature strategy

```text
TF-IDF
```

### Selected imbalance strategy

```text
Partial oversampling around 70–75% on training data only
```

### Why this model was selected

* Simple
* Stable
* Interpretable
* Good baseline for production
* Validation and test results are close
* Works well enough before moving to streaming inference

---

# Current Technical Issue

The current working model started as a hybrid system:

```text
Python / sklearn preprocessing
→ manual vector conversion
→ Spark ML model
```

This is acceptable for experimentation, but not ideal for production streaming.

Problems:

* Not fully scalable
* Extra conversion overhead
* Harder to use directly with Spark Structured Streaming
* More fragile in a real-time pipeline

---

# Strategic Decision

The project now moves toward a Spark-only production pipeline.

Old experimental code will remain available, but production code will be based on Spark.

```text
experiments/ = learning and experimentation layer
spark/       = production Spark pipeline
```

---

## Recommended Project Structure

```text
BIG-DATA-PROJECT/
│
├── src/
│   ├── ingestion/
│   │   └── producer.py
│   │
│   ├── experiments/
│   │   ├── preprocessing/
│   │   │   ├── clean.py
│   │   │   ├── vectorizer.py
│   │   │   ├── label.py
│   │   │   ├── dataset.py
│   │   │   └── resampling.py
│   │   │
│   │   └── training/
│   │       └── train.py
│   │
│   ├── spark/
│   │   ├── training/
│   │   │   └── train_spark_pipeline.py
│   │   │
│   │   ├── streaming/
│   │   │   └── predict_stream.py
│   │   │
│   │   └── models/
│   │       └── sentiment_pipeline_model/
│   │
│   ├── storage/
│   │   └── mongodb_writer.py
│   │
│   └── utils/
│
├── data/
│   └── raw/
│       └── Reviews.csv
│
├── kafka/
│   └── docker-compose.yml
│
├── orchestration/
│   └── dags/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Current Phase — Phase 8 Only

This chat is dedicated to Phase 8 only.

## Phase 8 Name

```text
Real-Time Pipeline Online
```

## Phase 8 Goal

Build real-time inference:

```text
Kafka → Spark Structured Streaming → Load saved model → Predict sentiment
```

## Phase 8 Inputs

Kafka topic:

```text
amazon_reviews
```

Incoming Kafka message:

```json
{
  "text": "review text here",
  "score": 5
}
```

## Phase 8 Output

A prediction result such as:

```json
{
  "text": "review text here",
  "score": 5,
  "prediction": "positive"
}
```

At first, predictions can be printed to the console.

MongoDB storage is Phase 9, so it should not be added yet.

---

# Phase 8 Scope

Phase 8 includes:

* Reading Kafka messages using Spark Structured Streaming
* Parsing JSON messages
* Applying the saved Spark model
* Producing sentiment predictions
* Displaying predictions in the console
* Keeping the solution minimal and working

Phase 8 does not include:

* MongoDB storage
* Dashboard
* Airflow orchestration
* Advanced deployment
* Model optimization
* New tools outside the project architecture

---

# Phase 8 Completion Checklist

Phase 8 will be considered finished when all of the following are true:

* Kafka is running
* Topic `amazon_reviews` exists
* Producer sends review messages successfully
* Spark streaming job reads from Kafka
* JSON messages are parsed correctly
* Saved model is loaded successfully
* Predictions are generated in real time
* Predictions are printed to console
* The pipeline runs end-to-end without crashing

Final Phase 8 target:

```text
Producer → Kafka → Spark Streaming → Prediction shown in console
```

Once this checklist is completed, Phase 8 is finished and the project can move to Phase 9.

---

# Next Phase After This Chat

## Phase 9 — Storage

After Phase 8 is complete, the next phase will be:

```text
Spark Streaming predictions → MongoDB
```

Phase 9 will focus on storing prediction results.

---

# Engineering Rules

* Do not restart the project
* Do not skip phases
* Do not introduce new tools outside the fixed architecture
* Do not mix validation and test sets
* Do not use the test set for tuning
* Keep old Python code as experimentation history
* Use Spark as the production path
* Build minimal working solutions before optimization

---

# Useful Commands

Start Kafka services:

```bash
cd kafka
docker-compose up -d
```

Run the producer:

```bash
python -m src.ingestion.producer
```

Run Spark training pipeline:

```bash
spark-submit src/spark/training/train_spark_pipeline.py
```

Run Spark streaming prediction pipeline:

```bash
spark-submit src/spark/streaming/predict_stream.py
```

---

# Current Status Summary

```text
Phase 1: Kafka ingestion                  DONE
Phase 2: Preprocessing                    DONE
Phase 3: Dataset creation                 DONE
Phase 4: Training                         DONE
Phase 5: Validation and tuning            DONE
Phase 6: Testing                          DONE
Phase 7: Model selection                  DONE
Phase 8: Real-time streaming prediction   IN PROGRESS
Phase 9: MongoDB storage                  NOT STARTED
Phase 10: Offline dashboard               NOT STARTED
Phase 11: Real-time dashboard             NOT STARTED
Phase 12: Finalization                    NOT STARTED
```

---

# Final Note

This repository is now moving from experimentation toward production-style streaming inference.

The current focus is only:

```text
Phase 8: Kafka → Spark Streaming → Prediction
```

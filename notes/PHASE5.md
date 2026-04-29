# PHASE 5 — VALIDATION

---

## 1. Objective
- Evaluate model performance after training
- Move from “model runs” → “model is measurable and understood”
- Identify strengths, weaknesses, and failure patterns

---

## 2. System Position

Kafka → Preprocessing → Training (Spark ML) → Validation (YOU ARE HERE) → Testing → Deployment

---

## 3. What Was Implemented

### Key Components
- Global evaluation metrics
- Confusion matrix
- Per-class metrics computation
- Model behavior analysis

### Files Involved
- train.py → training + prediction + evaluation
- Spark ML:
  - LogisticRegression
  - StringIndexer

### Important Logic
- model.transform(test_df)
- groupBy(label_index, prediction).count()
- TP / actual_total / pred_total extraction

---

## 4. Step-by-Step Pipeline

1. Load dataset (TF-IDF + labels)
2. Convert → Spark SparseVector
3. Create DataFrame (features, label)
4. Split into train / test (used as validation)
5. Apply StringIndexer → label_index
6. Train Logistic Regression
7. Generate predictions
8. Compute global metrics
9. Build confusion matrix
10. Extract TP / row sums / column sums
11. Compute per-class metrics
12. Analyze model behavior

---

## 5. Key Code Patterns (IMPORTANT)

- DataFrame → transformation → new DataFrame
- model.transform() adds prediction column
- groupBy + aggregation
- confusion matrix = groupBy(actual, predicted)
- join multiple aggregates to compute metrics
- Spark logic ≈ SQL logic

---

## 6. Core Concepts Learned

- Evaluation > just accuracy
- Precision = trust in predictions
- Recall = coverage of real cases
- F1 = balance
- Confusion matrix = error distribution
- Per-class metrics = true diagnosis
- Spark DataFrames are immutable

---

## 7. Metrics / Results

- Accuracy ≈ 0.75
- F1 ≈ 0.76
- Precision > Recall

Per-class:
- Positive → strong
- Negative → medium
- Neutral → weak

Meaning:
- Model biased toward dominant class
- Neutral poorly learned

---

## 8. Model / System Behavior (CRITICAL)

Worked:
- Strong positive detection
- Stable baseline

Failed:
- Neutral classification
- Subtle sentiment separation

Weaknesses:
- Class imbalance
- TF-IDF limitations
- Linear model limits

---

## 9. Mistakes & Pitfalls

Conceptual:
- Confusion between precision and recall
- Overtrusting accuracy

Implementation:
- Using test_df as validation
- No real test set

Critical:
- TF-IDF before split → DATA LEAKAGE

---

## 10. Confusions & Questions Raised

- Why predictions keeps original columns?
- Precision vs recall difference
- Why accuracy is misleading
- How confusion matrix works
- Why joins are needed
- Does more data fix everything?
- Why SQL not used yet?
- Offline vs real-time dashboard?
- Does Python disappear in Spark?

---

## 11. Answers & Clarifications

- transform() augments DataFrame
- Precision = correctness of predictions
- Recall = coverage of real data
- Accuracy fails with imbalance
- Confusion matrix shows exact errors
- Joins combine needed aggregates
- More data helps but doesn't fix model limits
- SQL = querying, not processing
- Offline first = validation
- Python remains orchestration layer

---

## 12. Engineering Insights

- Evaluation = diagnosis, not just numbers
- Always inspect per-class performance
- Data leakage silently breaks systems
- Baselines are expected to be imperfect
- Understanding > metrics
- Spark thinking = SQL thinking at scale

---

## 13. Rebuild From Memory

1. Load dataset
2. Convert to Spark vectors
3. Create DataFrame
4. Split data
5. Index labels
6. Train model
7. Predict
8. Compute metrics
9. Build confusion matrix
10. Compute per-class metrics
11. Analyze

---

## 14. What Comes Next

PHASE 6 — TESTING

- Introduce real test set
- Fix data leakage
- Split: train / validation / test
- Final unbiased evaluation

Challenges:
- strict separation
- no leakage
- correct pipeline rebuild

---

## 15. One-Page Mental Summary

Goal:
Evaluate model deeply

Steps:
Train → Predict → Metrics → Confusion → Per-class

Insight:
Model biased toward positive

Main mistake:
TF-IDF before split → leakage
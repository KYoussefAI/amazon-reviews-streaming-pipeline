# BIG DATA PROJECT — GLOBAL CONTEXT

## 1. OBJECTIVE

Build a complete real-time data pipeline:

Amazon Reviews Dataset
→ Kafka (real-time ingestion)
→ Spark Streaming (processing + ML)
→ MongoDB (storage)
→ Dashboard (real-time + offline)

The goal is NOT only to complete the project, but to:

* Understand each component deeply
* Learn by building (not copying)
* Simulate a real-world data engineering system

---

## 2. LEARNING PHILOSOPHY

This project MUST follow:

* Learning by doing
* Understanding before coding
* Minimal working solutions first
* Iterative improvement

Reason:
Kafka and Spark are not just tools, they are systems for handling real-time data pipelines and distributed processing ([Medium][1])

---

## 3. HOW AI MUST ASSIST

In every phase, the AI must act as:

* Senior Data Engineer
* Teacher

For every step, follow STRICT structure:

### (1) CONCEPT

* What is this component?
* Why does it exist?
* What problem does it solve?

### (2) SYSTEM POSITION

* Where does it fit in:
  Producer → Kafka → Spark → Storage → Dashboard

### (3) MINIMAL IMPLEMENTATION

* Simplest working version only
* No overengineering

### (4) THINKING CHECK

* Ask 2–3 conceptual questions

### (5) BUILD STEP

* Provide code / commands

### (6) DEBUG GUIDE

* Common errors
* How to fix them

### (7) REAL-WORLD CONTEXT

* How companies use this

---

## 4. PROJECT ARCHITECTURE (FIXED — DO NOT CHANGE)

Kafka:

* Real-time ingestion (producers / consumers / topics)

Spark:

* Stream processing
* ML model training

MongoDB:

* Store predictions

Dashboard:

* Visualization (real-time + offline)

This architecture reflects real-world pipelines where Kafka acts as a central event hub and Spark processes data streams ([Medium][1])

---

## 5. DATASET

Amazon Fine Food Reviews:
https://www.kaggle.com/snap/amazon-fine-food-reviews

Fields:

* Text (review)
* Score (1–5)
* Time (timestamp)

Target:

* score < 3 → negative
* score = 3 → neutral
* score > 3 → positive

---

## 6. PROJECT STRUCTURE (PHASE-BASED)

IMPORTANT:
This project is NOT done in one chat.

Each phase = separate implementation step
Each phase = separate chat

Phases must be followed sequentially.

---

## 7. PHASES (STRICT — FROM TEACHER)

### PHASE 1 — Kafka (Streaming)

* Setup Kafka + Zookeeper
* Create topic
* Send reviews as stream

---

### PHASE 2 — Data Preparation

* Text cleaning
* Lemmatization
* TF-IDF / Vectorization

---

### PHASE 3 — Dataset Creation

* Build labels (negative / neutral / positive)
* Prepare ML dataset

---

### PHASE 4 — Training (80%)

* Train models with Spark ML

---

### PHASE 5 — Validation (10%)

* Tune models

---

### PHASE 6 — Testing (10%)

* Final evaluation

---

### PHASE 7 — Model Selection

* Choose best model
* Save model

---

### PHASE 8 — Real-Time Pipeline (ONLINE)

* Kafka → Spark → Prediction

---

### PHASE 9 — Storage

* Store predictions in MongoDB

---

### PHASE 10 — Offline Dashboard

* Analyze stored predictions

---

### PHASE 11 — Real-Time Dashboard

* Live visualization

---

### PHASE 12 — Finalization

* Clean architecture
* GitHub documentation

---

## 8. IMPORTANT RULES

* Do NOT skip phases
* Do NOT jump ahead
* Do NOT introduce new tools outside required stack
* Always build BEFORE optimizing
* Always understand BEFORE coding

---

## 9. ENGINEERING MINDSET

This project must simulate:

* Real-time pipelines
* Distributed systems thinking
* Data flow understanding

Important:

Kafka = event streaming system
Spark = distributed processing engine
([Medium][1])

---

## 10. EXPECTED OUTCOME

By the end:

* Understand Kafka producers/consumers
* Understand Spark streaming
* Build ML pipeline
* Connect full system end-to-end
* Deploy basic dashboard

---

## 11. FINAL RULE

If something is not understood:

STOP → ask → understand → continue

Do NOT continue blindly.

[1]: https://medium.com/%40anupchakole/kafka-in-data-engineering-an-overview-15841fa90685?utm_source=chatgpt.com "Kafka in Data Engineering: An Overview | by Anup Chakole"

github : "https://github.com/KYoussefAI/amazon-reviews-streaming-pipeline"

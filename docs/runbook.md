# Runbook

This project has four main runtime surfaces: infrastructure startup, batch training, online streaming, and dashboard verification.

## Infrastructure

Start Kafka, Zookeeper, and MongoDB with:

```bash
docker compose -f kafka/docker-compose.yml up -d
```

Create the Kafka topic if it does not already exist:

```bash
docker exec -it kafka1 kafka-topics \
  --bootstrap-server kafka1:29092,kafka2:29093,kafka3:29094 \
  --create \
  --if-not-exists \
  --topic amazon_reviews \
  --partitions 6 \
  --replication-factor 3
```

## Batch Steps

Export the held-out test split used for streaming simulation:

```bash
python src/spark/training/export_test_split_for_streaming.py
```

Train and compare Spark models:

```bash
spark-submit src/spark/training/train_spark_pipeline.py
spark-submit src/spark/training/compare_spark_models.py
```

## Streaming Steps

Run the Spark streaming scorer:

```bash
spark-submit src/spark/streaming/predict_stream.py
```

Replay exported test reviews into Kafka:

```bash
python src/ingestion/producer.py --sleep-seconds 0.2
```

## Dashboard and Verification

Start the Flask dashboard:

```bash
python src/web/app.py
```

Open `http://localhost:5000` and verify recent documents in MongoDB:

```bash
docker exec -it mongodb mongosh
use amazon_reviews_db
db.sentiment_predictions.find().sort({ processed_at: -1 }).limit(5)
```

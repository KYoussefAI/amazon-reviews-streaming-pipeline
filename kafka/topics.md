# Kafka Topics

The local Docker Compose stack runs three Kafka brokers:

```text
localhost:9092
localhost:9093
localhost:9094
```

Create the production-style review topic with replication:

```bash
docker exec -it kafka1 kafka-topics.sh \
  --bootstrap-server kafka1:29092,kafka2:29093,kafka3:29094 \
  --create \
  --if-not-exists \
  --topic amazon_reviews \
  --partitions 6 \
  --replication-factor 3 \
  --config min.insync.replicas=2
```

Describe the topic:

```bash
docker exec -it kafka1 kafka-topics.sh \
  --bootstrap-server kafka1:29092,kafka2:29093,kafka3:29094 \
  --describe \
  --topic amazon_reviews
```

Local Python and Spark clients use this default bootstrap string:

```text
localhost:9092,localhost:9093,localhost:9094
```

Override it for another environment:

```bash
export KAFKA_BOOTSTRAP_SERVERS="broker-a:9092,broker-b:9092,broker-c:9092"
```

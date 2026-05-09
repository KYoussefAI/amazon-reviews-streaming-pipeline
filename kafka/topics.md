# Kafka Topic Commands

The local production-style stack uses three Kafka brokers:

```text
Host bootstrap servers:
localhost:9092,localhost:9093,localhost:9094

Container bootstrap servers:
kafka1:29092,kafka2:29093,kafka3:29094
```

## List Topics

```bash
docker exec -it kafka1 kafka-topics \
  --bootstrap-server kafka1:29092,kafka2:29093,kafka3:29094 \
  --list
```

## Create `amazon_reviews`

Create the topic with 6 partitions and replication factor 3:

```bash
docker exec -it kafka1 kafka-topics \
  --bootstrap-server kafka1:29092,kafka2:29093,kafka3:29094 \
  --create \
  --if-not-exists \
  --topic amazon_reviews \
  --partitions 6 \
  --replication-factor 3
```

## Describe Topic

```bash
docker exec -it kafka1 kafka-topics \
  --bootstrap-server kafka1:29092,kafka2:29093,kafka3:29094 \
  --describe \
  --topic amazon_reviews
```

Expected production-style properties:

```text
PartitionCount: 6
ReplicationFactor: 3
```

## Configure Minimum In-Sync Replicas

```bash
docker exec -it kafka1 kafka-configs \
  --bootstrap-server kafka1:29092,kafka2:29093,kafka3:29094 \
  --entity-type topics \
  --entity-name amazon_reviews \
  --alter \
  --add-config min.insync.replicas=2
```

## Consume Messages for Debugging

```bash
docker exec -it kafka1 kafka-console-consumer \
  --bootstrap-server kafka1:29092,kafka2:29093,kafka3:29094 \
  --topic amazon_reviews \
  --from-beginning \
  --max-messages 5
```

## Client Defaults

Python and Spark clients default to:

```bash
export KAFKA_BOOTSTRAP_SERVERS="localhost:9092,localhost:9093,localhost:9094"
export KAFKA_TOPIC="amazon_reviews"
```

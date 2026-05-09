from kafka import KafkaConsumer
import json
import os
from src.experiments.preprocessing.clean import clean_text

consumer = KafkaConsumer(
    os.environ.get("KAFKA_TOPIC", "amazon_reviews"),
    bootstrap_servers=os.environ.get(
        "KAFKA_BOOTSTRAP_SERVERS",
        "localhost:9092,localhost:9093,localhost:9094",
    ),
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for message in consumer:
    data = message.value

    cleaned = clean_text(data["text"])

    print({
        "clean_text": cleaned,
        "score": data["score"]
    })

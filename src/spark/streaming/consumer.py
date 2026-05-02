from kafka import KafkaConsumer
import json
from src.experiments.preprocessing.clean import clean_text

consumer = KafkaConsumer(
    'amazon_reviews',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for message in consumer:
    data = message.value

    cleaned = clean_text(data["text"])

    print({
        "clean_text": cleaned,
        "score": data["score"]
    })

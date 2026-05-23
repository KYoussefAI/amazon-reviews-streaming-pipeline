import json
import logging

from kafka import KafkaConsumer

from src.config import KafkaSettings
from src.experiments.preprocessing.clean import clean_text


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = KafkaSettings()

consumer = KafkaConsumer(
    settings.topic,
    bootstrap_servers=settings.bootstrap_servers,
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
)

for message in consumer:
    data = message.value
    cleaned = clean_text(data["text"])
    logger.info("Consumed score=%s clean_text=%s", data["score"], cleaned)

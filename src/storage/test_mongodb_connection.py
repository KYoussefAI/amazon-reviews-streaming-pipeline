import logging
from datetime import datetime, timezone

from pymongo import MongoClient

from src.config import MongoSettings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    settings = MongoSettings()
    client = MongoClient(settings.uri)

    db = client[settings.database]
    collection = db[settings.collection]

    document = {
        "text_preview": "python mongodb connection test",
        "score": 5,
        "prediction": 0.0,
        "predicted_label": "positive",
        "probability": [0.98, 0.01, 0.01],
        "processed_at": datetime.now(timezone.utc),
        "source": "python_test",
    }

    result = collection.insert_one(document)

    logger.info("Inserted document id: %s", result.inserted_id)

    inserted_document = collection.find_one({"_id": result.inserted_id})
    logger.info("Inserted document: %s", inserted_document)

    client.close()


if __name__ == "__main__":
    main()

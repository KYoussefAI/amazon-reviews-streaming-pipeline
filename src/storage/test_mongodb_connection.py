from datetime import datetime, timezone

from pymongo import MongoClient


MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "amazon_reviews_db"
COLLECTION_NAME = "sentiment_predictions"


def main():
    client = MongoClient(MONGO_URI)

    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

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

    print("========== MONGODB PYTHON TEST ==========")
    print(f"Inserted document id: {result.inserted_id}")

    inserted_document = collection.find_one({"_id": result.inserted_id})
    print(inserted_document)

    client.close()


if __name__ == "__main__":
    main()
from datetime import datetime, timezone

from pymongo import MongoClient


MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "amazon_reviews_db"
COLLECTION_NAME = "sentiment_predictions"


def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    return client, collection


def safe_probability_to_list(probability):
    if probability is None:
        return []

    return [float(value) for value in probability]


def write_predictions_to_mongodb(batch_df, batch_id):
    rows = batch_df.collect()

    if not rows:
        print(f"Batch {batch_id}: no rows to write.")
        return

    documents = []

    for row in rows:
        document = {
            "text_preview": row["text_preview"],
            "text": row["text"],
            "score": int(row["score"]) if row["score"] is not None else None,
            "prediction": float(row["prediction"]) if row["prediction"] is not None else None,
            "predicted_label": row["predicted_label"],
            "probability": safe_probability_to_list(row["probability"]),
            "batch_id": int(batch_id),
            "processed_at": datetime.now(timezone.utc),
            "source": "spark_structured_streaming",
        }

        documents.append(document)

    client, collection = get_mongo_collection()

    try:
        result = collection.insert_many(documents)
        print(f"Batch {batch_id}: inserted {len(result.inserted_ids)} documents into MongoDB.")
    finally:
        client.close()
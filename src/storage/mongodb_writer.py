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


def safe_get(row, field_name, default=None):
    try:
        value = row[field_name]
    except Exception:
        return default

    if value is None:
        return default

    return value


def build_prediction_document(row, batch_id):
    return {
        "product_id": safe_get(row, "product_id"),
        "user_id": safe_get(row, "user_id"),
        "review_time": safe_get(row, "review_time"),
        "review_date": safe_get(row, "review_date"),
        "score": int(row["score"]) if safe_get(row, "score") is not None else None,
        "summary": safe_get(row, "summary", ""),
        "text_preview": safe_get(row, "text_preview", ""),
        "text": safe_get(row, "text", ""),
        "true_label": safe_get(row, "true_label"),
        "source_split": safe_get(row, "source_split"),
        "source_row_index": (
            int(row["source_row_index"])
            if safe_get(row, "source_row_index") is not None
            else None
        ),
        "prediction": (
            float(row["prediction"])
            if safe_get(row, "prediction") is not None
            else None
        ),
        "predicted_label": safe_get(row, "predicted_label"),
        "probability": safe_probability_to_list(
            safe_get(row, "probability")
        ),
        "batch_id": int(batch_id),
        "processed_at": datetime.now(timezone.utc),
        "source": "spark_structured_streaming",
    }


def write_predictions_to_mongodb(batch_df, batch_id):
    rows = batch_df.collect()

    if not rows:
        print(f"Batch {batch_id}: no rows to write.")
        return

    documents = [
        build_prediction_document(
            row=row,
            batch_id=batch_id
        )
        for row in rows
    ]

    client, collection = get_mongo_collection()

    try:
        result = collection.insert_many(documents)
        print(
            f"Batch {batch_id}: inserted "
            f"{len(result.inserted_ids)} documents into MongoDB."
        )
    finally:
        client.close()
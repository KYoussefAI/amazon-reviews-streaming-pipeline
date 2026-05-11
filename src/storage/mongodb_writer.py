import os
from datetime import datetime, timezone

from pymongo import MongoClient


MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.environ.get("MONGO_DATABASE", "amazon_reviews_db")
COLLECTION_NAME = os.environ.get("MONGO_COLLECTION", "sentiment_predictions")
MONGO_INSERT_CHUNK_SIZE = int(os.environ.get("MONGO_INSERT_CHUNK_SIZE", "1000"))

DEFAULT_SOURCE = "spark_structured_streaming"
DEFAULT_MODEL_TYPE = "unknown"
DOCUMENT_SCHEMA_VERSION = "v2"
INDEXES_CREATED = False


def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    return client, collection


def safe_get(row, field_name, default=None):
    """
    Safely read a column from a Spark Row.

    Spark Rows raise an exception if the field does not exist. This helper keeps
    the writer compatible with different prediction schemas:
    - Logistic Regression
    - One-vs-Rest Linear SVC
    - Majority-vote ensemble
    """

    try:
        value = row[field_name]
    except Exception:
        return default

    if value is None:
        return default

    return value


def safe_int(value, default=None):
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=None):
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_string(value, default=""):
    if value is None:
        return default

    return str(value)


def safe_probability_to_list(probability):
    """
    Convert Spark ML probability output into a plain Python list.

    Supported inputs:
    - None
    - Python list / tuple
    - Spark DenseVector / SparseVector-like objects
    - Any iterable of numeric values

    For models such as LinearSVC that do not output probabilities, this returns [].
    """

    if probability is None:
        return []

    try:
        values = probability.toArray().tolist()
    except AttributeError:
        try:
            values = list(probability)
        except TypeError:
            return []

    cleaned_values = []

    for value in values:
        numeric_value = safe_float(value)

        if numeric_value is not None:
            cleaned_values.append(numeric_value)

    return cleaned_values


def confidence_from_probability(probability_values):
    """
    Confidence used by the dashboard.

    Logistic Regression / Naive Bayes:
        probability = [p_positive, p_negative, p_neutral]
        confidence = max(probability)

    LinearSVC:
        probability = []
        confidence = None
    """

    if not probability_values:
        return None

    return max(probability_values)


def build_confidence_status(confidence):
    if confidence is None:
        return "not_available"

    if confidence < 0.60:
        return "low"

    if confidence < 0.80:
        return "medium"

    return "high"


def build_text_preview(row):
    existing_preview = safe_get(row, "text_preview")

    if existing_preview:
        return safe_string(existing_preview)

    text = safe_string(
        safe_get(row, "text", ""),
        default="",
    )

    return text[:120]


def build_prediction_document(row, batch_id):
    probability_values = safe_probability_to_list(
        safe_get(row, "probability")
    )

    confidence = confidence_from_probability(probability_values)
    confidence_status = build_confidence_status(confidence)

    model_type = safe_get(
        row,
        "model_type",
        DEFAULT_MODEL_TYPE,
    )

    source = safe_get(
        row,
        "source",
        DEFAULT_SOURCE,
    )

    document = {
        "schema_version": DOCUMENT_SCHEMA_VERSION,

        # Source review metadata
        "product_id": safe_get(row, "product_id"),
        "user_id": safe_get(row, "user_id"),
        "review_time": safe_get(row, "review_time"),
        "review_date": safe_get(row, "review_date"),
        "score": safe_int(safe_get(row, "score")),
        "summary": safe_string(safe_get(row, "summary", "")),
        "text_preview": build_text_preview(row),
        "text": safe_string(safe_get(row, "text", "")),

        # Ground truth and source split
        "true_label": safe_get(row, "true_label"),
        "source_split": safe_get(row, "source_split"),
        "source_row_index": safe_int(
            safe_get(row, "source_row_index")
        ),

        # Prediction result
        "prediction": safe_float(
            safe_get(row, "prediction")
        ),
        "predicted_label": safe_get(row, "predicted_label"),

        # Model metadata
        "model_type": model_type,
        "model_name": model_type,

        # Probability / confidence metadata
        "probability": probability_values,
        "confidence": confidence,
        "confidence_status": confidence_status,
        "confidence_available": confidence is not None,

        # Streaming metadata
        "batch_id": safe_int(batch_id),
        "processed_at": datetime.now(timezone.utc),
        "source": source,
    }

    # Optional fields for ensemble streaming.
    # These are included only when the Spark DataFrame contains them.
    optional_fields = [
        "ensemble_members",
        "ensemble_confidence",
        "logistic_regression_predicted_label",
        "naive_bayes_predicted_label",
        "one_vs_rest_linear_svc_predicted_label",
    ]

    for field_name in optional_fields:
        value = safe_get(row, field_name)

        if value is not None:
            if field_name == "ensemble_confidence":
                document[field_name] = safe_float(value)
            elif field_name == "ensemble_members":
                try:
                    document[field_name] = [safe_string(item) for item in list(value)]
                except TypeError:
                    document[field_name] = []
            else:
                document[field_name] = value

    return document


def create_indexes(collection):
    """
    Create useful indexes for dashboard queries.

    MongoDB ignores index creation if the same index already exists.
    These indexes make the Flask dashboard filters faster.
    """

    collection.create_index("processed_at")
    collection.create_index("batch_id")
    collection.create_index("product_id")
    collection.create_index("predicted_label")
    collection.create_index("true_label")
    collection.create_index("score")
    collection.create_index("model_type")
    collection.create_index("source")
    collection.create_index("source_split")
    collection.create_index([("product_id", 1), ("processed_at", -1)])
    collection.create_index([("source", 1), ("processed_at", -1)])


def create_indexes_once(collection):
    global INDEXES_CREATED

    if INDEXES_CREATED:
        return

    create_indexes(collection)
    INDEXES_CREATED = True


def insert_document_chunk(collection, documents):
    if not documents:
        return 0

    result = collection.insert_many(documents, ordered=False)
    return len(result.inserted_ids)


def write_predictions_to_mongodb(batch_df, batch_id):
    client, collection = get_mongo_collection()

    try:
        create_indexes_once(collection)

        documents = []
        inserted_count = 0
        confidence_available_count = 0
        model_types = set()

        for row in batch_df.toLocalIterator():
            document = build_prediction_document(
                row=row,
                batch_id=batch_id
            )

            documents.append(document)
            model_types.add(document.get("model_type", DEFAULT_MODEL_TYPE))

            if document.get("confidence_available"):
                confidence_available_count += 1

            if len(documents) >= MONGO_INSERT_CHUNK_SIZE:
                inserted_count += insert_document_chunk(collection, documents)
                documents = []

        inserted_count += insert_document_chunk(collection, documents)

        if inserted_count == 0:
            print(f"Batch {batch_id}: no rows to write.")
            return

        print(
            f"Batch {batch_id}: inserted "
            f"{inserted_count} documents into MongoDB. "
            f"model_type={sorted(model_types)} | "
            f"confidence_available={confidence_available_count}/{inserted_count}"
        )
    finally:
        client.close()

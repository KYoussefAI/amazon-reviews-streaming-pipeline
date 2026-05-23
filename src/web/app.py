import os
from datetime import datetime
from io import BytesIO
from statistics import mean

from flask import Flask, jsonify, render_template, request, send_file
from pymongo import MongoClient, DESCENDING, ASCENDING

from src.config import MongoSettings

MONGO_SETTINGS = MongoSettings()
MONGO_URI = MONGO_SETTINGS.uri
DATABASE_NAME = MONGO_SETTINGS.database
COLLECTION_NAME = MONGO_SETTINGS.collection

SOURCE_NAME = os.environ.get("MONGO_SOURCE", MONGO_SETTINGS.source)
REQUIRED_PRODUCT_ID = os.environ.get("REQUIRED_PRODUCT_ID", "B001E4KFG0")

DEFAULT_LIMIT = 100
MAX_LIMIT = 5000

# These limits protect the Flask dashboard from trying to load the whole collection
# into memory for calculations that require Python-side probability parsing.
ANALYTICS_SCAN_LIMIT = int(os.environ.get("ANALYTICS_SCAN_LIMIT", "100000"))
RISK_SAMPLE_LIMIT = 50
LATEST_LIMIT_MAX = 5000

LABELS = ["positive", "negative", "neutral"]


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)


def get_collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2500)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    return client, collection


def parse_int(value, default=None):
    if value in (None, "", "All", "all"):
        return default

    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def parse_filters():
    limit = parse_int(request.args.get("limit"), DEFAULT_LIMIT)

    return {
        "source": request.args.get("source", SOURCE_NAME).strip(),
        "sentiment": request.args.get("sentiment", "All").strip(),
        "score": request.args.get("score", "All").strip(),
        "batch_id": request.args.get("batch_id", "All").strip(),
        "product_id": request.args.get("product_id", "All").strip(),
        "model_type": request.args.get("model_type", "All").strip(),
        "source_split": request.args.get("source_split", "All").strip(),
        "start_date": request.args.get("start_date", "").strip(),
        "end_date": request.args.get("end_date", "").strip(),
        "limit": min(limit or DEFAULT_LIMIT, MAX_LIMIT),
    }


def build_match(filters):
    query = {}

    source = filters.get("source") or SOURCE_NAME
    if source not in ("All", "all", ""):
        query["source"] = source

    sentiment = filters.get("sentiment")
    if sentiment not in (None, "", "All", "all"):
        query["predicted_label"] = sentiment

    score = parse_int(filters.get("score"))
    if score is not None:
        query["score"] = score

    batch_id = parse_int(filters.get("batch_id"))
    if batch_id is not None:
        query["batch_id"] = batch_id

    product_id = filters.get("product_id")
    if product_id not in (None, "", "All", "all"):
        query["product_id"] = product_id

    model_type = filters.get("model_type")
    if model_type not in (None, "", "All", "all"):
        query["model_type"] = model_type

    source_split = filters.get("source_split")
    if source_split not in (None, "", "All", "all"):
        query["source_split"] = source_split

    start_date = filters.get("start_date")
    end_date = filters.get("end_date")

    if start_date or end_date:
        date_query = {}
        if start_date:
            date_query["$gte"] = start_date
        if end_date:
            date_query["$lte"] = end_date
        query["review_date"] = date_query

    return query


def confidence_from_probability(probability):
    if not isinstance(probability, list) or not probability:
        return None

    numeric_values = []

    for value in probability:
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue

    if not numeric_values:
        return None

    return max(numeric_values)


def format_datetime(value):
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")

    return str(value)


def serialize_prediction(document):
    if not document:
        return None

    probability = document.get("probability", [])
    confidence = confidence_from_probability(probability)

    return {
        "product_id": document.get("product_id"),
        "user_id": document.get("user_id"),
        "review_time": document.get("review_time"),
        "review_date": document.get("review_date"),
        "score": document.get("score"),
        "summary": document.get("summary"),
        "text_preview": document.get("text_preview") or (document.get("text") or "")[:140],
        "text": document.get("text"),
        "true_label": document.get("true_label"),
        "predicted_label": document.get("predicted_label"),
        "prediction": document.get("prediction"),
        "probability": probability,
        "confidence": confidence,
        "confidence_display": f"{confidence * 100:.2f}%" if confidence is not None else "N/A",
        "confidence_status": "available" if confidence is not None else "not_available",
        "model_type": document.get("model_type", "unknown"),
        "source_split": document.get("source_split"),
        "source_row_index": document.get("source_row_index"),
        "batch_id": document.get("batch_id"),
        "processed_at": format_datetime(document.get("processed_at")),
        "source": document.get("source"),
    }


def aggregate_counts(collection, match, field, limit=None):
    pipeline = [
        {"$match": match},
        {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]

    if limit:
        pipeline.append({"$limit": limit})

    return [
        {
            "label": item["_id"] if item["_id"] is not None else "unknown",
            "count": item["count"],
        }
        for item in collection.aggregate(pipeline)
    ]


def aggregate_score_distribution(collection, match):
    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$score", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]

    return [
        {
            "score": item["_id"],
            "count": item["count"],
        }
        for item in collection.aggregate(pipeline)
    ]


def aggregate_score_by_sentiment(collection, match):
    pipeline = [
        {"$match": match},
        {
            "$group": {
                "_id": {
                    "score": "$score",
                    "predicted_label": "$predicted_label",
                },
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id.score": 1, "_id.predicted_label": 1}},
    ]

    return [
        {
            "score": item["_id"].get("score"),
            "predicted_label": item["_id"].get("predicted_label") or "unknown",
            "count": item["count"],
        }
        for item in collection.aggregate(pipeline)
    ]


def aggregate_predictions_by_date(collection, match, max_dates=420):
    pipeline = [
        {"$match": {**match, "review_date": {"$ne": None}}},
        {
            "$group": {
                "_id": {
                    "review_date": "$review_date",
                    "predicted_label": "$predicted_label",
                },
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id.review_date": 1}},
    ]

    raw = list(collection.aggregate(pipeline))

    grouped = {}

    for item in raw:
        date_value = item["_id"].get("review_date")
        label = item["_id"].get("predicted_label") or "unknown"

        if date_value not in grouped:
            grouped[date_value] = {
                "review_date": date_value,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "unknown": 0,
                "total": 0,
            }

        grouped[date_value][label] = grouped[date_value].get(label, 0) + item["count"]
        grouped[date_value]["total"] += item["count"]

    rows = list(grouped.values())

    if len(rows) > max_dates:
        step = max(1, len(rows) // max_dates)
        rows = rows[::step]

    return rows


def aggregate_batch_records(collection, match, max_batches=250):
    pipeline = [
        {"$match": {**match, "batch_id": {"$ne": None}}},
        {
            "$group": {
                "_id": "$batch_id",
                "records": {"$sum": 1},
                "first_processed_at": {"$min": "$processed_at"},
                "last_processed_at": {"$max": "$processed_at"},
            }
        },
        {"$sort": {"_id": -1}},
        {"$limit": max_batches},
        {"$sort": {"_id": 1}},
    ]

    return [
        {
            "batch_id": item["_id"],
            "records": item["records"],
            "first_processed_at": format_datetime(item.get("first_processed_at")),
            "last_processed_at": format_datetime(item.get("last_processed_at")),
        }
        for item in collection.aggregate(pipeline)
    ]


def aggregate_confusion_matrix(collection, match):
    pipeline = [
        {
            "$match": {
                **match,
                "true_label": {"$in": LABELS},
                "predicted_label": {"$in": LABELS},
            }
        },
        {
            "$group": {
                "_id": {
                    "true_label": "$true_label",
                    "predicted_label": "$predicted_label",
                },
                "count": {"$sum": 1},
            }
        },
    ]

    matrix = {
        true_label: {
            predicted_label: 0
            for predicted_label in LABELS
        }
        for true_label in LABELS
    }

    total = 0
    correct = 0

    for item in collection.aggregate(pipeline):
        true_label = item["_id"]["true_label"]
        predicted_label = item["_id"]["predicted_label"]
        count = item["count"]

        matrix[true_label][predicted_label] = count
        total += count

        if true_label == predicted_label:
            correct += count

    rows = []

    for true_label in LABELS:
        for predicted_label in LABELS:
            rows.append({
                "true_label": true_label,
                "predicted_label": predicted_label,
                "count": matrix[true_label][predicted_label],
            })

    return {
        "rows": rows,
        "accuracy": correct / total if total else None,
        "correct": correct,
        "total": total,
    }


def calculate_confidence_analytics(collection, match):
    cursor = collection.find(
        match,
        {
            "probability": 1,
            "predicted_label": 1,
            "score": 1,
            "model_type": 1,
        }
    ).sort("processed_at", DESCENDING).limit(ANALYTICS_SCAN_LIMIT)

    confidences = []
    by_label = {
        label: []
        for label in LABELS
    }

    model_confidence = {}
    missing_confidence = 0
    total_scanned = 0
    buckets = [
        {"label": "0.00-0.50", "min": 0.0, "max": 0.50, "count": 0},
        {"label": "0.50-0.60", "min": 0.50, "max": 0.60, "count": 0},
        {"label": "0.60-0.70", "min": 0.60, "max": 0.70, "count": 0},
        {"label": "0.70-0.80", "min": 0.70, "max": 0.80, "count": 0},
        {"label": "0.80-0.90", "min": 0.80, "max": 0.90, "count": 0},
        {"label": "0.90-1.00", "min": 0.90, "max": 1.01, "count": 0},
    ]

    for document in cursor:
        total_scanned += 1
        confidence = confidence_from_probability(document.get("probability"))

        if confidence is None:
            missing_confidence += 1
            continue

        confidences.append(confidence)

        predicted_label = document.get("predicted_label") or "unknown"
        if predicted_label in by_label:
            by_label[predicted_label].append(confidence)

        model_type = document.get("model_type", "unknown")
        model_confidence.setdefault(model_type, []).append(confidence)

        for bucket in buckets:
            if bucket["min"] <= confidence < bucket["max"]:
                bucket["count"] += 1
                break

    confidence_by_label = [
        {
            "label": label,
            "average_confidence": mean(values) if values else None,
            "average_confidence_display": f"{mean(values) * 100:.2f}%" if values else "N/A",
            "count": len(values),
        }
        for label, values in by_label.items()
    ]

    confidence_by_model = [
        {
            "label": model_type,
            "average_confidence": mean(values) if values else None,
            "average_confidence_display": f"{mean(values) * 100:.2f}%" if values else "N/A",
            "count": len(values),
        }
        for model_type, values in model_confidence.items()
    ]

    average_confidence = mean(confidences) if confidences else None

    return {
        "average_confidence": average_confidence,
        "average_confidence_display": f"{average_confidence * 100:.2f}%" if average_confidence is not None else "N/A",
        "confidence_available_count": len(confidences),
        "confidence_missing_count": missing_confidence,
        "confidence_coverage": len(confidences) / total_scanned if total_scanned else 0,
        "low_confidence_count": sum(1 for confidence in confidences if confidence < 0.60),
        "confidence_distribution": [
            {
                "label": bucket["label"],
                "count": bucket["count"],
            }
            for bucket in buckets
        ],
        "confidence_by_label": confidence_by_label,
        "confidence_by_model": confidence_by_model,
    }


def suspicious_match(base_match):
    return {
        **base_match,
        "$or": [
            {"score": {"$gte": 4}, "predicted_label": "negative"},
            {"score": {"$lte": 2}, "predicted_label": "positive"},
            {"score": 5, "predicted_label": "neutral"},
            {"score": 1, "predicted_label": "neutral"},
        ],
    }


def low_confidence_documents(collection, match):
    docs = []
    cursor = collection.find(
        match,
        {"_id": 0}
    ).sort("processed_at", DESCENDING).limit(ANALYTICS_SCAN_LIMIT)

    for document in cursor:
        confidence = confidence_from_probability(document.get("probability"))

        if confidence is not None and confidence < 0.60:
            docs.append(serialize_prediction(document))

        if len(docs) >= RISK_SAMPLE_LIMIT:
            break

    return docs


@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        required_product_id=REQUIRED_PRODUCT_ID,
        source_name=SOURCE_NAME,
    )


@app.route("/api/health")
def api_health():
    client = None

    try:
        client, collection = get_collection()
        client.admin.command("ping")
        total_documents = collection.count_documents({})
        return jsonify({
            "connected": True,
            "database": DATABASE_NAME,
            "collection": COLLECTION_NAME,
            "total_documents": total_documents,
            "message": "Connected",
        })
    except Exception as exc:
        return jsonify({
            "connected": False,
            "database": DATABASE_NAME,
            "collection": COLLECTION_NAME,
            "total_documents": 0,
            "message": str(exc),
        }), 503
    finally:
        if client:
            client.close()


@app.route("/api/options")
def api_options():
    client = None

    try:
        client, collection = get_collection()

        product_pipeline = [
            {"$match": {"product_id": {"$ne": None}}},
            {"$group": {"_id": "$product_id", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 3000},
        ]

        products = [item["_id"] for item in collection.aggregate(product_pipeline)]

        if REQUIRED_PRODUCT_ID not in products:
            exists = collection.count_documents({"product_id": REQUIRED_PRODUCT_ID}, limit=1)
            if exists:
                products.insert(0, REQUIRED_PRODUCT_ID)

        batch_pipeline = [
            {"$match": {"batch_id": {"$ne": None}}},
            {"$group": {"_id": "$batch_id"}},
            {"$sort": {"_id": -1}},
            {"$limit": 300},
        ]

        batches = [item["_id"] for item in collection.aggregate(batch_pipeline)]

        model_types = [
            item["label"]
            for item in aggregate_counts(collection, {}, "model_type")
        ]

        source_splits = [
            item["label"]
            for item in aggregate_counts(collection, {}, "source_split")
        ]

        return jsonify({
            "products": products,
            "batches": batches,
            "model_types": model_types,
            "source_splits": source_splits,
            "required_product_id": REQUIRED_PRODUCT_ID,
            "source_name": SOURCE_NAME,
        })
    finally:
        if client:
            client.close()


@app.route("/api/summary")
def api_summary():
    filters = parse_filters()
    match = build_match(filters)

    client = None

    try:
        client, collection = get_collection()
        total_predictions = collection.count_documents(match)

        label_counts = aggregate_counts(collection, match, "predicted_label")
        source_counts = aggregate_counts(collection, match, "source_split")
        model_counts = aggregate_counts(collection, match, "model_type")
        confidence = calculate_confidence_analytics(collection, match)
        confusion = aggregate_confusion_matrix(collection, match)

        latest_doc = collection.find_one(match, sort=[("processed_at", DESCENDING)])
        oldest_doc = collection.find_one(match, sort=[("processed_at", ASCENDING)])

        latest_batch = latest_doc.get("batch_id") if latest_doc else None

        counts_by_label = {
            item["label"]: item["count"]
            for item in label_counts
        }

        return jsonify({
            "filters": filters,
            "total_predictions": total_predictions,
            "latest_batch_id": latest_batch,
            "positive_predictions": counts_by_label.get("positive", 0),
            "negative_predictions": counts_by_label.get("negative", 0),
            "neutral_predictions": counts_by_label.get("neutral", 0),
            "label_counts": label_counts,
            "source_counts": source_counts,
            "model_counts": model_counts,
            "streamed_accuracy": confusion["accuracy"],
            "streamed_accuracy_display": f"{confusion['accuracy'] * 100:.2f}%" if confusion["accuracy"] is not None else "N/A",
            "streamed_correct": confusion["correct"],
            "streamed_labeled_total": confusion["total"],
            "latest_processed_at": format_datetime(latest_doc.get("processed_at")) if latest_doc else None,
            "oldest_processed_at": format_datetime(oldest_doc.get("processed_at")) if oldest_doc else None,
            **confidence,
        })
    finally:
        if client:
            client.close()


@app.route("/api/charts")
def api_charts():
    filters = parse_filters()
    match = build_match(filters)

    client = None

    try:
        client, collection = get_collection()

        confidence = calculate_confidence_analytics(collection, match)
        confusion = aggregate_confusion_matrix(collection, match)

        return jsonify({
            "sentiment_distribution": aggregate_counts(collection, match, "predicted_label"),
            "score_distribution": aggregate_score_distribution(collection, match),
            "score_by_sentiment": aggregate_score_by_sentiment(collection, match),
            "predictions_by_date": aggregate_predictions_by_date(collection, match),
            "source_split_distribution": aggregate_counts(collection, match, "source_split"),
            "model_type_distribution": aggregate_counts(collection, match, "model_type"),
            "batch_records": aggregate_batch_records(collection, match),
            "confidence_distribution": confidence["confidence_distribution"],
            "confidence_by_label": confidence["confidence_by_label"],
            "confidence_by_model": confidence["confidence_by_model"],
            "confusion_matrix": confusion["rows"],
            "confusion_accuracy": confusion["accuracy"],
        })
    finally:
        if client:
            client.close()


@app.route("/api/risk")
def api_risk():
    filters = parse_filters()
    match = build_match(filters)

    client = None

    try:
        client, collection = get_collection()

        suspicious_query = suspicious_match(match)

        suspicious_docs = [
            serialize_prediction(document)
            for document in collection.find(
                suspicious_query,
                {"_id": 0}
            ).sort("processed_at", DESCENDING).limit(RISK_SAMPLE_LIMIT)
        ]

        low_confidence_docs = low_confidence_documents(collection, match)

        return jsonify({
            "suspicious_count": collection.count_documents(suspicious_query),
            "suspicious_samples": suspicious_docs,
            "low_confidence_samples": low_confidence_docs,
        })
    finally:
        if client:
            client.close()


@app.route("/api/latest")
def api_latest():
    filters = parse_filters()
    match = build_match(filters)
    limit = min(filters["limit"], LATEST_LIMIT_MAX)

    client = None

    try:
        client, collection = get_collection()

        cursor = collection.find(
            match,
            {"_id": 0}
        ).sort("processed_at", DESCENDING).limit(limit)

        return jsonify([
            serialize_prediction(document)
            for document in cursor
        ])
    finally:
        if client:
            client.close()


@app.route("/api/product/<product_id>")
def api_product(product_id):
    client = None

    try:
        client, collection = get_collection()

        match = {
            "source": SOURCE_NAME,
            "product_id": product_id,
        }

        total = collection.count_documents(match)
        latest_doc = collection.find_one(match, sort=[("processed_at", DESCENDING)])

        label_counts = aggregate_counts(collection, match, "predicted_label")
        score_distribution = aggregate_score_distribution(collection, match)
        source_counts = aggregate_counts(collection, match, "source_split")
        model_counts = aggregate_counts(collection, match, "model_type")
        confidence = calculate_confidence_analytics(collection, match)
        confusion = aggregate_confusion_matrix(collection, match)

        latest_rows = collection.find(match, {"_id": 0}).sort("processed_at", DESCENDING).limit(30)

        return jsonify({
            "product_id": product_id,
            "total_predictions": total,
            "latest": serialize_prediction(latest_doc) if latest_doc else None,
            "sentiment_distribution": label_counts,
            "score_distribution": score_distribution,
            "source_distribution": source_counts,
            "model_distribution": model_counts,
            "streamed_accuracy": confusion["accuracy"],
            "streamed_accuracy_display": f"{confusion['accuracy'] * 100:.2f}%" if confusion["accuracy"] is not None else "N/A",
            "latest_predictions": [
                serialize_prediction(document)
                for document in latest_rows
            ],
            **confidence,
        })
    finally:
        if client:
            client.close()


@app.route("/api/report.pdf")
def api_report_pdf():
    filters = parse_filters()
    match = build_match(filters)

    client = None

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas
    except ImportError:
        return jsonify({
            "error": "reportlab is not installed. Install it with: pip install reportlab"
        }), 500

    try:
        client, collection = get_collection()

        total_predictions = collection.count_documents(match)
        label_counts = aggregate_counts(collection, match, "predicted_label")
        model_counts = aggregate_counts(collection, match, "model_type")
        source_counts = aggregate_counts(collection, match, "source_split")
        confidence = calculate_confidence_analytics(collection, match)
        confusion = aggregate_confusion_matrix(collection, match)

        required_latest = collection.find_one(
            {"source": SOURCE_NAME, "product_id": REQUIRED_PRODUCT_ID},
            sort=[("processed_at", DESCENDING)]
        )

        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 2 * cm

        def line(text, size=10, gap=0.55):
            nonlocal y
            if y < 2 * cm:
                pdf.showPage()
                y = height - 2 * cm

            pdf.setFont("Helvetica", size)
            pdf.drawString(2 * cm, y, str(text))
            y -= gap * cm

        pdf.setTitle("Amazon Reviews Professional Web Dashboard Report")

        line("Amazon Reviews Professional Web Dashboard Report", size=16, gap=0.9)
        line(f"Generated at: {datetime.now().isoformat(timespec='seconds')}", size=9)
        line(f"MongoDB: {DATABASE_NAME}.{COLLECTION_NAME}", size=9)
        line(f"Source filter: {SOURCE_NAME}", size=9)
        line("")

        line("1. Executive Summary", size=13, gap=0.75)
        line(f"Total predictions: {total_predictions}")
        line(f"Streamed labeled accuracy: {confusion['accuracy'] * 100:.2f}%" if confusion["accuracy"] is not None else "Streamed labeled accuracy: N/A")
        line(f"Average confidence: {confidence['average_confidence_display']}")
        line(f"Confidence available: {confidence['confidence_available_count']}")
        line(f"Confidence missing: {confidence['confidence_missing_count']}")
        line("")

        line("2. Sentiment Distribution", size=13, gap=0.75)
        for item in label_counts:
            line(f"{item['label']}: {item['count']}")

        line("")
        line("3. Model Type Distribution", size=13, gap=0.75)
        for item in model_counts:
            line(f"{item['label']}: {item['count']}")

        line("")
        line("4. Source Split Distribution", size=13, gap=0.75)
        for item in source_counts:
            line(f"{item['label']}: {item['count']}")

        line("")
        line(f"5. Required ProductId Analysis: {REQUIRED_PRODUCT_ID}", size=13, gap=0.75)

        if required_latest:
            latest = serialize_prediction(required_latest)
            line(f"Review date: {latest.get('review_date')}")
            line(f"Score: {latest.get('score')}")
            line(f"True label: {latest.get('true_label')}")
            line(f"Predicted label: {latest.get('predicted_label')}")
            line(f"Model type: {latest.get('model_type')}")
            line(f"Confidence: {latest.get('confidence_display')}")
            line(f"Source split: {latest.get('source_split')}")
            line(f"Batch ID: {latest.get('batch_id')}")
        else:
            line("No prediction found yet for the required ProductId.")

        line("")
        line("6. Risk Monitoring", size=13, gap=0.75)
        risk_query = suspicious_match(match)
        line(f"Suspicious score/sentiment combinations: {collection.count_documents(risk_query)}")
        line(f"Low confidence samples available: {len(low_confidence_documents(collection, match))}")

        pdf.save()
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="amazon_reviews_professional_web_dashboard_report.pdf",
        )
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    port = int(os.environ.get("WEB_DASHBOARD_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

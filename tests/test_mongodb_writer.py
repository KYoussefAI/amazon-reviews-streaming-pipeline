from datetime import datetime, timezone

from src.storage.mongodb_writer import (
    build_prediction_document,
    validate_prediction_document,
)


def test_build_prediction_document_generates_required_fields():
    row = {
        "product_id": "P1",
        "user_id": "U1",
        "review_time": "2024-01-01 12:00:00",
        "review_date": "2024-01-01",
        "score": 4,
        "summary": "Summary",
        "text_preview": "Preview",
        "text": "Full text",
        "true_label": "positive",
        "source_split": "test",
        "source_row_index": 9,
        "prediction": 0.0,
        "predicted_label": "positive",
        "probability": [0.9, 0.05, 0.05],
        "model_type": "one_vs_rest_linear_svc",
        "source": "spark_structured_streaming",
    }

    document = build_prediction_document(row, batch_id=3)

    assert document["schema_version"] == "v2"
    assert document["batch_id"] == 3
    assert document["confidence_available"] is True
    assert document["processed_at"].tzinfo == timezone.utc


def test_validate_prediction_document_rejects_missing_fields():
    document = {
        "schema_version": "v2",
        "product_id": "P1",
        "review_date": "2024-01-01",
        "score": 5,
        "true_label": "positive",
        "predicted_label": "positive",
        "model_type": "model",
        "source_split": "test",
        "batch_id": 1,
        "processed_at": datetime.now(timezone.utc),
        "source": "",
    }

    try:
        validate_prediction_document(document)
    except ValueError as exc:
        assert "missing required fields" in str(exc)
    else:
        raise AssertionError("Expected invalid MongoDB document to fail validation.")

from src.ingestion.producer import build_message, validate_message


def test_build_message_preserves_required_fields():
    row = {
        "product_id": "P1",
        "user_id": "U1",
        "review_time": "2024-01-01 00:00:00",
        "review_date": "2024-01-01",
        "score": 5,
        "summary": "Good",
        "text": "Useful review",
        "label": "positive",
        "source_split": "test",
    }

    message = build_message(row, source_row_index=7)

    assert message["product_id"] == "P1"
    assert message["score"] == 5
    assert message["source_row_index"] == 7


def test_validate_message_rejects_invalid_score():
    message = {
        "product_id": "P1",
        "user_id": "U1",
        "review_time": "2024-01-01 00:00:00",
        "review_date": "2024-01-01",
        "score": 8,
        "summary": "",
        "text": "Useful review",
        "label": "positive",
        "source_split": "test",
        "source_row_index": 1,
    }

    try:
        validate_message(message)
    except ValueError as exc:
        assert "invalid score" in str(exc)
    else:
        raise AssertionError("Expected invalid Kafka message score to fail validation.")

from kafka import KafkaProducer
import json
import os
import time
import pandas as pd


STREAM_DATA_PATH = "data/processed/test_reviews.jsonl"

KAFKA_TOPIC = "amazon_reviews"
KAFKA_BOOTSTRAP_SERVERS = os.environ.get(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092,localhost:9093,localhost:9094",
)

SLEEP_SECONDS = float(os.environ.get("PRODUCER_SLEEP_SECONDS", "0.2"))
START_ROW = 0

REQUIRED_COLUMNS = [
    "product_id",
    "user_id",
    "review_time",
    "review_date",
    "score",
    "summary",
    "text",
    "label",
    "source_split",
]


def validate_columns(df):
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns in streaming export file: "
            + ", ".join(missing_columns)
        )


def clean_stream_dataframe(df):
    validate_columns(df)

    df = df[REQUIRED_COLUMNS].copy()

    df = df.dropna(
        subset=[
            "product_id",
            "user_id",
            "review_time",
            "review_date",
            "score",
            "text",
            "label",
            "source_split",
        ]
    ).reset_index(drop=True)

    df["score"] = pd.to_numeric(
        df["score"],
        errors="coerce"
    )

    bad_rows = df[df["score"].isna()]

    if not bad_rows.empty:
        print("========== WARNING: BAD SCORE ROWS FOUND ==========")
        print(bad_rows.head())
        print(f"Bad rows removed: {len(bad_rows)}")

    df = df.dropna(subset=["score"])
    df["score"] = df["score"].astype(int)

    df = df[
        (df["score"] >= 1) &
        (df["score"] <= 5)
    ].reset_index(drop=True)

    df["summary"] = df["summary"].fillna("")

    return df


def build_message(row, source_row_index):
    return {
        "product_id": str(row["product_id"]),
        "user_id": str(row["user_id"]),
        "review_time": str(row["review_time"]),
        "review_date": str(row["review_date"]),
        "score": int(row["score"]),
        "summary": str(row["summary"]),
        "text": str(row["text"]),
        "label": str(row["label"]),
        "source_split": str(row["source_split"]),
        "source_row_index": int(source_row_index),
    }


def main():
    df = pd.read_json(
        STREAM_DATA_PATH,
        lines=True
    )

    df = clean_stream_dataframe(df)

    if START_ROW >= len(df):
        print("No rows left to stream.")
        print(f"START_ROW={START_ROW}, total rows={len(df)}")
        return

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(
            value,
            ensure_ascii=False
        ).encode("utf-8"),
        retries=5
    )

    print("========== PRODUCER STARTED ==========")
    print(f"Streaming from: {STREAM_DATA_PATH}")
    print(f"Rows available: {len(df)}")
    print(f"Starting from row index: {START_ROW}")
    print(f"Kafka topic: {KAFKA_TOPIC}")
    print("Message fields:")
    print(", ".join(REQUIRED_COLUMNS + ["source_row_index"]))

    try:
        for index, row in df.iloc[START_ROW:].iterrows():
            message = build_message(
                row=row,
                source_row_index=index
            )

            producer.send(
                KAFKA_TOPIC,
                message
            )

            print(
                f"Sent row {index + 1}/{len(df)} | "
                f"product_id={message['product_id']} | "
                f"review_date={message['review_date']} | "
                f"score={message['score']} | "
                f"label={message['label']} | "
                f"source_split={message['source_split']} | "
                f"text={message['text'][:80]}..."
            )

            if SLEEP_SECONDS > 0:
                time.sleep(SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("Producer stopped manually.")

    finally:
        producer.flush()
        producer.close()
        print("========== PRODUCER CLOSED ==========")


if __name__ == "__main__":
    main()

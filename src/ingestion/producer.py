import argparse
import json
import logging
import time

import pandas as pd
from kafka import KafkaProducer

from src.config import KafkaSettings, ProducerSettings

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


logger = logging.getLogger(__name__)


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
        logger.warning("Bad score rows found; removed %s rows.", len(bad_rows))
        logger.warning("Sample invalid rows:\n%s", bad_rows.head().to_string(index=False))

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


def validate_message(message):
    missing_fields = [
        field
        for field in REQUIRED_COLUMNS + ["source_row_index"]
        if field not in message
    ]

    if missing_fields:
        raise ValueError(
            "Kafka message is missing required fields: "
            + ", ".join(missing_fields)
        )

    score = message["score"]
    if not isinstance(score, int) or score < 1 or score > 5:
        raise ValueError(f"Kafka message has invalid score: {score}")


def parse_args():
    settings = ProducerSettings()

    parser = argparse.ArgumentParser(
        description="Stream exported Amazon review records into Kafka."
    )
    parser.add_argument(
        "--input-path",
        default=str(settings.stream_data_path),
        help="Path to the JSONL file produced for streaming simulation.",
    )
    parser.add_argument(
        "--topic",
        default=KafkaSettings().topic,
        help="Kafka topic to publish to.",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=KafkaSettings().bootstrap_servers,
        help="Kafka bootstrap servers list.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=settings.sleep_seconds,
        help="Delay between messages. Set to 0 for fastest replay.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=settings.start_row,
        help="Zero-based row index to start streaming from.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )

    return parser.parse_args()


def configure_logging(level):
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main():
    args = parse_args()
    configure_logging(args.log_level)

    df = pd.read_json(
        args.input_path,
        lines=True
    )

    df = clean_stream_dataframe(df)

    if args.start_row >= len(df):
        logger.info(
            "No rows left to stream. start_row=%s total_rows=%s",
            args.start_row,
            len(df),
        )
        return

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda value: json.dumps(
            value,
            ensure_ascii=False
        ).encode("utf-8"),
        retries=5
    )

    logger.info("Producer started.")
    logger.info("Streaming from %s", args.input_path)
    logger.info("Rows available: %s", len(df))
    logger.info("Starting from row index: %s", args.start_row)
    logger.info("Kafka topic: %s", args.topic)
    logger.info("Message fields: %s", ", ".join(REQUIRED_COLUMNS + ["source_row_index"]))

    try:
        for index, row in df.iloc[args.start_row:].iterrows():
            message = build_message(
                row=row,
                source_row_index=index
            )
            validate_message(message)

            producer.send(
                args.topic,
                message
            )

            logger.info(
                "Sent row %s/%s | product_id=%s | review_date=%s | score=%s | label=%s | source_split=%s | text=%s...",
                index + 1,
                len(df),
                message["product_id"],
                message["review_date"],
                message["score"],
                message["label"],
                message["source_split"],
                message["text"][:80],
            )

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    except KeyboardInterrupt:
        logger.info("Producer stopped manually.")

    finally:
        producer.flush()
        producer.close()
        logger.info("Producer closed.")


if __name__ == "__main__":
    main()

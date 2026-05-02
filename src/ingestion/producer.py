from kafka import KafkaProducer
import json
import time
import pandas as pd


STREAM_DATA_PATH = "data/processed/test_reviews.jsonl"
KAFKA_TOPIC = "amazon_reviews"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

SLEEP_SECONDS = 0.05
START_ROW = 0


def main():
    df = pd.read_json(
        STREAM_DATA_PATH,
        lines=True
    )

    df = df[["text", "score", "label"]].dropna().reset_index(drop=True)

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

    if START_ROW >= len(df):
        print("No rows left to stream.")
        print(f"START_ROW={START_ROW}, total rows={len(df)}")
        return

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(
            v,
            ensure_ascii=False
        ).encode("utf-8"),
        retries=5
    )

    print("========== PRODUCER STARTED ==========")
    print(f"Streaming from: {STREAM_DATA_PATH}")
    print(f"Rows available: {len(df)}")
    print(f"Starting from row index: {START_ROW}")
    print(f"Kafka topic: {KAFKA_TOPIC}")

    try:
        for index, row in df.iloc[START_ROW:].iterrows():
            message = {
                "text": str(row["text"]),
                "score": int(row["score"]),
                "source_split": "test",
                "source_row_index": int(index)
            }

            producer.send(KAFKA_TOPIC, message)

            print(
                f"Sent test row {index + 1}/{len(df)} | "
                f"score={message['score']} | "
                f"text={message['text'][:100]}..."
            )

            time.sleep(SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("Producer stopped manually.")

    finally:
        producer.flush()
        producer.close()
        print("========== PRODUCER CLOSED ==========")


if __name__ == "__main__":
    main()
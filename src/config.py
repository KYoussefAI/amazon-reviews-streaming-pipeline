import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STREAM_DATA_PATH = PROJECT_ROOT / "data/processed/test_reviews.jsonl"
DEFAULT_STREAMING_MODEL_PATH = PROJECT_ROOT / "src/spark/model/ensemble_models/one_vs_rest_linear_svc"
DEFAULT_STREAMING_CHECKPOINT_PATH = PROJECT_ROOT / "data/processed/checkpoints/spark_streaming_best_single_model"


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class KafkaSettings:
    bootstrap_servers: str = env(
        "KAFKA_BOOTSTRAP_SERVERS",
        "localhost:9092,localhost:9093,localhost:9094",
    )
    topic: str = env("KAFKA_TOPIC", "amazon_reviews")
    starting_offsets: str = env("KAFKA_STARTING_OFFSETS", "latest")
    max_offsets_per_trigger: str | None = os.environ.get("KAFKA_MAX_OFFSETS_PER_TRIGGER")


@dataclass(frozen=True)
class MongoSettings:
    uri: str = env("MONGO_URI", "mongodb://localhost:27017")
    database: str = env("MONGO_DATABASE", "amazon_reviews_db")
    collection: str = env("MONGO_COLLECTION", "sentiment_predictions")
    source: str = env("MONGO_SOURCE", "spark_structured_streaming")


@dataclass(frozen=True)
class ProducerSettings:
    stream_data_path: Path = Path(
        env("STREAM_DATA_PATH", str(DEFAULT_STREAM_DATA_PATH))
    )
    sleep_seconds: float = float(env("PRODUCER_SLEEP_SECONDS", "0.2"))
    start_row: int = int(env("PRODUCER_START_ROW", "0"))


@dataclass(frozen=True)
class StreamingSettings:
    best_single_model_name: str = env(
        "STREAMING_MODEL_NAME",
        "one_vs_rest_linear_svc",
    )
    best_single_model_path: Path = Path(
        env("STREAMING_MODEL_PATH", str(DEFAULT_STREAMING_MODEL_PATH))
    )
    checkpoint_location: Path = Path(
        env("STREAMING_CHECKPOINT_LOCATION", str(DEFAULT_STREAMING_CHECKPOINT_PATH))
    )
    enable_lemmatization: bool = env("ENABLE_STREAMING_LEMMATIZATION", "0") == "1"
    spark_driver_memory: str = env("SPARK_DRIVER_MEMORY", "4g")


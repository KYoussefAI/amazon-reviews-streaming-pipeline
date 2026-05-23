import argparse
import logging
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    ArrayType,
)

from pyspark.ml import PipelineModel

from src.config import KafkaSettings, StreamingSettings


PROJECT_ROOT = os.getcwd()

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.storage.mongodb_writer import write_predictions_to_mongodb
from src.spark.training.text_normalization import add_lemmatized_text_column

logger = logging.getLogger(__name__)


def parse_args():
    kafka_settings = KafkaSettings()
    streaming_settings = StreamingSettings()

    parser = argparse.ArgumentParser(
        description="Read Amazon review events from Kafka, score them with Spark, and write results to MongoDB."
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=kafka_settings.bootstrap_servers,
        help="Kafka bootstrap servers list.",
    )
    parser.add_argument(
        "--topic",
        default=kafka_settings.topic,
        help="Kafka topic to subscribe to.",
    )
    parser.add_argument(
        "--starting-offsets",
        default=kafka_settings.starting_offsets,
        help="Kafka starting offsets for the stream reader.",
    )
    parser.add_argument(
        "--max-offsets-per-trigger",
        default=kafka_settings.max_offsets_per_trigger,
        help="Optional Spark streaming rate limit.",
    )
    parser.add_argument(
        "--model-path",
        default=str(streaming_settings.best_single_model_path),
        help="Path to the saved Spark PipelineModel used for inference.",
    )
    parser.add_argument(
        "--model-name",
        default=streaming_settings.best_single_model_name,
        help="Model name to store in MongoDB metadata.",
    )
    parser.add_argument(
        "--checkpoint-location",
        default=str(streaming_settings.checkpoint_location),
        help="Checkpoint directory for Spark Structured Streaming.",
    )
    parser.add_argument(
        "--enable-lemmatization",
        action="store_true",
        default=streaming_settings.enable_lemmatization,
        help="Apply text lemmatization before prediction.",
    )
    parser.add_argument(
        "--spark-driver-memory",
        default=streaming_settings.spark_driver_memory,
        help="Spark driver memory setting.",
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


def create_spark_session(spark_driver_memory):
    spark = SparkSession.builder \
        .appName("AmazonReviewsSparkStreamingBestSingleModelPrediction") \
        .config("spark.driver.memory", spark_driver_memory) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    return spark


def load_model(model_name, model_path):
    logger.info("Loading Spark model %s from %s", model_name, model_path)
    model = PipelineModel.load(model_path)

    return model


def get_index_to_label_mapping(model):
    """
    Finds the StringIndexerModel inside the saved PipelineModel.

    The StringIndexerModel contains the label order learned during training.

    Example:
    0.0 -> positive
    1.0 -> negative
    2.0 -> neutral
    """

    for stage in model.stages:
        if hasattr(stage, "labels"):
            return stage.labels

    raise ValueError("No StringIndexerModel with labels found in the saved pipeline.")


def build_prediction_label_column(prediction_column, labels):
    prediction_label_expr = None

    for index, label in enumerate(labels):
        condition = F.col(prediction_column) == float(index)

        if prediction_label_expr is None:
            prediction_label_expr = F.when(condition, F.lit(label))
        else:
            prediction_label_expr = prediction_label_expr.when(
                condition,
                F.lit(label)
            )

    return prediction_label_expr.otherwise(F.lit("unknown"))


def build_message_schema():
    """
    Kafka message schema.

    These fields come from:
    data/processed/test_reviews.jsonl
    -> producer.py
    -> Kafka topic amazon_reviews
    -> Spark Structured Streaming
    """

    return StructType([
        StructField("product_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("review_time", StringType(), True),
        StructField("review_date", StringType(), True),
        StructField("score", IntegerType(), True),
        StructField("summary", StringType(), True),
        StructField("text", StringType(), True),
        StructField("label", StringType(), True),
        StructField("source_split", StringType(), True),
        StructField("source_row_index", IntegerType(), True),
    ])


def parse_kafka_messages(kafka_df, message_schema):
    parsed_df = kafka_df.select(
        F.from_json(
            F.col("value").cast("string"),
            message_schema
        ).alias("data")
    ).select(
        F.col("data.product_id").alias("product_id"),
        F.col("data.user_id").alias("user_id"),
        F.col("data.review_time").alias("review_time"),
        F.col("data.review_date").alias("review_date"),
        F.col("data.score").alias("score"),
        F.col("data.summary").alias("summary"),
        F.col("data.text").alias("text"),
        F.col("data.label").alias("label"),
        F.col("data.source_split").alias("source_split"),
        F.col("data.source_row_index").alias("source_row_index"),
    )

    parsed_df = parsed_df.dropna(
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
    )

    return parsed_df


def build_prediction_input(parsed_df, labels, enable_streaming_lemmatization):
    """
    The saved Spark PipelineModel expects:
    - text
    - label
    - class_weight

    The true label comes from exported test data and is useful for validation.
    class_weight is included for compatibility with shared project schema.
    During inference, it does not change predictions.
    """

    prediction_input_df = parsed_df.withColumn(
        "raw_text",
        F.col("text")
    )

    if enable_streaming_lemmatization:
        prediction_input_df = add_lemmatized_text_column(prediction_input_df)

    prediction_input_df = prediction_input_df.withColumn(
        "label",
        F.when(
            F.col("label").isin(labels),
            F.col("label")
        ).otherwise(F.lit(labels[0]))
    ).withColumn(
        "class_weight",
        F.lit(1.0)
    )

    return prediction_input_df


def build_probability_column(predictions):
    """
    Some Spark classifiers, such as Logistic Regression and Naive Bayes,
    produce a probability column.

    OneVsRest LinearSVC does not provide calibrated probabilities by default.
    For dashboard compatibility, we store an empty probability list.
    """

    if "probability" in predictions.columns:
        return F.col("probability")

    return F.array().cast(ArrayType(DoubleType()))


def build_output_dataframe(predictions, labels, model_name):
    output_df = predictions.withColumn(
        "predicted_label",
        build_prediction_label_column(
            prediction_column="prediction",
            labels=labels
        )
    ).withColumn(
        "true_label",
        F.col("label")
    ).withColumn(
        "probability",
        build_probability_column(predictions)
    ).withColumn(
        "model_type",
        F.lit(model_name)
    ).select(
        "product_id",
        "user_id",
        "review_time",
        "review_date",
        "score",
        "summary",
        F.substring("raw_text", 1, 120).alias("text_preview"),
        F.col("raw_text").alias("text"),
        "true_label",
        "source_split",
        "source_row_index",
        "prediction",
        "predicted_label",
        "probability",
        "model_type",
    )

    return output_df


def main():
    args = parse_args()
    configure_logging(args.log_level)

    spark = create_spark_session(args.spark_driver_memory)
    model = load_model(args.model_name, args.model_path)

    labels = get_index_to_label_mapping(model)

    logger.info("Label index mapping:")
    for index, label in enumerate(labels):
        logger.info("%s -> %s", float(index), label)

    message_schema = build_message_schema()

    logger.info("Reading from Kafka topic %s", args.topic)

    kafka_reader = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", args.bootstrap_servers) \
        .option("subscribe", args.topic) \
        .option("startingOffsets", args.starting_offsets) \
        .option("failOnDataLoss", "false")

    if args.max_offsets_per_trigger:
        kafka_reader = kafka_reader.option(
            "maxOffsetsPerTrigger",
            args.max_offsets_per_trigger,
        )

    kafka_df = kafka_reader.load()

    parsed_df = parse_kafka_messages(
        kafka_df=kafka_df,
        message_schema=message_schema
    )

    prediction_input_df = build_prediction_input(
        parsed_df=parsed_df,
        labels=labels,
        enable_streaming_lemmatization=args.enable_lemmatization,
    )

    predictions = model.transform(prediction_input_df)

    output_df = build_output_dataframe(
        predictions=predictions,
        labels=labels,
        model_name=args.model_name,
    )

    logger.info("Starting streaming prediction query with checkpoint %s", args.checkpoint_location)

    query = output_df.writeStream \
        .foreachBatch(write_predictions_to_mongodb) \
        .outputMode("append") \
        .option(
            "checkpointLocation",
            args.checkpoint_location,
        ) \
        .start()

    query.awaitTermination()


if __name__ == "__main__":
    main()

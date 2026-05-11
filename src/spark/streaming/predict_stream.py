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


PROJECT_ROOT = os.getcwd()

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.storage.mongodb_writer import write_predictions_to_mongodb
from src.spark.training.text_normalization import add_lemmatized_text_column


KAFKA_BOOTSTRAP_SERVERS = os.environ.get(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092,localhost:9093,localhost:9094",
)
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "amazon_reviews")
KAFKA_STARTING_OFFSETS = os.environ.get("KAFKA_STARTING_OFFSETS", "latest")
KAFKA_MAX_OFFSETS_PER_TRIGGER = os.environ.get("KAFKA_MAX_OFFSETS_PER_TRIGGER", "200")
ENABLE_STREAMING_LEMMATIZATION = (
    os.environ.get("ENABLE_STREAMING_LEMMATIZATION", "0") == "1"
)

# Best single saveable model from validation-based comparison.
# The ensemble had the best overall Macro F1, but it is heavier for streaming.
# Among single Spark PipelineModels, One-vs-Rest Linear SVC had the best validation Macro F1.
BEST_SINGLE_MODEL_NAME = "one_vs_rest_linear_svc"
BEST_SINGLE_MODEL_PATH = "src/spark/model/ensemble_models/one_vs_rest_linear_svc"

CHECKPOINT_LOCATION = "data/processed/checkpoints/spark_streaming_best_single_model"


def create_spark_session():
    spark = SparkSession.builder \
        .appName("AmazonReviewsSparkStreamingBestSingleModelPrediction") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    return spark


def load_model():
    print("========== LOADING BEST SINGLE SPARK MODEL ==========")
    print(f"Model name: {BEST_SINGLE_MODEL_NAME}")
    model = PipelineModel.load(BEST_SINGLE_MODEL_PATH)
    print(f"Model loaded from: {BEST_SINGLE_MODEL_PATH}")

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


def build_prediction_input(parsed_df, labels):
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

    if ENABLE_STREAMING_LEMMATIZATION:
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


def build_output_dataframe(predictions, labels):
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
        F.lit(BEST_SINGLE_MODEL_NAME)
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
    spark = create_spark_session()
    model = load_model()

    labels = get_index_to_label_mapping(model)

    print("========== LABEL INDEX MAPPING ==========")
    for index, label in enumerate(labels):
        print(f"{float(index)} -> {label}")

    message_schema = build_message_schema()

    print("========== READING FROM KAFKA ==========")

    kafka_reader = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", KAFKA_STARTING_OFFSETS) \
        .option("failOnDataLoss", "false")

    if KAFKA_MAX_OFFSETS_PER_TRIGGER:
        kafka_reader = kafka_reader.option(
            "maxOffsetsPerTrigger",
            KAFKA_MAX_OFFSETS_PER_TRIGGER,
        )

    kafka_df = kafka_reader.load()

    parsed_df = parse_kafka_messages(
        kafka_df=kafka_df,
        message_schema=message_schema
    )

    prediction_input_df = build_prediction_input(
        parsed_df=parsed_df,
        labels=labels
    )

    predictions = model.transform(prediction_input_df)

    output_df = build_output_dataframe(
        predictions=predictions,
        labels=labels
    )

    print("========== STREAMING BEST SINGLE MODEL PREDICTIONS TO MONGODB STARTED ==========")

    query = output_df.writeStream \
        .foreachBatch(write_predictions_to_mongodb) \
        .outputMode("append") \
        .option(
            "checkpointLocation",
            CHECKPOINT_LOCATION
        ) \
        .start()

    query.awaitTermination()


if __name__ == "__main__":
    main()

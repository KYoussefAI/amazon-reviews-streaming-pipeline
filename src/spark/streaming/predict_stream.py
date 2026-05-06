import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
)

from pyspark.ml import PipelineModel


PROJECT_ROOT = os.getcwd()

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.storage.mongodb_writer import write_predictions_to_mongodb


KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "amazon_reviews"

MODEL_PATH = "src/spark/model/sentiment_pipeline_model"


def create_spark_session():
    spark = SparkSession.builder \
        .appName("AmazonReviewsSparkStreamingPrediction") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    return spark


def load_model():
    print("========== LOADING SAVED PIPELINE MODEL ==========")
    model = PipelineModel.load(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")
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


def build_prediction_label_column(labels):
    prediction_label_expr = None

    for index, label in enumerate(labels):
        condition = F.col("prediction") == float(index)

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

    In streaming, we now receive the true label from the exported test data.
    This label is useful for later validation/debugging, but the prediction itself
    is based on the review text.

    class_weight is added only to satisfy the trained pipeline schema.
    During inference, it does not change the prediction.
    """

    prediction_input_df = parsed_df.withColumn(
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


def build_output_dataframe(predictions, labels):
    output_df = predictions.withColumn(
        "predicted_label",
        build_prediction_label_column(labels)
    ).withColumn(
        "true_label",
        F.col("label")
    ).select(
        "product_id",
        "user_id",
        "review_time",
        "review_date",
        "score",
        "summary",
        F.substring("text", 1, 120).alias("text_preview"),
        "text",
        "true_label",
        "source_split",
        "source_row_index",
        "prediction",
        "predicted_label",
        "probability"
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

    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

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

    print("========== STREAMING PREDICTIONS TO MONGODB STARTED ==========")

    query = output_df.writeStream \
        .foreachBatch(write_predictions_to_mongodb) \
        .outputMode("append") \
        .option("checkpointLocation", "data/processed/checkpoints/spark_streaming_predictions") \
        .start()

    query.awaitTermination()


if __name__ == "__main__":
    main()
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from pyspark.ml import PipelineModel

from src.storage.mongodb_writer import write_predictions_to_mongodb

PROJECT_ROOT = os.getcwd()

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


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
    It contains the label order learned during training.
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
            prediction_label_expr = prediction_label_expr.when(condition, F.lit(label))

    return prediction_label_expr.otherwise(F.lit("unknown"))


def main():
    spark = create_spark_session()
    model = load_model()

    labels = get_index_to_label_mapping(model)

    print("========== LABEL INDEX MAPPING ==========")
    for index, label in enumerate(labels):
        print(f"{float(index)} -> {label}")

    message_schema = StructType([
        StructField("text", StringType(), True),
        StructField("score", IntegerType(), True),
    ])

    print("========== READING FROM KAFKA ==========")

    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    parsed_df = kafka_df.select(
        F.from_json(
            F.col("value").cast("string"),
            message_schema
        ).alias("data")
    ).select(
        F.col("data.text").alias("text"),
        F.col("data.score").alias("score")
    ).dropna(subset=["text"])

    # The saved PipelineModel contains a fitted StringIndexerModel.
    # During streaming prediction we do not know the true label.
    # This dummy label only satisfies the pipeline schema.
    # It does not affect the prediction.
    prediction_input_df = parsed_df.withColumn(
        "label",
        F.lit(labels[0])
    ).withColumn(
        "class_weight",
        F.lit(1.0)
    )

    predictions = model.transform(prediction_input_df)

    output_df = predictions.withColumn(
        "predicted_label",
        build_prediction_label_column(labels)
    ).select(
        F.substring("text", 1, 120).alias("text_preview"),
        "text",
        "score",
        "prediction",
        "predicted_label",
        "probability"
    )

    print("========== STREAMING PREDICTIONS TO MONGODB STARTED ==========")

    query = output_df.writeStream \
        .foreachBatch(write_predictions_to_mongodb) \
        .outputMode("append") \
        .start()

    query.awaitTermination()


if __name__ == "__main__":
    main()
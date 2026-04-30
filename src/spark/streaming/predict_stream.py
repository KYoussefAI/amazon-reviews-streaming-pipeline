import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from pyspark.ml import PipelineModel


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
    The saved pipeline contains a StringIndexerModel.
    It stores the label order, for example:
    index 0 -> positive
    index 1 -> negative
    index 2 -> neutral
    """

    for stage in model.stages:
        if hasattr(stage, "labels"):
            return stage.labels

    raise ValueError("No StringIndexerModel with labels found in the saved pipeline.")


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

    # Important:
    # The saved training PipelineModel contains a StringIndexerModel
    # that expects a "label" column during transform().
    # In streaming inference, we do not know the true label.
    # So we add a dummy label only to satisfy the saved pipeline.
    # It does NOT affect prediction.
    prediction_input_df = parsed_df.withColumn(
        "label",
        F.lit(labels[0])
    ).withColumn(
        "class_weight",
        F.lit(1.0)
    )

    predictions = model.transform(prediction_input_df)

    prediction_label_expr = None

    for index, label in enumerate(labels):
        condition = F.col("prediction") == float(index)

        if prediction_label_expr is None:
            prediction_label_expr = F.when(condition, F.lit(label))
        else:
            prediction_label_expr = prediction_label_expr.when(condition, F.lit(label))

    prediction_label_expr = prediction_label_expr.otherwise(F.lit("unknown"))

    output_df = predictions.withColumn(
        "predicted_label",
        prediction_label_expr
    ).select(
        "text",
        "score",
        "prediction",
        "predicted_label",
        "probability"
    )

    print("========== STREAMING PREDICTIONS STARTED ==========")

    query = output_df.writeStream \
        .format("console") \
        .outputMode("append") \
        .option("truncate", "false") \
        .start()

    query.awaitTermination()


if __name__ == "__main__":
    main()
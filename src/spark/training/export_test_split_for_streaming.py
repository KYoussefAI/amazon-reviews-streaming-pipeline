import os
import sys
import shutil
import glob

from pyspark.sql import functions as F


PROJECT_ROOT = os.getcwd()

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.spark.training.train_spark_pipeline import (
    create_spark_session,
    split_dataset,
)


RAW_DATA_PATH = "data/raw/Reviews.csv"
OUTPUT_DIR = "data/processed/test_reviews_stream_json"
FINAL_JSONL_PATH = "data/processed/test_reviews.jsonl"


def load_clean_raw_data(spark):
    """
    Load Amazon Reviews safely and keep only rows valid for the sentiment pipeline.

    Important:
    - Score must be exactly 1, 2, 3, 4, or 5.
    - Text must not be null.
    - We clean before splitting so train/val/test are based on valid records only.
    """

    df = spark.read.csv(
        RAW_DATA_PATH,
        header=True,
        inferSchema=False,
        multiLine=True,
        quote='"',
        escape='"',
        mode="PERMISSIVE"
    )

    df = df.select(
        F.col("Text").alias("text"),
        F.trim(F.col("Score")).alias("Score")
    ).dropna(subset=["text", "Score"])

    df = df.filter(
        F.col("Score").rlike("^[1-5]$")
    )

    df = df.withColumn(
        "Score",
        F.col("Score").cast("int")
    )

    df = df.withColumn(
        "label",
        F.when(F.col("Score") < 3, "negative")
         .when(F.col("Score") == 3, "neutral")
         .otherwise("positive")
    )

    return df.select("text", "Score", "label")


def main():
    spark = create_spark_session()

    print("========== EXPORTING TEST SPLIT FROM TRAINING SPLIT FUNCTION ==========")

    df = load_clean_raw_data(spark)

    print("========== CLEAN DATA CHECK BEFORE SPLIT ==========")
    print(f"Clean rows: {df.count()}")

    print("Score distribution:")
    df.groupBy("Score").count().orderBy("Score").show()

    print("Label distribution:")
    df.groupBy("label").count().show()

    train_df, val_df, test_df = split_dataset(df)

    print("========== TEST SPLIT CHECK ==========")
    print(f"Test rows: {test_df.count()}")

    print("Test score distribution:")
    test_df.groupBy("Score").count().orderBy("Score").show()

    print("Test label distribution:")
    test_df.groupBy("label").count().show()

    os.makedirs("data/processed", exist_ok=True)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    if os.path.exists(FINAL_JSONL_PATH):
        os.remove(FINAL_JSONL_PATH)

    print("========== WRITING TEST SPLIT TO JSONL ==========")

    test_df.select(
        F.col("text").alias("text"),
        F.col("Score").alias("score"),
        F.col("label").alias("label")
    ).coalesce(1).write.mode("overwrite").json(OUTPUT_DIR)

    part_files = glob.glob(f"{OUTPUT_DIR}/part-*.json")

    if not part_files:
        raise FileNotFoundError("No Spark JSON part file found.")

    shutil.copy(part_files[0], FINAL_JSONL_PATH)

    print("========== EXPORT COMPLETE ==========")
    print(f"Final file: {FINAL_JSONL_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
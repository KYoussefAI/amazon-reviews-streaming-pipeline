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

REQUIRED_DASHBOARD_PRODUCT_ID = "B001E4KFG0"
MAX_PRODUCT_DEMO_ROWS = 500


def load_clean_raw_data(spark):
    """
    Load Amazon Reviews safely and keep only rows valid for the sentiment pipeline.

    This export is used for the online prediction simulation:
    Reviews.csv
    -> fixed train/validation/test split
    -> export test split
    -> ensure required dashboard ProductId sample exists
    -> Kafka producer streams the exported rows.

    Important fields preserved:
    - ProductId: needed for product-level dashboard analysis
    - UserId: useful review metadata
    - Time: original Unix timestamp from Amazon Reviews
    - review_time: timestamp converted from Unix time
    - review_date: date used for prediction-by-date dashboard charts
    - Summary: short review summary
    - Text: full review text used by the Spark ML model
    - Score: original score used to create the true label
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
        F.trim(F.col("ProductId")).alias("product_id"),
        F.trim(F.col("UserId")).alias("user_id"),
        F.trim(F.col("Time")).alias("raw_time"),
        F.trim(F.col("Score")).alias("Score"),
        F.col("Summary").alias("summary"),
        F.col("Text").alias("text")
    )

    df = df.dropna(
        subset=[
            "product_id",
            "user_id",
            "raw_time",
            "Score",
            "text"
        ]
    )

    # Keep only clean numeric scores from 1 to 5.
    df = df.filter(
        F.col("Score").rlike("^[1-5]$")
    )

    # Keep only valid Unix timestamps.
    df = df.filter(
        F.col("raw_time").rlike("^[0-9]+$")
    )

    df = df.withColumn(
        "Score",
        F.col("Score").cast("int")
    )

    df = df.withColumn(
        "review_timestamp",
        F.from_unixtime(
            F.col("raw_time").cast("long")
        ).cast("timestamp")
    )

    df = df.withColumn(
        "review_date",
        F.to_date(F.col("review_timestamp"))
    )

    df = df.withColumn(
        "review_time",
        F.date_format(
            F.col("review_timestamp"),
            "yyyy-MM-dd HH:mm:ss"
        )
    )

    df = df.withColumn(
        "label",
        F.when(F.col("Score") < 3, "negative")
         .when(F.col("Score") == 3, "neutral")
         .otherwise("positive")
    )

    df = df.fillna(
        {
            "summary": ""
        }
    )

    return df.select(
        "product_id",
        "user_id",
        "review_time",
        "review_date",
        "Score",
        "summary",
        "text",
        "label"
    )


def build_streaming_export_df(clean_df, test_df):
    """
    Build the final streaming export dataset.

    Normal rule:
    - Use the real 10% test split for online prediction simulation.

    Requirement-compliance rule:
    - The project specification asks for dashboard analytics for ProductId B001E4KFG0.
    - If this ProductId is not naturally present in the test split, append a limited sample
      from the clean dataset and mark it as source_split='product_demo'.

    This keeps the export honest:
    - source_split='test' means real test split row.
    - source_split='product_demo' means row added only for the required ProductId dashboard section.
    """

    test_export_df = test_df.withColumn(
        "source_split",
        F.lit("test")
    )

    product_count_in_test = test_export_df.filter(
        F.col("product_id") == REQUIRED_DASHBOARD_PRODUCT_ID
    ).count()

    print("========== DASHBOARD PRODUCTID REQUIREMENT CHECK ==========")
    print(f"Required ProductId: {REQUIRED_DASHBOARD_PRODUCT_ID}")
    print(
        f"Rows in test split for ProductId {REQUIRED_DASHBOARD_PRODUCT_ID}: "
        f"{product_count_in_test}"
    )

    if product_count_in_test > 0:
        print("Required ProductId already exists in the test split.")
        return test_export_df

    product_demo_df = clean_df.filter(
        F.col("product_id") == REQUIRED_DASHBOARD_PRODUCT_ID
    ).limit(MAX_PRODUCT_DEMO_ROWS)

    product_demo_count = product_demo_df.count()

    print(
        f"Rows available in clean dataset for ProductId "
        f"{REQUIRED_DASHBOARD_PRODUCT_ID}: {product_demo_count}"
    )

    if product_demo_count == 0:
        print(
            "WARNING: Required ProductId was not found in the clean dataset. "
            "Export will contain only the test split."
        )
        return test_export_df

    product_demo_df = product_demo_df.withColumn(
        "source_split",
        F.lit("product_demo")
    )

    export_df = test_export_df.unionByName(product_demo_df)

    print(
        f"Added {product_demo_count} product_demo rows for "
        f"ProductId {REQUIRED_DASHBOARD_PRODUCT_ID}."
    )

    return export_df


def write_jsonl(export_df, spark):
    os.makedirs("data/processed", exist_ok=True)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    if os.path.exists(FINAL_JSONL_PATH):
        os.remove(FINAL_JSONL_PATH)

    print("========== WRITING STREAMING EXPORT TO JSONL ==========")

    export_df.select(
        F.col("product_id").alias("product_id"),
        F.col("user_id").alias("user_id"),
        F.col("review_time").alias("review_time"),
        F.col("review_date").cast("string").alias("review_date"),
        F.col("Score").alias("score"),
        F.col("summary").alias("summary"),
        F.col("text").alias("text"),
        F.col("label").alias("label"),
        F.col("source_split").alias("source_split")
    ).coalesce(1).write.mode("overwrite").json(OUTPUT_DIR)

    part_files = glob.glob(f"{OUTPUT_DIR}/part-*.json")

    if not part_files:
        raise FileNotFoundError("No Spark JSON part file found.")

    shutil.copy(part_files[0], FINAL_JSONL_PATH)

    print("========== EXPORT COMPLETE ==========")
    print(f"Final file: {FINAL_JSONL_PATH}")

    print("========== EXPORTED JSONL SAMPLE ==========")
    exported_df = spark.read.json(FINAL_JSONL_PATH)

    exported_df.select(
        "product_id",
        "user_id",
        "review_time",
        "review_date",
        "score",
        "summary",
        "text",
        "label",
        "source_split"
    ).show(5, truncate=80)

    print("========== EXPORTED SOURCE SPLIT DISTRIBUTION ==========")
    exported_df.groupBy("source_split").count().show()

    print("========== EXPORTED PRODUCTID REQUIREMENT CHECK ==========")
    exported_product_count = exported_df.filter(
        F.col("product_id") == REQUIRED_DASHBOARD_PRODUCT_ID
    ).count()

    print(
        f"Rows exported for ProductId {REQUIRED_DASHBOARD_PRODUCT_ID}: "
        f"{exported_product_count}"
    )


def main():
    spark = create_spark_session()

    print("========== EXPORTING TEST SPLIT FOR STREAMING ==========")

    clean_df = load_clean_raw_data(spark)

    print("========== CLEAN DATA CHECK BEFORE SPLIT ==========")
    print(f"Clean rows: {clean_df.count()}")

    print("Score distribution:")
    clean_df.groupBy("Score").count().orderBy("Score").show()

    print("Label distribution:")
    clean_df.groupBy("label").count().show()

    print("ProductId check:")
    clean_df.select("product_id").show(5, truncate=False)

    print("Review date check:")
    clean_df.select("review_time", "review_date").show(5, truncate=False)

    train_df, val_df, test_df = split_dataset(clean_df)

    print("========== TEST SPLIT CHECK ==========")
    print(f"Test rows: {test_df.count()}")

    print("Test score distribution:")
    test_df.groupBy("Score").count().orderBy("Score").show()

    print("Test label distribution:")
    test_df.groupBy("label").count().show()

    print("Test ProductId check:")
    test_df.select("product_id").show(5, truncate=False)

    print("Test review date check:")
    test_df.select("review_time", "review_date").show(5, truncate=False)

    export_df = build_streaming_export_df(
        clean_df=clean_df,
        test_df=test_df
    )

    print("========== FINAL STREAMING EXPORT CHECK ==========")
    print(f"Final export rows: {export_df.count()}")

    write_jsonl(
        export_df=export_df,
        spark=spark
    )

    spark.stop()


if __name__ == "__main__":
    main()
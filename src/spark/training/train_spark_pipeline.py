from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    StringIndexer,
    NGram,
    VectorAssembler
)
from pyspark.ml.classification import LogisticRegression


DATA_PATH = "data/raw/Reviews.csv"
MODEL_OUTPUT_PATH = "src/spark/model/sentiment_pipeline_model"

SAMPLE_SIZE = None # means full-data

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42


def create_spark_session():
    spark = SparkSession.builder \
        .appName("AmazonReviewsSparkPipelineTraining") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    return spark


def load_raw_data(spark):
    df = spark.read.csv(
        DATA_PATH,
        header=True,
        inferSchema=True
    )

    df = df.select("Text", "Score").dropna()

    if SAMPLE_SIZE is not None:
        df = df.limit(SAMPLE_SIZE)

    return df


def add_label_column(df):
    df = df.withColumn(
        "label",
        F.when(F.col("Score") < 3, "negative")
         .when(F.col("Score") == 3, "neutral")
         .otherwise("positive")
    )

    df = df.withColumnRenamed("Text", "text")

    return df.select("text", "Score", "label")


def split_dataset(df):
    df = df.withColumn("rand", F.rand(seed=RANDOM_SEED))

    train_df = df.filter(F.col("rand") < TRAIN_RATIO)

    val_df = df.filter(
        (F.col("rand") >= TRAIN_RATIO) &
        (F.col("rand") < TRAIN_RATIO + VAL_RATIO)
    )

    test_df = df.filter(F.col("rand") >= TRAIN_RATIO + VAL_RATIO)

    train_df = train_df.drop("rand")
    val_df = val_df.drop("rand")
    test_df = test_df.drop("rand")

    print("========== SPLIT SIZES ==========")
    print(f"Train rows: {train_df.count()}")
    print(f"Validation rows: {val_df.count()}")
    print(f"Test rows: {test_df.count()}")

    print("========== TRAIN CLASS DISTRIBUTION ==========")
    train_df.groupBy("label").count().show()

    print("========== VALIDATION CLASS DISTRIBUTION ==========")
    val_df.groupBy("label").count().show()

    print("========== TEST CLASS DISTRIBUTION ==========")
    test_df.groupBy("label").count().show()

    return train_df, val_df, test_df


def add_class_weights(train_df, val_df, test_df):
    total_count = train_df.count()

    class_counts = train_df.groupBy("label").count().collect()
    num_classes = len(class_counts)

    class_weights = {}

    for row in class_counts:
        label = row["label"]
        count = row["count"]

        weight = total_count / (num_classes * count)
        class_weights[label] = weight

    print("========== CLASS WEIGHTS ==========")
    for label, weight in class_weights.items():
        print(f"{label} -> {weight:.4f}")

    weight_expr = F.when(
        F.col("label") == list(class_weights.keys())[0],
        list(class_weights.values())[0]
    )

    for label, weight in list(class_weights.items())[1:]:
        weight_expr = weight_expr.when(
            F.col("label") == label,
            weight
        )

    train_df = train_df.withColumn("class_weight", weight_expr)

    # Validation and test do not need weights for evaluation,
    # but adding the column keeps schemas consistent.
    val_df = val_df.withColumn("class_weight", F.lit(1.0))
    test_df = test_df.withColumn("class_weight", F.lit(1.0))

    return train_df, val_df, test_df


def build_pipeline(
    vocab_size=20000,
    min_df=2,
    max_iter=30,
    reg_param=0.0001,
    use_bigrams=False
):
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="words",
        pattern="\\W+",
        toLowercase=True
    )

    remover = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words"
    )

    label_indexer = StringIndexer(
        inputCol="label",
        outputCol="label_index"
    )

    logistic_regression = LogisticRegression(
        featuresCol="features",
        labelCol="label_index",
        weightCol="class_weight",
        predictionCol="prediction",
        probabilityCol="probability",
        maxIter=max_iter,
        regParam=reg_param
    )

    if not use_bigrams:
        count_vectorizer = CountVectorizer(
            inputCol="filtered_words",
            outputCol="raw_features",
            vocabSize=vocab_size,
            minDF=min_df
        )

        idf = IDF(
            inputCol="raw_features",
            outputCol="features"
        )

        pipeline = Pipeline(stages=[
            tokenizer,
            remover,
            count_vectorizer,
            idf,
            label_indexer,
            logistic_regression
        ])

        return pipeline

    ngram = NGram(
        n=2,
        inputCol="filtered_words",
        outputCol="bigrams"
    )

    unigram_vectorizer = CountVectorizer(
        inputCol="filtered_words",
        outputCol="unigram_raw_features",
        vocabSize=vocab_size,
        minDF=min_df
    )

    unigram_idf = IDF(
        inputCol="unigram_raw_features",
        outputCol="unigram_features"
    )

    bigram_vectorizer = CountVectorizer(
        inputCol="bigrams",
        outputCol="bigram_raw_features",
        vocabSize=vocab_size,
        minDF=min_df
    )

    bigram_idf = IDF(
        inputCol="bigram_raw_features",
        outputCol="bigram_features"
    )

    assembler = VectorAssembler(
        inputCols=["unigram_features", "bigram_features"],
        outputCol="features"
    )

    pipeline = Pipeline(stages=[
        tokenizer,
        remover,
        ngram,

        unigram_vectorizer,
        unigram_idf,

        bigram_vectorizer,
        bigram_idf,

        assembler,
        label_indexer,
        logistic_regression
    ])

    return pipeline


def extract_metrics(predictions):
    total = predictions.count()

    correct = predictions.filter(
        F.col("label_index") == F.col("prediction")
    ).count()

    accuracy = correct / total if total > 0 else 0.0

    confusion_matrix = predictions.groupBy(
        "label_index",
        "prediction"
    ).count()

    actual_totals = confusion_matrix.groupBy("label_index") \
        .agg(F.sum("count").alias("actual_total"))

    predicted_totals = confusion_matrix.groupBy("prediction") \
        .agg(F.sum("count").alias("predicted_total"))

    true_positives = confusion_matrix.filter(
        F.col("label_index") == F.col("prediction")
    ).select(
        F.col("label_index").alias("class_index"),
        F.col("count").alias("true_positive")
    )

    metrics = true_positives \
        .join(
            actual_totals,
            true_positives["class_index"] == actual_totals["label_index"],
            "inner"
        ) \
        .join(
            predicted_totals,
            true_positives["class_index"] == predicted_totals["prediction"],
            "inner"
        ) \
        .select(
            "class_index",
            "true_positive",
            "actual_total",
            "predicted_total"
        )

    metrics = metrics.withColumn(
        "precision",
        F.col("true_positive") / F.col("predicted_total")
    ).withColumn(
        "recall",
        F.col("true_positive") / F.col("actual_total")
    ).withColumn(
        "f1_score",
        2 * (F.col("precision") * F.col("recall")) /
        (F.col("precision") + F.col("recall"))
    )

    rows = metrics.collect()

    per_class = {
        "0.0": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "1.0": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "2.0": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    }

    for row in rows:
        class_key = str(float(row["class_index"]))
        per_class[class_key] = {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1_score"]),
        }

    macro_f1 = (
        per_class["0.0"]["f1"] +
        per_class["1.0"]["f1"] +
        per_class["2.0"]["f1"]
    ) / 3

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,

        "positive_precision": per_class["0.0"]["precision"],
        "positive_recall": per_class["0.0"]["recall"],
        "positive_f1": per_class["0.0"]["f1"],

        "negative_precision": per_class["1.0"]["precision"],
        "negative_recall": per_class["1.0"]["recall"],
        "negative_f1": per_class["1.0"]["f1"],

        "neutral_precision": per_class["2.0"]["precision"],
        "neutral_recall": per_class["2.0"]["recall"],
        "neutral_f1": per_class["2.0"]["f1"],
    }


def main():
    spark = create_spark_session()

    df = load_raw_data(spark)
    df = add_label_column(df)

    print("========== FULL DATA CLASS DISTRIBUTION ==========")
    df.groupBy("label").count().show()

    train_df, val_df, test_df = split_dataset(df)
    train_df, val_df, test_df = add_class_weights(train_df, val_df, test_df)

    # Best parameters found yet with SA ranking
    print("========== TRAINING PIPELINE ON TRAIN DATA ONLY ==========")

    pipeline = build_pipeline(
        vocab_size=10000,
        min_df=4,
        max_iter=15,
        reg_param=0.000025,
        use_bigrams=True
    )

    pipeline_model = pipeline.fit(train_df)

    print("========== PIPELINE TRAINED SUCCESSFULLY ==========")

    print("========== SAVING PIPELINE MODEL ==========")
    pipeline_model.write().overwrite().save(MODEL_OUTPUT_PATH)
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")

    print("========== VALIDATION PREDICTIONS ==========")
    val_predictions = pipeline_model.transform(val_df)

    val_metrics = extract_metrics(val_predictions)

    print("========== VALIDATION METRICS ==========")
    print(f"Accuracy:    {val_metrics['accuracy']:.4f}")
    print(f"Macro F1:    {val_metrics['macro_f1']:.4f}")
    print(f"Positive F1: {val_metrics['positive_f1']:.4f}")
    print(f"Negative F1: {val_metrics['negative_f1']:.4f}")
    print(f"Neutral F1:  {val_metrics['neutral_f1']:.4f}")

    print("========== TEST PREDICTIONS ==========")
    test_predictions = pipeline_model.transform(test_df)

    test_metrics = extract_metrics(test_predictions)

    print("========== TEST METRICS ==========")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"Positive F1: {test_metrics['positive_f1']:.4f}")
    print(f"Negative F1: {test_metrics['negative_f1']:.4f}")
    print(f"Neutral F1:  {test_metrics['neutral_f1']:.4f}")

    # Important:
    # We created test_df, but we do not evaluate it yet.
    # Test data must stay untouched until final model selection.

    spark.stop()


if __name__ == "__main__":
    main()


"""
SMART NOTES — PIPELINE VERSION

1. Why split before fitting?

   We split before fitting because these stages learn from data:
   - CountVectorizer learns vocabulary
   - IDF learns word importance weights
   - StringIndexer learns label mapping
   - LogisticRegression learns classification weights

   If we fit before splitting, validation and test data influence the learned vocabulary,
   IDF values, and model weights. That is data leakage.

2. Correct ML order:

   load data
   -> create labels
   -> split train / validation / test
   -> fit pipeline on train only
   -> transform validation
   -> tune using validation
   -> test once at the end

3. What does Pipeline do?

   Pipeline groups all stages together:

   RegexTokenizer
   -> StopWordsRemover
   -> CountVectorizer
   -> IDF
   -> StringIndexer
   -> LogisticRegression

   Instead of manually calling fit/transform for every stage,
   we call:

   pipeline_model = pipeline.fit(train_df)

4. What happens inside pipeline.fit(train_df)?

   Spark executes each stage in order.

   RegexTokenizer:
   - transforms text into words
   - no fitting needed

   StopWordsRemover:
   - removes common words
   - no fitting needed

   CountVectorizer:
   - fits on train_df only
   - learns vocabulary
   - creates raw_features

   IDF:
   - fits on train_df only
   - learns IDF weights
   - creates features

   StringIndexer:
   - fits on train_df only
   - learns label -> label_index mapping

   LogisticRegression:
   - fits on train_df only
   - learns classification weights

5. What does pipeline_model.transform(val_df) do?

   It applies the already learned transformations to validation data:

   validation text
   -> words
   -> filtered_words
   -> raw_features using train vocabulary
   -> features using train IDF weights
   -> label_index using train label mapping
   -> prediction and probability using trained LogisticRegression

6. Why do we not evaluate test_df now?

   Because test data is the final exam.
   We only use validation while improving/tuning the model.
   Test is used once after final model selection.

7. Current SAMPLE_SIZE

   SAMPLE_SIZE = 10000 is for learning and debugging.

   Later:
   SAMPLE_SIZE = None

   This will use the full dataset.
"""
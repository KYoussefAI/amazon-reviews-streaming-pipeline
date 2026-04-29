from src.experiments.preprocessing.dataset import load_dataset
from src.experiments.preprocessing.vectorizer import fit_transform, transform
from src.experiments.preprocessing.resampling import undersample,oversample
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("Training") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()


def convert_to_spark_vectors(X):
    vectors = []

    for i in range(X.shape[0]):
        row = X[i].tocoo()

        indices = row.col.tolist()
        values = row.data.tolist()

        sorted_pairs = sorted(zip(indices, values))  # cuz spark needs indices sorted (increasing)
        indices, values = zip(*sorted_pairs) if sorted_pairs else ([], [])

        vec = Vectors.sparse(X.shape[1], list(indices), list(values))
        vectors.append(vec)

    return vectors


# 1. load data
x_train, x_val, x_test, y_train, y_val, y_test = load_dataset()
# x_train, y_train = oversample(x_train, y_train)  # 75%
# x_train, y_train = undersample(x_train, y_train)
X_train_vec = fit_transform(x_train)
x_val_vec = transform(x_val)
x_test_vec = transform(x_test)

# 2. convert features
train_vectors = convert_to_spark_vectors(X_train_vec)
val_vectors = convert_to_spark_vectors(x_val_vec)
test_vectors = convert_to_spark_vectors(x_test_vec)

# 3. create DataFrame
train_df = spark.createDataFrame(list(zip(train_vectors, y_train)), ["features", "label"])
val_df = spark.createDataFrame(list(zip(val_vectors, y_val)), ["features", "label"])
test_df = spark.createDataFrame(list(zip(test_vectors, y_test)), ["features", "label"])

# 4. Create model
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label_index",
    maxIter=10
)

indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexer_model = indexer.fit(train_df)
train_df = indexer_model.transform(train_df)
val_df = indexer_model.transform(val_df)
test_df = indexer_model.transform(test_df)
# train_df.select("label", "label_index").show(5)
model = lr.fit(train_df)
print("Model trained successfully")
val_df.groupBy("label").count().show()

val_predictions = model.transform(val_df)
test_predictions = model.transform(test_df)
# val_predictions.select("features", "label", "label_index", "prediction").show(5)

# confusion matrix
cm = val_predictions.groupBy("label_index", "prediction").count()


def metrics_val_test(total, correct):
    accuracy = correct / total

    print("==================================")
    print(f"Validation Accuracy: {accuracy}")
    print("==================================")

    # total actual per class (row sum)
    actual_totals = cm.groupBy("label_index") \
        .agg(F.sum("count").alias("actual_total"))

    # total predicted per class (column sum)
    pred_totals = cm.groupBy("prediction") \
        .agg(F.sum("count").alias("pred_total"))

    # true positives (diagonal)
    tp = cm.filter(F.col("label_index") == F.col("prediction")) \
        .select(
            F.col("label_index").alias("class"),
            F.col("count").alias("tp")
        )

    # join everything
    metrics = tp \
        .join(actual_totals, tp["class"] == actual_totals["label_index"]) \
        .join(pred_totals, tp["class"] == pred_totals["prediction"]) \
        .select(
            "class",
            "tp",
            "actual_total",
            "pred_total"
        )

    # compute metrics
    metrics = metrics.withColumn(
        "precision", F.col("tp") / F.col("pred_total")
    ).withColumn(
        "recall", F.col("tp") / F.col("actual_total")
    ).withColumn(
        "f1",
        2 * (F.col("precision") * F.col("recall")) /
        (F.col("precision") + F.col("recall"))
    )

    metrics.show()
    print("==================================")
    print("END[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]")
    print("==================================")
    spark.stop()

# Used for tunning
# total_val = val_predictions.count()
# correct_val = val_predictions.filter(
#     F.col("label_index") == F.col("prediction")
# ).count()


total_test = test_predictions.count()
correct_test = test_predictions.filter(
    F.col("label_index") == F.col("prediction")
).count()

metrics_val_test(total_test, correct_test)

# Phase : Tunning & Validation - ~10k records - 10% for validation
# Unmodified Dataset                                                      
# +--------+-----+                                                                
# |   label|count|
# +--------+-----+
# |positive|  761|
# | neutral|   86|
# |negative|  153|
# +--------+-----+

# ==================================                                              
# Validation Accuracy: 0.803
# ==================================
# +-----+---+------------+----------+------------------+------------------+------------------+
# |class| tp|actual_total|pred_total|         precision|            recall|                f1|
# +-----+---+------------+----------+------------------+------------------+------------------+
# |  0.0|691|         761|       785|0.8802547770700637|0.9080157687253614|0.8939197930142303|
# |  1.0| 90|         153|       153|0.5882352941176471|0.5882352941176471|0.5882352941176471|
# |  2.0| 22|          86|        62|0.3548387096774194|0.2558139534883721|0.2972972972972973|
# +-----+---+------------+----------+------------------+------------------+------------------+


# Oversampling 100% - bad
# +--------+-----+                                                                
# |   label|count|
# +--------+-----+
# |positive|  762|
# | neutral|   86|
# |negative|  152|
# +--------+-----+

# ==================================                                              
# Validation Accuracy: 0.793
# ==================================
# +-----+---+------------+----------+------------------+-------------------+------------------+
# |class| tp|actual_total|pred_total|         precision|             recall|                f1|
# +-----+---+------------+----------+------------------+-------------------+------------------+
# |  0.0| 77|         152|       130|0.5923076923076923|  0.506578947368421|0.5460992907801419|
# |  1.0| 24|          86|        72|0.3333333333333333|0.27906976744186046|0.3037974683544304|
# |  2.0|696|         762|       798|0.8721804511278195| 0.9133858267716536|0.8923076923076924|
# +-----+---+------------+----------+------------------+-------------------+------------------+


# Undersampling 75% didn't improve noticibaly
# +--------+-----+                                                                
# |   label|count|
# +--------+-----+
# |positive| 1541|
# | neutral|  165|
# |negative|  294|
# +--------+-----+

# ==================================                                              
# Validation Accuracy: 0.7605
# ==================================
# +-----+----+------------+----------+------------------+------------------+------------------+
# |class|  tp|actual_total|pred_total|         precision|            recall|                f1|
# +-----+----+------------+----------+------------------+------------------+------------------+
# |  0.0|1290|        1541|      1436|0.8983286908077994|0.8371187540558079|0.8666442727578098|
# |  1.0| 170|         294|       332|0.5120481927710844|0.5782312925170068|0.5431309904153354|
# |  2.0|  61|         165|       232|0.2629310344827586|0.3696969696969697|0.3073047858942065|
# +-----+----+------------+----------+------------------+------------------+------------------+


# Oversampling 75% - Sweet spot so far 
# +--------+-----+
# |   label|count|
# +--------+-----+
# |positive|  761|
# | neutral|   86|
# |negative|  153|
# +--------+-----+

# ==================================
# Validation Accuracy: 0.8
# ==================================
# +-----+---+------------+----------+-------------------+------------------+-------------------+
# |class| tp|actual_total|pred_total|          precision|            recall|                 f1|
# +-----+---+------------+----------+-------------------+------------------+-------------------+
# |  0.0|701|         761|       809| 0.8665018541409147|0.9211563731931669| 0.8929936305732484|
# |  1.0| 73|         153|       114| 0.6403508771929824| 0.477124183006536| 0.5468164794007491|
# |  2.0| 26|          86|        77|0.33766233766233766|0.3023255813953488|0.31901840490797545|
# +-----+---+------------+----------+-------------------+------------------+-------------------+

# Final TESTING : ~10K RECORDS - 10% for testing
# +--------+-----+                                                                
# |   label|count|
# +--------+-----+
# |positive|  761|
# | neutral|   86|
# |negative|  153|
# +--------+-----+

# ==================================                                              
# Validation Accuracy: 0.796
# ==================================
# +-----+---+------------+----------+-------------------+-------------------+------------------+
# |class| tp|actual_total|pred_total|          precision|             recall|                f1|
# +-----+---+------------+----------+-------------------+-------------------+------------------+
# |  0.0|685|         761|       767| 0.8930899608865711|  0.900131406044678|0.8965968586387434|
# |  1.0| 99|         153|       168| 0.5892857142857143| 0.6470588235294118|0.6168224299065421|
# |  2.0| 24|          86|        65|0.36923076923076925|0.27906976744186046|0.3178807947019867|
# +-----+---+------------+----------+-------------------+-------------------+------------------+
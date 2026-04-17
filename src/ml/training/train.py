from src.preprocessing.dataset import load_dataset
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer


spark = SparkSession.builder \
    .appName("Training") \
    .config("spark.driver.memory", "4g") \
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
X, y = load_dataset()

# 2. convert features
spark_vectors = convert_to_spark_vectors(X)

# 3. combine (THIS LINE)
data = list(zip(spark_vectors, y))

# 4. create DataFrame
df_spark = spark.createDataFrame(data, ["features", "label"])

# 5. Train / Test split
train_df, test_df = df_spark.randomSplit([0.8, 0.2], seed=42)

# 6. Create model
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label_index",
    maxIter=10
)

indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexer_model = indexer.fit(train_df)
train_df = indexer_model.transform(train_df)
test_df = indexer_model.transform(test_df)
train_df.select("label", "label_index").show(5)
model = lr.fit(train_df)
print("Model trained successfully")

predictions = model.transform(test_df)
predictions.select("features", "label", "label_index", "prediction").show(5)

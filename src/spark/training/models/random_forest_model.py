from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier

from src.spark.training.train_spark_pipeline import RANDOM_SEED
from src.spark.training.models.feature_pipeline import (
    build_text_feature_stages,
    build_label_indexer,
)


MODEL_NAME = "random_forest_light"


def build_model_pipeline():
    """
    Light Random Forest comparison model.

    Important:
        Random Forest is not usually ideal for very high-dimensional sparse TF-IDF
        text vectors. We keep the configuration small to make the comparison practical.
    """

    classifier = RandomForestClassifier(
        featuresCol="features",
        labelCol="label_index",
        predictionCol="prediction",
        probabilityCol="probability",
        numTrees=20,
        maxDepth=8,
        seed=RANDOM_SEED,
    )

    return Pipeline(
        stages=[
            *build_text_feature_stages(
                vocab_size=5000,
                min_df=5,
                use_bigrams=False,
            ),
            build_label_indexer(),
            classifier,
        ]
    )


def get_model_config():
    return {
        "model_name": MODEL_NAME,
        "builder": build_model_pipeline,
        "saveable_pipeline": True,
        "supports_probability": True,
        "notes": "Light Random Forest comparison; less ideal for sparse TF-IDF text.",
    }

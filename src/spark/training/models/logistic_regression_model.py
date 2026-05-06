from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

from src.spark.training.models.feature_pipeline import (
    build_text_feature_stages,
    build_label_indexer,
)


MODEL_NAME = "logistic_regression"


def build_model_pipeline():
    """
    Current strong baseline model.

    Good for:
        - sparse TF-IDF text features
        - multiclass sentiment classification
        - Spark Structured Streaming deployment as a saved PipelineModel
    """

    classifier = LogisticRegression(
        featuresCol="features",
        labelCol="label_index",
        weightCol="class_weight",
        predictionCol="prediction",
        probabilityCol="probability",
        maxIter=15,
        regParam=0.000025,
    )

    return Pipeline(
        stages=[
            *build_text_feature_stages(
                vocab_size=10000,
                min_df=4,
                use_bigrams=True,
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
        "notes": "Strong baseline for sparse TF-IDF text classification.",
    }

from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC, OneVsRest

from src.spark.training.models.feature_pipeline import (
    build_text_feature_stages,
    build_label_indexer,
)


MODEL_NAME = "one_vs_rest_linear_svc"


def build_model_pipeline():
    """
    Linear Support Vector Classifier adapted to multiclass using OneVsRest.

    Good for:
        - sparse text features
        - strong linear decision boundaries

    Limitation:
        - probability output is not as direct as Logistic Regression.
    """

    base_classifier = LinearSVC(
        featuresCol="features",
        labelCol="label_index",
        predictionCol="prediction",
        maxIter=20,
        regParam=0.0001,
    )

    classifier = OneVsRest(
        featuresCol="features",
        labelCol="label_index",
        predictionCol="prediction",
        classifier=base_classifier,
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
        "supports_probability": False,
        "notes": "Linear SVC adapted to multiclass using OneVsRest.",
    }

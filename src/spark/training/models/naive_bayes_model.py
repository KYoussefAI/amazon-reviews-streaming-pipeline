from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes

from src.spark.training.models.feature_pipeline import (
    build_text_feature_stages,
    build_label_indexer,
)


MODEL_NAME = "naive_bayes"


def build_model_pipeline():
    """
    Multinomial Naive Bayes.

    Good for:
        - fast text classification
        - sparse count/TF-IDF-style features

    Note:
        Naive Bayes in Spark does not use class_weight directly like Logistic Regression.
    """

    classifier = NaiveBayes(
        featuresCol="features",
        labelCol="label_index",
        predictionCol="prediction",
        probabilityCol="probability",
        modelType="multinomial",
        smoothing=1.0,
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
        "notes": "Fast classic text-classification baseline using multinomial Naive Bayes.",
    }

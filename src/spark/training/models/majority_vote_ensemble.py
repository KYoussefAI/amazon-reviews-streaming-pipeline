from functools import reduce

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from src.spark.training.train_spark_pipeline import extract_metrics


MODEL_NAME = "majority_vote_ensemble"


def prediction_column_for_model(predictions, model_name):
    return predictions.select(
        "row_id",
        "label",
        "label_index",
        F.col("prediction").alias(f"{model_name}_prediction"),
    )


def majority_vote_from_predictions(*values):
    clean_values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not clean_values:
        return None

    counts = {}

    for value in clean_values:
        counts[value] = counts.get(value, 0) + 1

    max_count = max(counts.values())

    tied_values = [
        value
        for value, count in counts.items()
        if count == max_count
    ]

    # Tie-break rule:
    # choose the first model prediction among tied classes.
    for value in clean_values:
        if value in tied_values:
            return value

    return tied_values[0]


def evaluate_majority_vote_ensemble(model_results, split_name):
    """
    Evaluate a batch majority-vote ensemble.

    This ensemble is useful for comparison and model-selection analysis.
    It is not automatically saved as one Spark PipelineModel for streaming.
    """

    prediction_dfs = []

    for result in model_results:
        predictions_key = f"{split_name}_predictions"

        if predictions_key not in result:
            continue

        prediction_dfs.append(
            prediction_column_for_model(
                predictions=result[predictions_key],
                model_name=result["model_name"],
            )
        )

    if len(prediction_dfs) < 2:
        return None

    ensemble_df = reduce(
        lambda left, right: left.join(
            right.drop("label", "label_index"),
            on="row_id",
            how="inner",
        ),
        prediction_dfs,
    )

    prediction_columns = [
        column
        for column in ensemble_df.columns
        if column.endswith("_prediction")
    ]

    majority_vote_udf = F.udf(
        majority_vote_from_predictions,
        DoubleType(),
    )

    ensemble_df = ensemble_df.withColumn(
        "prediction",
        majority_vote_udf(
            *[
                F.col(column)
                for column in prediction_columns
            ]
        ),
    )

    return extract_metrics(ensemble_df)

import os
import sys
import csv
import shutil
from datetime import datetime

from pyspark.sql import functions as F

PROJECT_ROOT = os.getcwd()

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spark.training.train_spark_pipeline import (
    create_spark_session,
    load_raw_data,
    add_label_column,
    split_dataset,
    add_class_weights,
    extract_metrics,
)

from src.spark.training.models.logistic_regression_model import (
    get_model_config as get_logistic_regression_config,
)
from src.spark.training.models.naive_bayes_model import (
    get_model_config as get_naive_bayes_config,
)
from src.spark.training.models.linear_svc_ovr_model import (
    get_model_config as get_linear_svc_ovr_config,
)
from src.spark.training.models.random_forest_model import (
    get_model_config as get_random_forest_config,
)
from src.spark.training.models.majority_vote_ensemble import (
    MODEL_NAME as ENSEMBLE_MODEL_NAME,
    evaluate_majority_vote_ensemble,
)


RESULTS_DIR = "results"
CSV_RESULTS_PATH = os.path.join(RESULTS_DIR, "model_comparison_results.csv")
MD_RESULTS_PATH = os.path.join(RESULTS_DIR, "model_comparison_results.md")

CANDIDATE_MODEL_PATH = "src/spark/model/candidate_sentiment_pipeline_model"
PRODUCTION_MODEL_PATH = "src/spark/model/sentiment_pipeline_model"

# Keep False for the first comparison run.
# Change to True only after reviewing results and deciding to replace
# the model used by Spark Structured Streaming.
SAVE_SELECTED_MODEL_TO_PRODUCTION = False

# Random Forest can be slow on text TF-IDF features.
RUN_RANDOM_FOREST = True


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def add_row_id(df):
    """
    Adds a stable row identifier for joining predictions from multiple models.

    This is required for the batch ensemble evaluation.
    """
    return df.withColumn(
        "row_id",
        F.monotonically_increasing_id()
    )


def get_model_configs():
    configs = [
        get_logistic_regression_config(),
        get_naive_bayes_config(),
        get_linear_svc_ovr_config(),
    ]

    if RUN_RANDOM_FOREST:
        configs.append(get_random_forest_config())

    return configs


def train_and_evaluate_model(train_df, val_df, model_config):
    model_name = model_config["model_name"]

    print(f"========== TRAINING MODEL: {model_name} ==========")

    pipeline = model_config["builder"]()
    model = pipeline.fit(train_df)

    print(f"========== VALIDATION PREDICTIONS: {model_name} ==========")
    val_predictions = model.transform(val_df)
    val_metrics = extract_metrics(val_predictions)

    return {
        "model_name": model_name,
        "model": model,
        "saveable_pipeline": model_config["saveable_pipeline"],
        "supports_probability": model_config["supports_probability"],
        "notes": model_config["notes"],
        "validation_predictions": val_predictions,
        "validation_metrics": val_metrics,
    }


def build_metric_row(
    model_name,
    split_name,
    metrics,
    notes="",
    selected=False,
    saveable_pipeline=True,
    supports_probability=True,
):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "split": split_name,
        "selected_by_validation": selected,
        "saveable_pipeline": saveable_pipeline,
        "supports_probability": supports_probability,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "positive_f1": metrics["positive_f1"],
        "negative_f1": metrics["negative_f1"],
        "neutral_f1": metrics["neutral_f1"],
        "positive_precision": metrics["positive_precision"],
        "positive_recall": metrics["positive_recall"],
        "negative_precision": metrics["negative_precision"],
        "negative_recall": metrics["negative_recall"],
        "neutral_precision": metrics["neutral_precision"],
        "neutral_recall": metrics["neutral_recall"],
        "notes": notes,
    }


def save_results_csv(rows):
    ensure_results_dir()

    fieldnames = [
        "timestamp",
        "model_name",
        "split",
        "selected_by_validation",
        "saveable_pipeline",
        "supports_probability",
        "accuracy",
        "macro_f1",
        "positive_f1",
        "negative_f1",
        "neutral_f1",
        "positive_precision",
        "positive_recall",
        "negative_precision",
        "negative_recall",
        "neutral_precision",
        "neutral_recall",
        "notes",
    ]

    with open(CSV_RESULTS_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results_markdown(rows, selected_validation_row, selected_test_row):
    ensure_results_dir()

    validation_rows = sorted(
        [
            row for row in rows
            if row["split"] == "validation"
        ],
        key=lambda row: row["macro_f1"],
        reverse=True,
    )

    with open(MD_RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write("# Spark Model Comparison Results\n\n")
        file.write(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n\n")

        file.write("## Selection Rule\n\n")
        file.write("All candidate models are trained on the training split.\n\n")
        file.write("Models are compared on the validation split using **Macro F1**.\n\n")
        file.write("The test split is used only once for the model selected by validation Macro F1.\n\n")

        file.write("## Validation Ranking\n\n")
        file.write("| Rank | Model | Accuracy | Macro F1 | Positive F1 | Negative F1 | Neutral F1 | Saveable Pipeline | Supports Probability | Selected |\n")
        file.write("|---:|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|\n")

        for rank, row in enumerate(validation_rows, start=1):
            file.write(
                f"| {rank} "
                f"| {row['model_name']} "
                f"| {row['accuracy']:.4f} "
                f"| {row['macro_f1']:.4f} "
                f"| {row['positive_f1']:.4f} "
                f"| {row['negative_f1']:.4f} "
                f"| {row['neutral_f1']:.4f} "
                f"| {row['saveable_pipeline']} "
                f"| {row['supports_probability']} "
                f"| {row['selected_by_validation']} |\n"
            )

        file.write("\n## Selected Model\n\n")
        file.write(f"Selected model: `{selected_validation_row['model_name']}`\n\n")
        file.write(f"Validation Macro F1: `{selected_validation_row['macro_f1']:.4f}`\n\n")

        file.write("## Final Test Metrics for Selected Model\n\n")
        file.write("| Metric | Value |\n")
        file.write("|---|---:|\n")
        file.write(f"| Accuracy | {selected_test_row['accuracy']:.4f} |\n")
        file.write(f"| Macro F1 | {selected_test_row['macro_f1']:.4f} |\n")
        file.write(f"| Positive F1 | {selected_test_row['positive_f1']:.4f} |\n")
        file.write(f"| Negative F1 | {selected_test_row['negative_f1']:.4f} |\n")
        file.write(f"| Neutral F1 | {selected_test_row['neutral_f1']:.4f} |\n")

        file.write("\n## Model Notes\n\n")
        for row in validation_rows:
            file.write(f"- `{row['model_name']}`: {row['notes']}\n")

        file.write("\n## Engineering Decision\n\n")
        file.write("Only a saveable Spark PipelineModel can directly replace the streaming model without changing `predict_stream.py`.\n")
        file.write("The majority-vote ensemble is evaluated as a batch comparison method first.\n")


def select_best_validation_model(rows):
    validation_rows = [
        row for row in rows
        if row["split"] == "validation"
    ]

    return sorted(
        validation_rows,
        key=lambda row: row["macro_f1"],
        reverse=True,
    )[0]


def save_selected_model(model_results, selected_model_name):
    if selected_model_name == ENSEMBLE_MODEL_NAME:
        print("========== ENSEMBLE SELECTED ==========")
        print("The majority-vote ensemble is a batch evaluation model.")
        print("It is not saved as one Spark PipelineModel for streaming yet.")
        return

    selected_result = next(
        result for result in model_results
        if result["model_name"] == selected_model_name
    )

    if os.path.exists(CANDIDATE_MODEL_PATH):
        shutil.rmtree(CANDIDATE_MODEL_PATH)

    print("========== SAVING CANDIDATE BEST MODEL ==========")
    selected_result["model"].write().overwrite().save(CANDIDATE_MODEL_PATH)
    print(f"Candidate model saved to: {CANDIDATE_MODEL_PATH}")

    if SAVE_SELECTED_MODEL_TO_PRODUCTION:
        if os.path.exists(PRODUCTION_MODEL_PATH):
            shutil.rmtree(PRODUCTION_MODEL_PATH)

        selected_result["model"].write().overwrite().save(PRODUCTION_MODEL_PATH)
        print(f"Production model replaced at: {PRODUCTION_MODEL_PATH}")
    else:
        print("Production model was NOT replaced.")
        print("Review comparison results before replacing the streaming model.")


def main():
    ensure_results_dir()

    spark = create_spark_session()

    print("========== LOADING AND PREPARING DATA ==========")

    raw_df = load_raw_data(spark)
    labeled_df = add_label_column(raw_df)

    train_df, val_df, test_df = split_dataset(labeled_df)

    train_df, val_df, test_df = add_class_weights(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    val_df = add_row_id(val_df).cache()
    test_df = add_row_id(test_df).cache()

    print(f"Validation rows with row_id: {val_df.count()}")
    print(f"Test rows with row_id: {test_df.count()}")

    model_configs = get_model_configs()

    model_results = []
    result_rows = []

    for model_config in model_configs:
        result = train_and_evaluate_model(
            train_df=train_df,
            val_df=val_df,
            model_config=model_config,
        )

        result_rows.append(
            build_metric_row(
                model_name=result["model_name"],
                split_name="validation",
                metrics=result["validation_metrics"],
                notes=result["notes"],
                selected=False,
                saveable_pipeline=result["saveable_pipeline"],
                supports_probability=result["supports_probability"],
            )
        )

        model_results.append(result)

    print("========== GENERATING TEST PREDICTIONS FOR CANDIDATE MODELS ==========")

    for result in model_results:
        print(f"Generating test predictions for: {result['model_name']}")
        result["test_predictions"] = result["model"].transform(test_df)

    print("========== EVALUATING MAJORITY VOTE ENSEMBLE ON VALIDATION ==========")

    ensemble_validation_metrics = evaluate_majority_vote_ensemble(
        model_results=model_results,
        split_name="validation",
    )

    if ensemble_validation_metrics is not None:
        result_rows.append(
            build_metric_row(
                model_name=ENSEMBLE_MODEL_NAME,
                split_name="validation",
                metrics=ensemble_validation_metrics,
                notes="Batch majority vote across trained model predictions.",
                selected=False,
                saveable_pipeline=False,
                supports_probability=False,
            )
        )

    selected_validation_row = select_best_validation_model(result_rows)
    selected_model_name = selected_validation_row["model_name"]

    for row in result_rows:
        if (
            row["split"] == "validation"
            and row["model_name"] == selected_model_name
        ):
            row["selected_by_validation"] = True

    print("========== SELECTED MODEL BY VALIDATION MACRO F1 ==========")
    print(f"Selected model: {selected_model_name}")
    print(f"Validation Macro F1: {selected_validation_row['macro_f1']:.4f}")

    print("========== FINAL TEST EVALUATION OF SELECTED MODEL ==========")

    if selected_model_name == ENSEMBLE_MODEL_NAME:
        selected_test_metrics = evaluate_majority_vote_ensemble(
            model_results=model_results,
            split_name="test",
        )

        selected_test_row = build_metric_row(
            model_name=selected_model_name,
            split_name="test",
            metrics=selected_test_metrics,
            notes="Final test evaluation after validation-based ensemble selection.",
            selected=True,
            saveable_pipeline=False,
            supports_probability=False,
        )
    else:
        selected_result = next(
            result for result in model_results
            if result["model_name"] == selected_model_name
        )

        selected_test_metrics = extract_metrics(
            selected_result["test_predictions"]
        )

        selected_test_row = build_metric_row(
            model_name=selected_model_name,
            split_name="test",
            metrics=selected_test_metrics,
            notes="Final test evaluation after validation-based model selection.",
            selected=True,
            saveable_pipeline=True,
            supports_probability=selected_result["supports_probability"],
        )

    result_rows.append(selected_test_row)

    print("========== SAVING COMPARISON RESULTS ==========")

    save_results_csv(result_rows)
    save_results_markdown(
        rows=result_rows,
        selected_validation_row=selected_validation_row,
        selected_test_row=selected_test_row,
    )

    print(f"CSV results saved to: {CSV_RESULTS_PATH}")
    print(f"Markdown results saved to: {MD_RESULTS_PATH}")

    save_selected_model(
        model_results=model_results,
        selected_model_name=selected_model_name,
    )

    print("========== MODEL COMPARISON COMPLETE ==========")
    print(f"Selected model: {selected_model_name}")
    print("Review results/model_comparison_results.md before replacing the production streaming model.")

    spark.stop()


if __name__ == "__main__":
    main()

import os
import sys
import shutil
from datetime import datetime

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


ENSEMBLE_MODEL_ROOT = "src/spark/model/ensemble_models"

MODELS_TO_TRAIN = [
    get_logistic_regression_config(),
    get_naive_bayes_config(),
    get_linear_svc_ovr_config(),
]


def clean_model_output_dir():
    if os.path.exists(ENSEMBLE_MODEL_ROOT):
        print(f"Removing old ensemble model folder: {ENSEMBLE_MODEL_ROOT}")
        shutil.rmtree(ENSEMBLE_MODEL_ROOT)

    os.makedirs(ENSEMBLE_MODEL_ROOT, exist_ok=True)


def save_model(model, model_name):
    output_path = os.path.join(
        ENSEMBLE_MODEL_ROOT,
        model_name
    )

    print(f"Saving {model_name} to: {output_path}")
    model.write().overwrite().save(output_path)


def write_ensemble_metadata(model_metrics):
    metadata_path = os.path.join(
        ENSEMBLE_MODEL_ROOT,
        "ensemble_metadata.md"
    )

    with open(metadata_path, "w", encoding="utf-8") as file:
        file.write("# Ensemble Models Metadata\n\n")
        file.write(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n\n")
        file.write("## Purpose\n\n")
        file.write("These models are trained separately and saved under one folder for streaming experiments and model-by-model deployment.\n\n")
        file.write("This script does not implement a streaming majority-vote scorer by itself; it only materializes compatible saved PipelineModel artifacts and records their validation metrics.\n\n")
        file.write("## Saved Models\n\n")
        file.write("| Model | Validation Accuracy | Validation Macro F1 | Positive F1 | Negative F1 | Neutral F1 |\n")
        file.write("|---|---:|---:|---:|---:|---:|\n")

        for row in model_metrics:
            file.write(
                f"| {row['model_name']} "
                f"| {row['accuracy']:.4f} "
                f"| {row['macro_f1']:.4f} "
                f"| {row['positive_f1']:.4f} "
                f"| {row['negative_f1']:.4f} "
                f"| {row['neutral_f1']:.4f} |\n"
            )

        file.write("\n## Saved Model Order\n\n")
        file.write("The models are saved in this order:\n\n")
        file.write("```text\n")
        file.write("logistic_regression\n")
        file.write("naive_bayes\n")
        file.write("one_vs_rest_linear_svc\n")
        file.write("```\n\n")
        file.write("If you later build a majority-vote streaming scorer, this order is the natural deterministic tie-break sequence.\n")

    print(f"Metadata written to: {metadata_path}")


def main():
    spark = create_spark_session()

    print("========== TRAINING ENSEMBLE MODELS FOR STREAMING ==========")

    raw_df = load_raw_data(spark)
    labeled_df = add_label_column(raw_df)

    train_df, val_df, test_df = split_dataset(labeled_df)

    train_df, val_df, test_df = add_class_weights(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    clean_model_output_dir()

    model_metrics = []

    for model_config in MODELS_TO_TRAIN:
        model_name = model_config["model_name"]

        print(f"========== TRAINING ENSEMBLE MEMBER: {model_name} ==========")

        pipeline = model_config["builder"]()
        model = pipeline.fit(train_df)

        print(f"========== VALIDATING ENSEMBLE MEMBER: {model_name} ==========")
        val_predictions = model.transform(val_df)
        metrics = extract_metrics(val_predictions)

        model_metrics.append(
            {
                "model_name": model_name,
                **metrics,
            }
        )

        save_model(
            model=model,
            model_name=model_name,
        )

    write_ensemble_metadata(model_metrics)

    print("========== ENSEMBLE MODELS TRAINING COMPLETE ==========")
    print(f"Saved ensemble models root: {ENSEMBLE_MODEL_ROOT}")

    spark.stop()


if __name__ == "__main__":
    main()

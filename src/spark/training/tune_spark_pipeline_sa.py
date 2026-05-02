import os
import sys
import csv
import math
import random
from datetime import datetime

PROJECT_ROOT = os.getcwd()

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spark.training.train_spark_pipeline import (
    RANDOM_SEED,
    SAMPLE_SIZE,
    create_spark_session,
    load_raw_data,
    add_label_column,
    split_dataset,
    add_class_weights,
    build_pipeline,
    extract_metrics,
)

RESULTS_DIR = "results"
CSV_RESULTS_PATH = os.path.join(RESULTS_DIR, "spark_sa_tuning_results.csv")
MD_RESULTS_PATH = os.path.join(RESULTS_DIR, "spark_sa_tuning_results.md")

VOCAB_OPTIONS = [10000, 12000, 14000]
MIN_DF_OPTIONS = [3, 4]
MAX_ITER_OPTIONS = [15, 20, 25]
REG_PARAM_OPTIONS = [0.0, 0.000025, 0.00005, 0.0001]
BIGRAM_OPTIONS = [True]

MAX_STEPS = 8
INITIAL_TEMPERATURE = 0.015
COOLING_RATE = 0.70

random.seed(RANDOM_SEED)


def initial_config():
    return {
        "vocab_size": 12000,
        "min_df": 3,
        "max_iter": 20,
        "reg_param": 0.00005,
        "use_bigrams": True,
    }


def neighbor_config(config):
    new_config = dict(config)

    parameter_to_change = random.choice([
        "vocab_size",
        "min_df",
        "max_iter",
        "reg_param",
    ])
    if parameter_to_change == "vocab_size":
        options = VOCAB_OPTIONS
    elif parameter_to_change == "min_df":
        options = MIN_DF_OPTIONS
    elif parameter_to_change == "max_iter":
        options = MAX_ITER_OPTIONS
    elif parameter_to_change == "reg_param":
        options = REG_PARAM_OPTIONS
    else:
        options = BIGRAM_OPTIONS

    current_value = new_config[parameter_to_change]
    current_index = options.index(current_value)

    possible_indexes = []

    if current_index > 0:
        possible_indexes.append(current_index - 1)

    if current_index < len(options) - 1:
        possible_indexes.append(current_index + 1)

    new_index = random.choice(possible_indexes)
    new_config[parameter_to_change] = options[new_index]

    return new_config


def acceptance_probability(current_score, candidate_score, temperature):
    if candidate_score >= current_score:
        return 1.0

    return math.exp((candidate_score - current_score) / temperature)


def train_and_score(train_df, val_df, config):
    pipeline = build_pipeline(
        vocab_size=config["vocab_size"],
        min_df=config["min_df"],
        max_iter=config["max_iter"],
        reg_param=config["reg_param"],
        use_bigrams=config["use_bigrams"],
    )

    pipeline_model = pipeline.fit(train_df)
    val_predictions = pipeline_model.transform(val_df)

    return extract_metrics(val_predictions)


def save_results_csv(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fieldnames = [
        "step",
        "timestamp",
        "accepted",
        "temperature",
        "sample_size",
        "vocab_size",
        "min_df",
        "max_iter",
        "reg_param",
        "use_bigrams",
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
    ]

    with open(CSV_RESULTS_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_results_markdown(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sorted_results = sorted(
        results,
        key=lambda row: row["macro_f1"],
        reverse=True
    )

    with open(MD_RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write("# Spark Pipeline Simulated Annealing Results\n\n")
        file.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n\n")
        file.write(f"Sample size: `{SAMPLE_SIZE}`\n\n")
        file.write("Ranking is sorted by `macro_f1` because the dataset is imbalanced.\n\n")

        file.write("| Rank | Step | Accepted | Temp | vocabSize | minDF | maxIter | regParam | useBigrams | Accuracy | Macro F1 | Positive F1 | Negative F1 | Neutral F1 |\n")
        file.write("|---:|---:|:---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|\n")

        for rank, row in enumerate(sorted_results, start=1):
            file.write(
                f"| {rank} "
                f"| {row['step']} "
                f"| {row['accepted']} "
                f"| {row['temperature']:.4f} "
                f"| {row['vocab_size']} "
                f"| {row['min_df']} "
                f"| {row['max_iter']} "
                f"| {row['reg_param']} "
                f"| {row['use_bigrams']} "
                f"| {row['accuracy']:.4f} "
                f"| {row['macro_f1']:.4f} "
                f"| {row['positive_f1']:.4f} "
                f"| {row['negative_f1']:.4f} "
                f"| {row['neutral_f1']:.4f} |\n"
            )


def record_result(step, accepted, temperature, config, metrics):
    return {
        "step": step,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "accepted": accepted,
        "temperature": temperature,
        "sample_size": SAMPLE_SIZE,
        **config,
        **metrics,
    }


def run_simulated_annealing(train_df, val_df):
    results = []

    current_config = initial_config()
    current_metrics = train_and_score(train_df, val_df, current_config)
    current_score = current_metrics["macro_f1"]

    best_config = dict(current_config)
    best_metrics = dict(current_metrics)
    best_score = current_score

    temperature = INITIAL_TEMPERATURE

    first_result = record_result(
        step=0,
        accepted=True,
        temperature=temperature,
        config=current_config,
        metrics=current_metrics,
    )

    results.append(first_result)
    save_results_csv(results)
    save_results_markdown(results)

    print("========== INITIAL CONFIG ==========")
    print(current_config)
    print(f"Initial Macro F1: {current_score:.4f}")

    for step in range(1, MAX_STEPS + 1):
        candidate_config = neighbor_config(current_config)

        print("\n" + "=" * 80)
        print(f"SA STEP {step}/{MAX_STEPS}")
        print(f"Temperature: {temperature:.4f}")
        print(f"Current config:   {current_config}")
        print(f"Candidate config: {candidate_config}")
        print("=" * 80)

        candidate_metrics = train_and_score(train_df, val_df, candidate_config)
        candidate_score = candidate_metrics["macro_f1"]

        probability = acceptance_probability(
            current_score=current_score,
            candidate_score=candidate_score,
            temperature=temperature,
        )

        random_value = random.random()
        accepted = random_value < probability

        print(f"Current Macro F1:   {current_score:.4f}")
        print(f"Candidate Macro F1: {candidate_score:.4f}")
        print(f"Acceptance prob:    {probability:.4f}")
        print(f"Random value:       {random_value:.4f}")
        print(f"Accepted:           {accepted}")

        if accepted:
            current_config = candidate_config
            current_metrics = candidate_metrics
            current_score = candidate_score

        if candidate_score > best_score:
            best_config = dict(candidate_config)
            best_metrics = dict(candidate_metrics)
            best_score = candidate_score

        results.append(
            record_result(
                step=step,
                accepted=accepted,
                temperature=temperature,
                config=candidate_config,
                metrics=candidate_metrics,
            )
        )

        save_results_csv(results)
        save_results_markdown(results)

        print("Candidate result:")
        print(f"Accuracy:    {candidate_metrics['accuracy']:.4f}")
        print(f"Macro F1:    {candidate_metrics['macro_f1']:.4f}")
        print(f"Positive F1: {candidate_metrics['positive_f1']:.4f}")
        print(f"Negative F1: {candidate_metrics['negative_f1']:.4f}")
        print(f"Neutral F1:  {candidate_metrics['neutral_f1']:.4f}")

        temperature *= COOLING_RATE

    return best_config, best_metrics, results


def main():
    spark = create_spark_session()

    df = load_raw_data(spark)
    df = add_label_column(df)

    print("========== FULL DATA CLASS DISTRIBUTION ==========")
    df.groupBy("label").count().show()

    train_df, val_df, test_df = split_dataset(df)
    train_df, val_df, test_df = add_class_weights(train_df, val_df, test_df)

    train_df = train_df.cache()
    val_df = val_df.cache()

    print("========== STARTING SIMULATED ANNEALING ==========")

    best_config, best_metrics, _ = run_simulated_annealing(
        train_df=train_df,
        val_df=val_df,
    )

    print("\n========== BEST CONFIG FOUND ==========")
    print(best_config)
    print(f"Accuracy:    {best_metrics['accuracy']:.4f}")
    print(f"Macro F1:    {best_metrics['macro_f1']:.4f}")
    print(f"Positive F1: {best_metrics['positive_f1']:.4f}")
    print(f"Negative F1: {best_metrics['negative_f1']:.4f}")
    print(f"Neutral F1:  {best_metrics['neutral_f1']:.4f}")

    print("\nSaved results to:")
    print(CSV_RESULTS_PATH)
    print(MD_RESULTS_PATH)

    # Test set remains untouched.
    spark.stop()


if __name__ == "__main__":
    main()
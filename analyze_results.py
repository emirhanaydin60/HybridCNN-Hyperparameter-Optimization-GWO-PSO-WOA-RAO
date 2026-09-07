import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from plot_colors import COLOR_MAP, COLOR_SEQUENCE
import numpy as np

from utils import ensure_dir, read_json, write_json


def load_summaries(results_dir, dataset):
    dataset_dir = os.path.join(results_dir, dataset.upper())
    summaries = {}
    for algorithm in ["GWO", "PSO", "WOA", "RAO"]:
        summary_path = os.path.join(dataset_dir, algorithm, "summary.json")
        if os.path.exists(summary_path):
            summaries[algorithm] = read_json(summary_path)
    return summaries


def flatten_records(summaries):
    rows = []
    for algorithm, payload in summaries.items():
        for record in payload.get("runs", []):
            rows.append(
                {
                    "algorithm": algorithm,
                    "best_validation_accuracy": record.get("best_validation_accuracy", 0.0),
                    "best_test_accuracy": record.get("best_test_accuracy", 0.0),
                    "precision": record.get("precision", 0.0),
                    "recall": record.get("recall", 0.0),
                    "f1": record.get("f1", 0.0),
                    "runtime_seconds": record.get("runtime_seconds", 0.0),
                    "fitness_evaluations": record.get("fitness_evaluations", 0),
                    "average_fitness_evaluation_duration": record.get("average_fitness_evaluation_duration", 0.0),
                }
            )
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["algorithm"]].append(row)

    summary = {}
    for algorithm, items in grouped.items():
        validation = np.array([float(item["best_validation_accuracy"]) for item in items], dtype=float)
        test = np.array([float(item["best_test_accuracy"]) for item in items], dtype=float)
        runtime = np.array([float(item["runtime_seconds"]) for item in items], dtype=float)
        f1 = np.array([float(item["f1"]) for item in items], dtype=float)
        summary[algorithm] = {
            "mean_validation_accuracy": float(np.mean(validation)) if len(validation) else 0.0,
            "std_validation_accuracy": float(np.std(validation, ddof=0)) if len(validation) else 0.0,
            "mean_test_accuracy": float(np.mean(test)) if len(test) else 0.0,
            "std_test_accuracy": float(np.std(test, ddof=0)) if len(test) else 0.0,
            "mean_runtime": float(np.mean(runtime)) if len(runtime) else 0.0,
            "std_runtime": float(np.std(runtime, ddof=0)) if len(runtime) else 0.0,
            "mean_f1": float(np.mean(f1)) if len(f1) else 0.0,
            "std_f1": float(np.std(f1, ddof=0)) if len(f1) else 0.0,
        }
    return summary


def aggregate_convergence(summaries):
    convergence = {}
    for algorithm, payload in summaries.items():
        histories = [run.get("convergence_history", []) for run in payload.get("runs", []) if run.get("convergence_history")]
        if not histories:
            continue
        max_length = max(len(history) for history in histories)
        padded = [history + [history[-1]] * (max_length - len(history)) for history in histories]
        mean_history = np.mean(np.asarray(padded, dtype=float), axis=0)
        convergence[algorithm] = mean_history.tolist()
    return convergence


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_metric(metric_map, key, ylabel, out_path):
    ensure_dir(os.path.dirname(out_path))
    algorithms = list(metric_map.keys())
    values = [metric_map[name][key] for name in algorithms]
    plt.figure(figsize=(8, 4))
    plt.bar(algorithms, values, color=COLOR_SEQUENCE)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_box(rows, key, ylabel, out_path):
    ensure_dir(os.path.dirname(out_path))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["algorithm"]].append(float(row[key]))
    labels = list(grouped.keys())
    data = [grouped[label] for label in labels]
    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="cifar10")
    args = parser.parse_args()

    summaries = load_summaries(args.results_dir, args.dataset)
    rows = flatten_records(summaries)
    aggregate_summary = aggregate(rows)
    convergence_summary = aggregate_convergence(summaries)

    comparison_dir = os.path.join(args.results_dir, args.dataset.upper(), "comparison")
    ensure_dir(comparison_dir)

    write_json(
        os.path.join(comparison_dir, "overall_summary.json"),
        {
            "dataset": args.dataset,
            "algorithms": list(summaries.keys()),
            "aggregate": aggregate_summary,
        },
    )

    write_csv(
        os.path.join(comparison_dir, "performance_comparison.csv"),
        rows,
        ["algorithm", "best_validation_accuracy", "best_test_accuracy", "precision", "recall", "f1", "runtime_seconds", "fitness_evaluations", "average_fitness_evaluation_duration"],
    )

    runtime_rows = [{"algorithm": algorithm, **metrics} for algorithm, metrics in aggregate_summary.items()]
    runtime_rows = [{"algorithm": algorithm, "mean_runtime": metrics["mean_runtime"], "std_runtime": metrics["std_runtime"]} for algorithm, metrics in aggregate_summary.items()]
    stability_rows = [
        {
            "algorithm": algorithm,
            "mean_validation_accuracy": metrics["mean_validation_accuracy"],
            "std_validation_accuracy": metrics["std_validation_accuracy"],
            "mean_test_accuracy": metrics["mean_test_accuracy"],
            "std_test_accuracy": metrics["std_test_accuracy"],
            "mean_f1": metrics["mean_f1"],
            "std_f1": metrics["std_f1"],
        }
        for algorithm, metrics in aggregate_summary.items()
    ]
    write_csv(os.path.join(comparison_dir, "runtime_comparison.csv"), runtime_rows, ["algorithm", "mean_runtime", "std_runtime"])
    write_csv(
        os.path.join(comparison_dir, "stability_comparison.csv"),
        stability_rows,
        ["algorithm", "mean_validation_accuracy", "std_validation_accuracy", "mean_test_accuracy", "std_test_accuracy", "mean_f1", "std_f1"],
    )

    convergence_rows = []
    for algorithm, series in convergence_summary.items():
        for iteration, value in enumerate(series, start=1):
            convergence_rows.append({"algorithm": algorithm, "iteration": iteration, "mean_global_best": value})
    if convergence_rows:
        write_csv(os.path.join(comparison_dir, "convergence_comparison.csv"), convergence_rows, ["algorithm", "iteration", "mean_global_best"])

    ranking_rows = sorted(aggregate_summary.items(), key=lambda item: item[1]["mean_test_accuracy"], reverse=True)
    write_csv(
        os.path.join(comparison_dir, "ranking_table.csv"),
        [{"rank": index + 1, "algorithm": algorithm, "mean_test_accuracy": metrics["mean_test_accuracy"]} for index, (algorithm, metrics) in enumerate(ranking_rows)],
        ["rank", "algorithm", "mean_test_accuracy"],
    )

    if aggregate_summary:
        plot_metric(aggregate_summary, "mean_test_accuracy", "Mean Test Accuracy", os.path.join(comparison_dir, "accuracy_comparison.png"))
        plot_metric(aggregate_summary, "mean_runtime", "Mean Runtime (s)", os.path.join(comparison_dir, "runtime_plot.png"))
        plot_box(rows, "best_test_accuracy", "Test Accuracy", os.path.join(comparison_dir, "boxplot_accuracy.png"))
        plot_box(rows, "runtime_seconds", "Runtime (s)", os.path.join(comparison_dir, "boxplot_runtime.png"))

    if convergence_summary:
        plt.figure(figsize=(9, 5))
        for algorithm, series in convergence_summary.items():
            color = COLOR_MAP.get(algorithm)
            plt.plot(range(1, len(series) + 1), series, label=algorithm, color=color)
        plt.xlabel("Iteration")
        plt.ylabel("Mean Global Best Fitness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "convergence_plot.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    main()

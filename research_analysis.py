import csv
import json
import math
import os
from collections import defaultdict
from statistics import NormalDist

import matplotlib

from plot_colors import COLOR_MAP

matplotlib.use("Agg")
import sys
import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir, read_json, write_json

ALGORITHMS = ["GWO", "PSO", "WOA", "RAO"]
ALPHA = 0.05


def _read_run_summary(run_dir):
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summary_path):
        return None
    payload = read_json(summary_path)
    payload["run_dir"] = run_dir
    return payload


def _collect_runs(results_dir, dataset):
    dataset_dir = os.path.join(results_dir, dataset.upper())
    runs_by_algorithm = {}
    for algorithm in ALGORITHMS:
        algorithm_dir = os.path.join(dataset_dir, algorithm)
        if not os.path.isdir(algorithm_dir):
            continue
        run_entries = []
        for entry in sorted(os.listdir(algorithm_dir)):
            if not entry.startswith("run_"):
                continue
            run_dir = os.path.join(algorithm_dir, entry)
            if os.path.isdir(run_dir):
                summary = _read_run_summary(run_dir)
                if summary is not None:
                    summary["run_name"] = entry
                    summary["run_index"] = int(entry.split("_", 1)[1]) if "_" in entry and entry.split("_", 1)[1].isdigit() else None
                    run_entries.append(summary)
        if run_entries:
            run_entries.sort(key=lambda item: item.get("run_index") or 0)
            runs_by_algorithm[algorithm] = run_entries
    return runs_by_algorithm


def _flat_best_rows(runs_by_algorithm):
    rows = []
    for algorithm, runs in runs_by_algorithm.items():
        for run in runs:
            best_config = run.get("best_hyperparameters", {})
            row = {
                "algorithm": algorithm,
                "run": run.get("run_name"),
                "seed": run.get("run_seed"),
                "best_validation_accuracy": run.get("best_validation_accuracy", 0.0),
                "best_test_accuracy": run.get("best_test_accuracy", 0.0),
                "precision": run.get("precision", 0.0),
                "recall": run.get("recall", 0.0),
                "f1": run.get("f1", 0.0),
                "runtime_seconds": run.get("runtime_seconds", 0.0),
                "total_search_time_seconds": run.get("total_search_time_seconds", 0.0),
                "final_training_time_seconds": run.get("final_training_time_seconds", 0.0),
                "fitness_evaluations": run.get("fitness_evaluations", 0),
                "average_fitness_evaluation_duration": run.get("average_fitness_evaluation_duration", 0.0),
            }
            row.update({f"hp_{key}": value for key, value in best_config.items()})
            rows.append(row)
    return rows


def _aggregate_algorithm_metrics(rows):
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
            "runs": len(items),
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


def _aggregate_algorithm_runs(runs_by_algorithm):
    rows = []
    for algorithm, runs in runs_by_algorithm.items():
        for run in runs:
            rows.append(
                {
                    "algorithm": algorithm,
                    "run": run.get("run_name"),
                    "run_index": run.get("run_index"),
                    "best_test_accuracy": run.get("best_test_accuracy", 0.0),
                    "best_validation_accuracy": run.get("best_validation_accuracy", 0.0),
                    "precision": run.get("precision", 0.0),
                    "recall": run.get("recall", 0.0),
                    "f1": run.get("f1", 0.0),
                    "runtime_seconds": run.get("runtime_seconds", 0.0),
                    "learning_rate": run.get("best_hyperparameters", {}).get("learning_rate"),
                    "dropout": run.get("best_hyperparameters", {}).get("dropout"),
                    "batch_size": run.get("best_hyperparameters", {}).get("batch_size"),
                    "final_neurons": run.get("best_hyperparameters", {}).get("final_neurons"),
                    "base_filters": run.get("best_hyperparameters", {}).get("base_filters"),
                    "shared_conv_kernel_size": run.get("best_hyperparameters", {}).get("shared_conv_kernel_size"),
                    "dilation": run.get("best_hyperparameters", {}).get("dilation"),
                    "se_ratio": run.get("best_hyperparameters", {}).get("se_ratio"),
                }
            )
    return rows


def _friedman_test(runs_by_algorithm):
    common_run_indices = None
    for runs in runs_by_algorithm.values():
        indices = {run["run_index"] for run in runs if run.get("run_index") is not None}
        common_run_indices = indices if common_run_indices is None else common_run_indices & indices
    common_run_indices = sorted(common_run_indices or [])

    aligned = []
    algorithms = []
    for algorithm in ALGORITHMS:
        runs = runs_by_algorithm.get(algorithm, [])
        lookup = {run.get("run_index"): run for run in runs}
        if all(run_index in lookup for run_index in common_run_indices):
            algorithms.append(algorithm)
            aligned.append([lookup[run_index]["best_test_accuracy"] for run_index in common_run_indices])

    if len(algorithms) < 2 or len(common_run_indices) < 2:
        return {
            "metric": "test_accuracy",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "n_runs": len(common_run_indices),
            "n_algorithms": len(algorithms),
        }

    matrix = np.asarray(aligned, dtype=float).T
    ranks = np.zeros_like(matrix)
    for row_index, row in enumerate(matrix):
        order = np.argsort(row)
        sorted_values = row[order]
        local_ranks = np.empty(len(row), dtype=float)
        start = 0
        while start < len(row):
            end = start + 1
            while end < len(row) and sorted_values[end] == sorted_values[start]:
                end += 1
            rank_value = (start + end + 1) / 2.0
            local_ranks[order[start:end]] = rank_value
            start = end
        ranks[row_index] = local_ranks

    n = matrix.shape[0]
    k = matrix.shape[1]
    rank_sums = ranks.sum(axis=0)
    statistic = (12.0 / (n * k * (k + 1.0))) * np.sum(rank_sums**2) - 3.0 * n * (k + 1.0)
    if statistic < 0:
        statistic = 0.0
    p_value = _chi_square_sf(statistic, k - 1)
    return {
        "metric": "test_accuracy",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < ALPHA),
        "n_runs": int(n),
        "n_algorithms": int(k),
    }


def _rank_abs(values):
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end + 1) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _wilcoxon_signed_rank(x, y):
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0, 1.0, False, 0

    abs_diff = np.abs(diff)
    ranks = _rank_abs(abs_diff)
    w_plus = float(np.sum(ranks[diff > 0]))
    w_minus = float(np.sum(ranks[diff < 0]))
    statistic = min(w_plus, w_minus)

    n = len(diff)
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    _, counts = np.unique(abs_diff, return_counts=True)
    tie_correction = np.sum(counts * (counts**2 - 1)) / 48.0
    variance -= tie_correction
    if variance <= 0:
        p_value = 1.0
    else:
        z = (w_plus - mean - 0.5 * np.sign(w_plus - mean)) / math.sqrt(variance)
        p_value = 2.0 * (1.0 - NormalDist().cdf(abs(z)))
    return float(statistic), float(max(min(p_value, 1.0), 0.0)), bool(p_value < ALPHA), n


def _wilcoxon_pairwise(runs_by_algorithm):
    pair_rows = []
    for i, first in enumerate(ALGORITHMS):
        for second in ALGORITHMS[i + 1 :]:
            first_runs = {run.get("run_index"): run for run in runs_by_algorithm.get(first, []) if run.get("run_index") is not None}
            second_runs = {run.get("run_index"): run for run in runs_by_algorithm.get(second, []) if run.get("run_index") is not None}
            common = sorted(set(first_runs) & set(second_runs))
            x = [first_runs[index]["best_test_accuracy"] for index in common]
            y = [second_runs[index]["best_test_accuracy"] for index in common]
            statistic, p_value, significant, n_pairs = _wilcoxon_signed_rank(x, y)
            pair_rows.append(
                {
                    "algorithm_a": first,
                    "algorithm_b": second,
                    "statistic": statistic,
                    "p_value": p_value,
                    "significant": significant,
                    "n_pairs": n_pairs,
                }
            )
    return pair_rows


def _chi_square_sf(statistic, degrees_of_freedom):
    x = max(float(statistic), 0.0)
    a = max(float(degrees_of_freedom) / 2.0, 0.5)
    return _regularized_gamma_q(a, x / 2.0)


def _regularized_gamma_q(a, x):
    if x < 0 or a <= 0:
        return 1.0
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _regularized_gamma_p(a, x)
    return _gamma_continued_fraction_q(a, x)


def _regularized_gamma_p(a, x):
    eps = 1e-12
    gln = math.lgamma(a)
    if x <= 0:
        return 0.0
    ap = a
    summation = 1.0 / a
    delta = summation
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        summation += delta
        if abs(delta) < abs(summation) * eps:
            break
    return summation * math.exp(-x + a * math.log(x) - gln)


def _gamma_continued_fraction_q(a, x):
    eps = 1e-12
    fpmin = 1e-300
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / max(b, fpmin)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def _write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_plot(path, figure):
    ensure_dir(os.path.dirname(path))
    figure.tight_layout()
    abs_path = os.path.abspath(path)
    # Save to an in-memory buffer then write to disk to avoid Windows open/save issues
    try:
        from io import BytesIO

        buf = BytesIO()
        figure.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        with open(abs_path, "wb") as fh:
            fh.write(buf.getvalue())
    finally:
        buf.close()
    plt.close(figure)


def _plot_convergence(runs_by_algorithm, final_report_dir, save_individual_runs=True):
    mean_series = {}
    per_algorithm_all = {}
    for algorithm, runs in runs_by_algorithm.items():
        histories = [run.get("convergence_history", []) for run in runs if run.get("convergence_history")]
        if not histories:
            continue
        max_len = max(len(history) for history in histories)
        padded = [history + [history[-1]] * (max_len - len(history)) for history in histories]
        array = np.asarray(padded, dtype=float)
        mean_series[algorithm] = array.mean(axis=0)
        per_algorithm_all[algorithm] = array

    if not mean_series:
        return

    # Use shared color map
    color_map = COLOR_MAP

    # Mean convergence plot: use algorithm colors
    figure, axis = plt.subplots(figsize=(10, 6))
    for algorithm, series in mean_series.items():
        color = color_map.get(algorithm, None)
        axis.plot(range(1, len(series) + 1), series, label=f"Mean {algorithm}", linewidth=2.5, color=color)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Validation Accuracy")
    axis.set_title("Mean Convergence Comparison")
    axis.grid(True, alpha=0.3)
    axis.legend()
    _write_plot(os.path.join(final_report_dir, "mean_convergence_comparison.png"), figure)

    # Convergence across runs: per-run curves use the same algorithm color but lighter (more transparent)
    figure, axis = plt.subplots(figsize=(10, 6))
    for algorithm, array in per_algorithm_all.items():
        color = color_map.get(algorithm, None)
        for run_index, series in enumerate(array, start=1):
            axis.plot(range(1, len(series) + 1), series, color=color, alpha=0.40)  # burayı değiştir
        axis.plot(range(1, len(mean_series[algorithm]) + 1), mean_series[algorithm], linewidth=3.5, label=algorithm, color=color)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Validation Accuracy")
    axis.set_title("Convergence Comparison Across Runs")
    axis.grid(True, alpha=0.3)
    axis.legend()
    _write_plot(os.path.join(final_report_dir, "convergence_comparison.png"), figure)

    if save_individual_runs:
        for algorithm, array in per_algorithm_all.items():
            figure, axis = plt.subplots(figsize=(10, 6))
            cmap = plt.get_cmap("tab10")
            for run_index, series in enumerate(array, start=1):
                # Use a distinct color per run from a qualitative colormap (not the shared COLOR_MAP)
                color = cmap((run_index - 1) % cmap.N)
                axis.plot(range(1, len(series) + 1), series, alpha=0.75, label=f"Run {run_index}", color=color)
            axis.set_xlabel("Iteration")
            axis.set_ylabel("Validation Accuracy")
            axis.set_title(f"{algorithm} Convergence Across Runs")
            axis.grid(True, alpha=0.3)
            axis.legend()
            _write_plot(os.path.join(final_report_dir, f"{algorithm}_convergence_runs.png"), figure)


def _plot_diversity(runs_by_algorithm, final_report_dir):
    figure, axis = plt.subplots(figsize=(10, 6))
    diversity_rows = []
    for algorithm, runs in runs_by_algorithm.items():
        histories = []
        for run in runs:
            series = [summary.get("population_diversity", 0.0) for summary in run.get("iteration_summaries", [])]
            if series:
                histories.append(series)
                for iteration, value in enumerate(series):
                    diversity_rows.append({"algorithm": algorithm, "run": run.get("run_name"), "iteration": iteration, "diversity": value})
        if not histories:
            continue
        max_len = max(len(history) for history in histories)
        padded = [history + [history[-1]] * (max_len - len(history)) for history in histories]
        array = np.asarray(padded, dtype=float)
        mean_series = array.mean(axis=0)
        color = COLOR_MAP.get(algorithm, None)
        axis.plot(range(1, len(mean_series) + 1), mean_series, linewidth=2.5, label=algorithm, color=color)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Population Diversity")
    axis.set_title("Diversity Comparison")
    axis.grid(True, alpha=0.3)
    axis.legend()
    _write_plot(os.path.join(final_report_dir, "diversity_comparison.png"), figure)
    return diversity_rows


def _plot_boxplots(all_rows, final_report_dir):
    metrics = [
        ("best_test_accuracy", "accuracy_boxplot.png", "Test Accuracy"),
        ("f1", "f1_boxplot.png", "F1 Score"),
        ("runtime_seconds", "runtime_boxplot.png", "Runtime (s)"),
    ]
    for key, filename, ylabel in metrics:
        values_by_algorithm = defaultdict(list)
        for row in all_rows:
            values_by_algorithm[row["algorithm"]].append(float(row[key]))
        if not values_by_algorithm:
            continue
        figure, axis = plt.subplots(figsize=(10, 6))
        labels = list(values_by_algorithm.keys())
        data = [values_by_algorithm[label] for label in labels]
        axis.boxplot(data, labels=labels, showmeans=True)
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} Stability")
        axis.grid(True, axis="y", alpha=0.3)
        _write_plot(os.path.join(final_report_dir, filename), figure)


def _hyperparameter_fields(all_rows):
    fields = set()
    for row in all_rows:
        for key in row.keys():
            if key.startswith("hp_"):
                fields.add(key)
    return sorted(fields)


def _summarize_hyperparameters(all_rows):
    summary_rows = []
    fields = _hyperparameter_fields(all_rows)
    for field in fields:
        values = [row[field] for row in all_rows if row.get(field) is not None]
        numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
        if not numeric_values:
            continue
        summary_rows.append(
            {
                "hyperparameter": field.removeprefix("hp_"),
                "mean": float(np.mean(numeric_values)),
                "std": float(np.std(numeric_values, ddof=0)),
                "min": float(np.min(numeric_values)),
                "max": float(np.max(numeric_values)),
            }
        )
    return summary_rows


def _plot_hyperparameter_distributions(all_rows, final_report_dir):
    analysis_dir = os.path.join(final_report_dir, "hyperparameter_analysis")
    distribution_dir = os.path.join(analysis_dir, "hyperparameter_distributions")
    ensure_dir(distribution_dir)
    fields = _hyperparameter_fields(all_rows)
    for field in fields:
        values = [row[field] for row in all_rows if row.get(field) is not None]
        numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
        if not numeric_values:
            continue
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.hist(numeric_values, bins=min(10, max(3, len(set(numeric_values)))), color="#2a9d8f", edgecolor="black", alpha=0.85)
        axis.set_title(f"{field.removeprefix('hp_')} Distribution")
        axis.set_xlabel(field.removeprefix("hp_"))
        axis.set_ylabel("Frequency")
        axis.grid(True, axis="y", alpha=0.25)
        _write_plot(os.path.join(distribution_dir, f"{field.removeprefix('hp_')}_histogram.png"), figure)

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.boxplot(numeric_values, vert=True, showmeans=True)
        axis.set_title(f"{field.removeprefix('hp_')} Boxplot")
        axis.set_ylabel(field.removeprefix("hp_"))
        axis.grid(True, axis="y", alpha=0.25)
        _write_plot(os.path.join(distribution_dir, f"{field.removeprefix('hp_')}_boxplot.png"), figure)


def _population_statistics(runs_by_algorithm):
    rows = []
    for algorithm, runs in runs_by_algorithm.items():
        for run in runs:
            for summary in run.get("iteration_summaries", []):
                rows.append(
                    {
                        "algorithm": algorithm,
                        "run": run.get("run_name"),
                        "seed": run.get("run_seed"),
                        "iteration": summary.get("iteration"),
                        "mean_population_fitness": summary.get("average_fitness", 0.0),
                        "std_population_fitness": summary.get("std_population_fitness", 0.0),
                        "best_fitness": summary.get("best_fitness", 0.0),
                        "worst_fitness": summary.get("worst_fitness", 0.0),
                        "population_diversity": summary.get("population_diversity", 0.0),
                        "exploration_ratio": summary.get("exploration_ratio", 0.0),
                        "exploitation_ratio": summary.get("exploitation_ratio", 0.0),
                        "unique_solution_count": summary.get("unique_solution_count", 0),
                        "repeat_rate": summary.get("repeat_rate", 0.0),
                        "iteration_time": summary.get("iteration_time", 0.0),
                    }
                )
    return rows


def _best_hyperparameter_tables(all_rows):
    per_algorithm = []
    if not all_rows:
        return [], []
    by_algorithm = defaultdict(list)
    for row in all_rows:
        by_algorithm[row["algorithm"]].append(row)

    for algorithm, rows in by_algorithm.items():
        best_row = max(rows, key=lambda item: item.get("best_test_accuracy", 0.0))
        per_algorithm.append(best_row)
    return all_rows, per_algorithm


def _project_rows(rows, fieldnames):
    projected = []
    for row in rows:
        projected.append({field: row.get(field) for field in fieldnames})
    return projected


def _ranking_rows(algorithm_summary):
    ranked = sorted(algorithm_summary.items(), key=lambda item: item[1]["mean_test_accuracy"], reverse=True)
    rows = []
    for index, (algorithm, metrics) in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": index,
                "algorithm": algorithm,
                "mean_test_accuracy": metrics["mean_test_accuracy"],
                "std_test_accuracy": metrics["std_test_accuracy"],
                "mean_f1": metrics["mean_f1"],
                "std_f1": metrics["std_f1"],
                "mean_runtime": metrics["mean_runtime"],
                "std_runtime": metrics["std_runtime"],
            }
        )
    return rows


def build_final_report(results_dir, dataset, config=None):
    runs_by_algorithm = _collect_runs(results_dir, dataset)
    all_rows = _flat_best_rows(runs_by_algorithm)
    if not all_rows:
        raise FileNotFoundError(f"No run summaries were found in {os.path.join(results_dir, dataset.upper())}")

    algorithm_summary = _aggregate_algorithm_metrics(all_rows)
    population_rows = _population_statistics(runs_by_algorithm)
    diversity_rows = [{"algorithm": row["algorithm"], "run": row["run"], "iteration": row["iteration"], "diversity": row["population_diversity"]} for row in population_rows]
    hyperparameter_summary_rows = _summarize_hyperparameters(all_rows)
    best_all_rows, best_per_algorithm_rows = _best_hyperparameter_tables(all_rows)
    ranking_rows = _ranking_rows(algorithm_summary)

    final_report_dir = os.path.join(results_dir, dataset.upper(), "final_report")
    ensure_dir(final_report_dir)

    all_run_fieldnames = [
        "algorithm",
        "run",
        "seed",
        "best_validation_accuracy",
        "best_test_accuracy",
        "precision",
        "recall",
        "f1",
        "runtime_seconds",
        "total_search_time_seconds",
        "final_training_time_seconds",
        "fitness_evaluations",
        "average_fitness_evaluation_duration",
    ]
    best_hp_fields = [field for field in sorted(best_all_rows[0].keys()) if field.startswith("hp_")]
    best_algorithm_fields = [field for field in sorted(best_per_algorithm_rows[0].keys()) if field.startswith("hp_")]

    _write_csv(
        os.path.join(final_report_dir, "best_hyperparameters_all_runs.csv"),
        _project_rows(best_all_rows, all_run_fieldnames + best_hp_fields),
        all_run_fieldnames + best_hp_fields,
    )
    _write_csv(
        os.path.join(final_report_dir, "best_hyperparameters_per_algorithm.csv"),
        _project_rows(best_per_algorithm_rows, ["algorithm", "run", "seed", "best_validation_accuracy", "best_test_accuracy", "precision", "recall", "f1", "runtime_seconds"] + best_algorithm_fields),
        ["algorithm", "run", "seed", "best_validation_accuracy", "best_test_accuracy", "precision", "recall", "f1", "runtime_seconds"] + best_algorithm_fields,
    )
    _write_csv(
        os.path.join(final_report_dir, "population_statistics.csv"),
        population_rows,
        [
            "algorithm",
            "run",
            "seed",
            "iteration",
            "mean_population_fitness",
            "std_population_fitness",
            "best_fitness",
            "worst_fitness",
            "population_diversity",
            "exploration_ratio",
            "exploitation_ratio",
            "unique_solution_count",
            "repeat_rate",
            "iteration_time",
        ],
    )
    _write_csv(os.path.join(final_report_dir, "diversity_comparison.csv"), diversity_rows, ["algorithm", "run", "iteration", "diversity"])
    _write_csv(
        os.path.join(final_report_dir, "hyperparameter_summary.csv"),
        hyperparameter_summary_rows,
        ["hyperparameter", "mean", "std", "min", "max"],
    )
    _write_csv(
        os.path.join(final_report_dir, "ranking_table.csv"),
        ranking_rows,
        ["rank", "algorithm", "mean_test_accuracy", "std_test_accuracy", "mean_f1", "std_f1", "mean_runtime", "std_runtime"],
    )

    friedman_results = _friedman_test(runs_by_algorithm)
    write_json(os.path.join(final_report_dir, "friedman_results.json"), friedman_results)
    wilcoxon_rows = _wilcoxon_pairwise(runs_by_algorithm)
    _write_csv(os.path.join(final_report_dir, "wilcoxon_results.csv"), wilcoxon_rows, ["algorithm_a", "algorithm_b", "statistic", "p_value", "significant", "n_pairs"])

    overall_summary = {
        "dataset": dataset,
        "config": config,
        "algorithm_summary": algorithm_summary,
        "friedman_results": friedman_results,
        "wilcoxon_pairs": wilcoxon_rows,
        "total_runs": len(all_rows),
        "algorithms": list(runs_by_algorithm.keys()),
        "final_report_dir": final_report_dir,
    }
    write_json(os.path.join(final_report_dir, "overall_summary.json"), overall_summary)

    _plot_convergence(runs_by_algorithm, final_report_dir)
    _plot_diversity(runs_by_algorithm, final_report_dir)
    _plot_boxplots(all_rows, final_report_dir)
    _plot_hyperparameter_distributions(all_rows, final_report_dir)

    return {
        "final_report_dir": final_report_dir,
        "overall_summary_path": os.path.join(final_report_dir, "overall_summary.json"),
        "friedman_results_path": os.path.join(final_report_dir, "friedman_results.json"),
        "wilcoxon_results_path": os.path.join(final_report_dir, "wilcoxon_results.csv"),
    }


if __name__ == "__main__":
    # Make running the file in editors (Run File) produce the final report.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    dataset = "cifar10"
    print("[research_analysis] Running as script")
    print("[research_analysis] Python:", sys.executable)
    print("[research_analysis] CWD:", os.getcwd())
    print("[research_analysis] results_dir:", results_dir)
    print("[research_analysis] dataset:", dataset)
    # Only regenerate convergence/diversity plots when running the file directly
    runs = _collect_runs(results_dir, dataset)
    final_report_dir = os.path.join(results_dir, dataset.upper(), "final_report")
    ensure_dir(final_report_dir)
    _plot_convergence(runs, final_report_dir, save_individual_runs=False)
    _plot_diversity(runs, final_report_dir)
    print(f"[research_analysis] Updated plots in {final_report_dir}")

import argparse
import copy
import csv
import os
import random
import time
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, random_split

from config import config_to_dict, load_config
from data_loader import build_data_bundle, build_data_loader
from metrics import build_confusion_matrix, evaluate_model, train_one_epoch
from model import HybridCNN
from optimizers import GreyWolfOptimizer, ParticleSwarmOptimizer, RaoOptimizer, WhaleOptimizationOptimizer
from plotting import plot_confusion_matrix, plot_curves, plot_global, plot_locals
from research_analysis import build_final_report
from utils import ensure_dir, make_torch_generator, read_json, set_global_seed, setup_logging, write_json

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

OPTIMIZER_REGISTRY = {
    "gwo": GreyWolfOptimizer,
    "pso": ParticleSwarmOptimizer,
    "woa": WhaleOptimizationOptimizer,
    "rao": RaoOptimizer,
}

INDIVIDUAL_LABELS = {
    "gwo": "Wolf",
    "pso": "Particle",
    "woa": "Whale",
    "rao": "Agent",
}


def build_run_title_suffix(run_number, run_seed):
    return f" - Run {run_number} | Seed {run_seed}"


def get_individual_label(algorithm_name):
    return INDIVIDUAL_LABELS.get(algorithm_name.lower(), "Individual")


def build_model(model_config, in_channels, img_size, num_classes):
    return HybridCNN(
        in_channels=in_channels,
        img_size=img_size,
        shared_conv_kernel_size=model_config["shared_conv_kernel_size"],
        base_filters=model_config["base_filters"],
        dilation=model_config["dilation"],
        final_neurons=model_config["final_neurons"],
        dropout=model_config["dropout"],
        se_ratio=model_config["se_ratio"],
        num_classes=num_classes,
    )


def map_position_to_config(position, config):
    search_space = config.search_space
    kernel_sizes = search_space["shared_conv_kernel_sizes"]
    base_filters = search_space["base_filters"]
    dilations = search_space["dilations"]
    final_neurons = search_space["final_neurons"]
    batch_sizes = search_space["batch_sizes"]
    se_ratios = search_space["se_ratios"]

    kernel_idx = max(0, min(int(round(position[0])), len(kernel_sizes) - 1))
    base_idx = max(0, min(int(round(position[1])), len(base_filters) - 1))
    dilation_idx = max(0, min(int(round(position[2])), len(dilations) - 1))
    final_idx = max(0, min(int(round(position[3])), len(final_neurons) - 1))
    dropout = float(position[4])
    learning_rate_log = float(position[5])
    batch_idx = max(0, min(int(round(position[6])), len(batch_sizes) - 1))
    se_idx = max(0, min(int(round(position[7])), len(se_ratios) - 1))

    learning_rate_min = search_space["learning_rate_min"]
    learning_rate_max = search_space["learning_rate_max"]
    learning_rate_log_min = float(np.log10(learning_rate_min))
    learning_rate_log_max = float(np.log10(learning_rate_max))
    learning_rate_log = max(learning_rate_log_min, min(learning_rate_log, learning_rate_log_max))
    learning_rate = float(10**learning_rate_log)

    return {
        "shared_conv_kernel_size": kernel_sizes[kernel_idx],
        "base_filters": base_filters[base_idx],
        "dilation": dilations[dilation_idx],
        "final_neurons": final_neurons[final_idx],
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_sizes[batch_idx],
        "se_ratio": se_ratios[se_idx],
    }


def build_bounds(config):
    search_space = config.search_space
    return [
        (0, len(search_space["shared_conv_kernel_sizes"]) - 1),
        (0, len(search_space["base_filters"]) - 1),
        (0, len(search_space["dilations"]) - 1),
        (0, len(search_space["final_neurons"]) - 1),
        (search_space["dropout_min"], search_space["dropout_max"]),
        (float(np.log10(search_space["learning_rate_min"])), float(np.log10(search_space["learning_rate_max"]))),
        (0, len(search_space["batch_sizes"]) - 1),
        (0, len(search_space["se_ratios"]) - 1),
    ]


def position_signature(position, config):
    model_config = map_position_to_config(position, config)
    return (
        model_config["shared_conv_kernel_size"],
        model_config["base_filters"],
        model_config["dilation"],
        model_config["final_neurons"],
        round(model_config["dropout"], 4),
        round(model_config["learning_rate"], 6),
        model_config["batch_size"],
        model_config["se_ratio"],
    )


def create_optimizer(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    learning_rate,
    epochs,
    patience,
    logger,
    lr_reduce_factor=0.5,
    lr_reduce_patience=4,
    checkpoint_path=None,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_reduce_factor,
        patience=lr_reduce_patience,
    )

    history = {"train_accuracy": [], "val_accuracy": [], "train_loss": [], "val_loss": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = -1.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_metrics = evaluate_model(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        logger.info(
            "epoch=%s train_loss=%.6f train_acc=%.6f val_loss=%.6f val_acc=%.6f",
            epoch,
            train_loss,
            train_accuracy,
            val_metrics["loss"],
            val_metrics["accuracy"],
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path is not None:
                ensure_dir(os.path.dirname(checkpoint_path))
                torch.save(best_state, checkpoint_path)
        else:
            patience_counter += 1

        scheduler.step(val_metrics["accuracy"])

        if patience is not None and patience_counter >= patience:
            logger.info("early_stopping triggered at epoch=%s", epoch)
            break

    train_time = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    return model, history, best_val_accuracy, best_epoch, train_time


def build_candidate_loaders(bundle, batch_size, seed, num_workers):
    train_loader = build_data_loader(bundle.train_dataset, batch_size, seed, num_workers, shuffle=True)
    val_loader = build_data_loader(bundle.val_dataset, batch_size, seed + 1, num_workers, shuffle=False)
    return train_loader, val_loader


def build_final_training_bundle(bundle, batch_size, seed, num_workers):
    combined_dataset = ConcatDataset([bundle.train_dataset, bundle.val_dataset])
    monitor_size = max(1, int(len(combined_dataset) * 0.1))
    train_size = len(combined_dataset) - monitor_size
    split_generator = make_torch_generator(seed + 101)
    final_train_subset, monitor_subset = random_split(combined_dataset, [train_size, monitor_size], generator=split_generator)

    loader_generator = make_torch_generator(seed + 102)

    def seed_worker(worker_id):
        worker_seed = seed + 102 + worker_id
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    final_train_loader = DataLoader(
        final_train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
        worker_init_fn=seed_worker,
    )
    monitor_loader = DataLoader(
        monitor_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return final_train_loader, monitor_loader


def summarize_confusion_matrix(confusion_matrix):
    matrix = np.asarray(confusion_matrix, dtype=float)
    true_positive = np.diag(matrix)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive

    precision_per_class = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall_per_class = true_positive / np.maximum(true_positive + false_negative, 1.0)
    f1_per_class = 2 * precision_per_class * recall_per_class / np.maximum(precision_per_class + recall_per_class, 1e-12)

    return {
        "precision": float(np.mean(precision_per_class)),
        "recall": float(np.mean(recall_per_class)),
        "f1": float(np.mean(f1_per_class)),
    }


def write_search_history(csv_path, iteration_summaries):
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, mode="w", newline="", encoding="utf-8") as file_handle:
        fieldnames = [
            "iteration",
            "unique_solution_count",
            "repeat_rate",
            "average_fitness",
            "best_fitness",
            "worst_fitness",
            "population_diversity",
            "exploration_ratio",
            "exploitation_ratio",
            "iteration_time",
        ]
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in iteration_summaries:
            writer.writerow({key: summary.get(key) for key in fieldnames})


def build_search_result(bundle, config, device, logger, run_dir, algorithm_name):
    search_history_path = os.path.join(run_dir, "search_history.csv")
    if os.path.exists(search_history_path):
        os.remove(search_history_path)

    bounds = build_bounds(config)
    fitness_cache = {}
    cache_hits = 0
    cache_misses = 0
    saved_estimated_time = 0.0
    search_start = time.perf_counter()

    def fitness(position, iteration=None, wolf_id=None):
        nonlocal cache_hits, cache_misses, saved_estimated_time
        model_config = map_position_to_config(position, config)
        solution_key = tuple(model_config.items())

        if solution_key in fitness_cache:
            cached_entry = fitness_cache[solution_key]
            val_accuracy = cached_entry["fitness"]
            cache_hit = True
            evaluation_time = 0.0
            cache_hits += 1
            saved_estimated_time += cached_entry["evaluation_time"]
        else:
            eval_seed = config.random_seed + (iteration or 0) * 100 + (wolf_id or 0)
            train_loader, val_loader = build_candidate_loaders(bundle, model_config["batch_size"], eval_seed, config.num_workers)
            model = build_model(model_config, bundle.in_channels, bundle.img_size, bundle.num_classes).to(device)
            eval_start = time.perf_counter()
            _, _, val_accuracy, _, _ = train_model(
                model,
                train_loader,
                val_loader,
                device,
                learning_rate=model_config["learning_rate"],
                epochs=config.search_epochs,
                patience=config.patience,
                lr_reduce_factor=config.lr_reduce_factor,
                lr_reduce_patience=config.lr_reduce_patience,
                logger=logger,
            )
            evaluation_time = time.perf_counter() - eval_start
            fitness_cache[solution_key] = {"fitness": val_accuracy, "evaluation_time": evaluation_time}
            cache_misses += 1
            cache_hit = False

        logger.info(
            "algorithm=%s iteration=%s agent_id=%s val_acc=%.6f cache_hit=%s config=%s",
            algorithm_name,
            iteration,
            wolf_id,
            val_accuracy,
            cache_hit,
            model_config,
        )
        return val_accuracy

    optimizer_cls = OPTIMIZER_REGISTRY[algorithm_name]
    optimizer = optimizer_cls(
        fitness,
        bounds,
        population=config.population_size,
        iterations=config.iteration_count,
        solution_signature_func=lambda position: position_signature(position, config),
        random_seed=config.random_seed,
    )
    result = optimizer.optimize()
    result.update(
        {
            "optimization_time": time.perf_counter() - search_start,
            "search_history_path": search_history_path,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_reuse_rate": (cache_hits / max(cache_hits + cache_misses, 1)) * 100.0,
            "unique_solutions_evaluated": len(fitness_cache),
            "real_training_count": cache_misses,
            "saved_training_count": cache_hits,
            "saved_estimated_time": saved_estimated_time,
            "fitness_cache": fitness_cache,
        }
    )
    write_search_history(search_history_path, result["iteration_summaries"])
    return result


def run_final_training(bundle, config, best_config, device, logger, run_dir):
    final_train_loader, monitor_loader = build_final_training_bundle(bundle, best_config["batch_size"], config.random_seed, config.num_workers)
    model = build_model(best_config, bundle.in_channels, bundle.img_size, bundle.num_classes).to(device)
    best_model_path = os.path.join(run_dir, "best_model.pth")
    model, history, best_monitor_accuracy, _, training_time = train_model(
        model,
        final_train_loader,
        monitor_loader,
        device,
        learning_rate=best_config["learning_rate"],
        epochs=config.final_epochs,
        patience=config.patience,
        lr_reduce_factor=config.lr_reduce_factor,
        lr_reduce_patience=config.lr_reduce_patience,
        logger=logger,
        checkpoint_path=best_model_path,
    )
    test_metrics = evaluate_model(model, bundle.test_loader, device)
    confusion_matrix = build_confusion_matrix(test_metrics["targets"], test_metrics["predictions"], bundle.num_classes)
    test_metrics.update(summarize_confusion_matrix(confusion_matrix))
    return {
        "model": model,
        "history": history,
        "monitor_accuracy": best_monitor_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_metrics": test_metrics,
        "confusion_matrix": confusion_matrix,
        "training_time": training_time,
        "best_model_path": best_model_path,
    }


def save_run_outputs(run_dir, algorithm_name, config, search_result, final_result, best_config, run_seed, run_number):
    title_suffix = build_run_title_suffix(run_number, run_seed)
    individual_label = get_individual_label(algorithm_name)
    write_json(os.path.join(run_dir, "config_used.json"), config_to_dict(config))
    plot_global(
        search_result["global_bests"],
        os.path.join(run_dir, "global_best.png"),
        title=f"{algorithm_name.upper()} Global Best Fitness{title_suffix}",
    )
    plot_locals(
        search_result["local_bests"],
        os.path.join(run_dir, "local_bests.png"),
        title=f"{algorithm_name.upper()} Local Bests{title_suffix}",
        individual_label=individual_label,
    )
    plot_curves(final_result["history"], os.path.join(run_dir, "training_curves.png"), title_suffix=title_suffix)
    plot_confusion_matrix(
        final_result["confusion_matrix"],
        CLASS_NAMES,
        os.path.join(run_dir, "confusion_matrix.png"),
        title_suffix=title_suffix,
    )

    run_summary = {
        "algorithm": algorithm_name,
        "dataset": config.dataset,
        "run_seed": run_seed,
        "best_validation_accuracy": float(search_result["best_fitness"]),
        "best_test_accuracy": float(final_result["test_accuracy"]),
        "precision": float(final_result["test_metrics"]["precision"]),
        "recall": float(final_result["test_metrics"]["recall"]),
        "f1": float(final_result["test_metrics"]["f1"]),
        "training_loss": final_result["history"]["train_loss"],
        "validation_loss": final_result["history"]["val_loss"],
        "total_search_time_seconds": float(search_result["optimization_time"]),
        "final_training_time_seconds": float(final_result["training_time"]),
        "runtime_seconds": float(search_result["optimization_time"] + final_result["training_time"]),
        "fitness_evaluations": int(search_result["evaluation_count"]),
        "average_fitness_evaluation_duration": float(search_result["optimization_time"] / max(search_result["evaluation_count"], 1)),
        "best_hyperparameters": best_config,
        "convergence_history": search_result["global_bests"],
        "mean_population_fitness_history": [summary["average_fitness"] for summary in search_result["iteration_summaries"]],
        "worst_population_fitness_history": [summary["worst_fitness"] for summary in search_result["iteration_summaries"]],
        "diversity_history": [summary["population_diversity"] for summary in search_result["iteration_summaries"]],
        "exploration_history": [summary["exploration_ratio"] for summary in search_result["iteration_summaries"]],
        "exploitation_history": [summary["exploitation_ratio"] for summary in search_result["iteration_summaries"]],
        "iteration_summaries": search_result["iteration_summaries"],
        "search_space": config.search_space,
        "config": config_to_dict(config),
        "optimizer_state": {
            "cache_hits": search_result["cache_hits"],
            "cache_misses": search_result["cache_misses"],
            "cache_reuse_rate": search_result["cache_reuse_rate"],
            "unique_solutions_evaluated": search_result["unique_solutions_evaluated"],
        },
        "package_versions": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    write_json(os.path.join(run_dir, "summary.json"), run_summary)
    return run_summary


def run_single_experiment(base_config, algorithm_name, run_index, logger):
    run_config = copy.deepcopy(base_config)

    run_number = run_index + 1
    run_dir = _run_dir_for_index(base_config.results_dir, base_config.dataset, algorithm_name, run_number)
    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(summary_path):
        logger.info("run_skip=%s algorithm=%s run_dir=%s reason=completed", run_number, algorithm_name, run_dir)
        return {
            "run_dir": run_dir,
            "search_result": None,
            "final_result": None,
            "best_config": None,
            "summary": read_json(summary_path),
            "skipped": True,
        }

    ensure_dir(run_dir)
    run_seed = base_config.random_seed + run_number - 1
    run_config.random_seed = run_seed

    set_global_seed(run_seed)

    bundle = build_data_bundle(
        dataset=run_config.dataset,
        data_dir=run_config.data_dir,
        batch_size=run_config.batch_size,
        train_size=run_config.train_size,
        val_size=run_config.val_size,
        seed=run_seed,
        num_workers=run_config.num_workers,
        subset_size=run_config.subset_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("run_start=%s algorithm=%s seed=%s run_dir=%s", run_index + 1, algorithm_name, run_seed, run_dir)
    search_result = build_search_result(bundle, run_config, device, logger, run_dir, algorithm_name)
    best_config = map_position_to_config(search_result["best_pos"], run_config)
    final_result = run_final_training(bundle, run_config, best_config, device, logger, run_dir)
    run_summary = save_run_outputs(run_dir, algorithm_name, run_config, search_result, final_result, best_config, run_seed, run_number)
    logger.info("run_complete=%s algorithm=%s run_dir=%s", run_index + 1, algorithm_name, run_dir)
    return {
        "run_dir": run_dir,
        "search_result": search_result,
        "final_result": final_result,
        "best_config": best_config,
        "summary": run_summary,
    }


def run_algorithm(base_config, algorithm_name, logger):
    algorithm_dir = os.path.join(base_config.results_dir, base_config.dataset.upper(), algorithm_name.upper())
    algorithm_runs = []
    for run_index in range(base_config.runs):
        algorithm_runs.append(run_single_experiment(base_config, algorithm_name, run_index, logger))

    all_run_summaries = []
    if os.path.isdir(algorithm_dir):
        for entry in sorted(os.listdir(algorithm_dir)):
            if not entry.startswith("run_"):
                continue
            summary_path = os.path.join(algorithm_dir, entry, "summary.json")
            if os.path.exists(summary_path):
                all_run_summaries.append(read_json(summary_path))
    summary = {
        "algorithm": algorithm_name,
        "dataset": base_config.dataset,
        "runs": all_run_summaries,
        "config": config_to_dict(base_config),
    }
    summary["aggregate"] = {
        "runs": len(summary["runs"]),
        "mean_test_accuracy": float(np.mean([item["best_test_accuracy"] for item in summary["runs"]])) if summary["runs"] else 0.0,
        "std_test_accuracy": float(np.std([item["best_test_accuracy"] for item in summary["runs"]], ddof=0)) if summary["runs"] else 0.0,
        "mean_validation_accuracy": float(np.mean([item["best_validation_accuracy"] for item in summary["runs"]])) if summary["runs"] else 0.0,
        "std_validation_accuracy": float(np.std([item["best_validation_accuracy"] for item in summary["runs"]], ddof=0)) if summary["runs"] else 0.0,
        "mean_runtime_seconds": float(np.mean([item["runtime_seconds"] for item in summary["runs"]])) if summary["runs"] else 0.0,
        "std_runtime_seconds": float(np.std([item["runtime_seconds"] for item in summary["runs"]], ddof=0)) if summary["runs"] else 0.0,
        "mean_f1": float(np.mean([item["f1"] for item in summary["runs"]])) if summary["runs"] else 0.0,
        "std_f1": float(np.std([item["f1"] for item in summary["runs"]], ddof=0)) if summary["runs"] else 0.0,
    }
    write_json(os.path.join(algorithm_dir, "summary.json"), summary)
    return summary


def _allocate_run_dir(results_dir, dataset, algorithm_name):
    algorithm_dir = os.path.join(results_dir, dataset.upper(), algorithm_name.upper())
    ensure_dir(algorithm_dir)
    existing_indices = []
    for entry in os.listdir(algorithm_dir):
        match = re.fullmatch(r"run_(\d+)", entry)
        if match:
            existing_indices.append(int(match.group(1)))
    next_index = max(existing_indices, default=0) + 1
    return os.path.join(algorithm_dir, f"run_{next_index:02d}")


def _run_dir_for_index(results_dir, dataset, algorithm_name, run_number):
    algorithm_dir = os.path.join(results_dir, dataset.upper(), algorithm_name.upper())
    return os.path.join(algorithm_dir, f"run_{run_number:02d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--iteration-count", type=int, default=None)
    parser.add_argument("--search-epochs", type=int, default=None)
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    for key, value in {
        "runs": args.runs,
        "population_size": args.population_size,
        "iteration_count": args.iteration_count,
        "search_epochs": args.search_epochs,
        "final_epochs": args.final_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "subset_size": args.subset_size,
        "random_seed": args.random_seed,
        "patience": args.patience,
        "dataset": args.dataset,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "num_workers": args.num_workers,
        "results_dir": args.results_dir,
        "data_dir": args.data_dir,
    }.items():
        if value is not None:
            setattr(config, key, value)

    if args.optimizer is not None:
        config.optimizers = [args.optimizer]

    ensure_dir(config.results_dir)
    logger = setup_logging(os.path.join(config.results_dir, "run.log"))
    write_json(os.path.join(config.results_dir, "config_used.json"), config_to_dict(config))
    logger.info("experiment_start config=%s", config_to_dict(config))

    start_time = time.perf_counter()
    chosen_optimizers = config.optimizers if isinstance(config.optimizers, list) else list(config.optimizers)
    for optimizer_name in chosen_optimizers:
        if optimizer_name not in OPTIMIZER_REGISTRY:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        run_algorithm(config, optimizer_name, logger)

    total_time = time.perf_counter() - start_time
    comparison_dir = os.path.join(config.results_dir, config.dataset.upper(), "comparison")
    ensure_dir(comparison_dir)
    write_json(
        os.path.join(comparison_dir, "overall_summary.json"),
        {
            "dataset": config.dataset,
            "optimizers": chosen_optimizers,
            "total_experiment_time": total_time,
            "config": config_to_dict(config),
        },
    )

    final_report = build_final_report(config.results_dir, config.dataset, config_to_dict(config))
    write_json(os.path.join(comparison_dir, "final_report_pointer.json"), final_report)
    logger.info("experiment_complete total_time=%.2f", total_time)


if __name__ == "__main__":
    main()

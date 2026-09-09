import os
import csv
import time
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from utils import ensure_dir, set_global_seed, setup_logging, write_json
from data_loader import build_data_bundle
from experiment_runner import build_model, train_model, summarize_confusion_matrix
from metrics import evaluate_model, build_confusion_matrix

MODELS = [
    ("HybridCNN_GWO_Run1", "Optimized_HybridCNN", "GWO", "results/CIFAR10/GWO/run_01/summary.json"),
    ("HybridCNN_GWO_Run3", "Optimized_HybridCNN", "GWO", "results/CIFAR10/GWO/run_03/summary.json"),
    ("HybridCNN_WOA_Run3", "Optimized_HybridCNN", "WOA", "results/CIFAR10/WOA/run_03/summary.json"),
    ("efficientnet-b0", "Baseline_CNN", "Baseline", None),
    ("mobilenetv3-large", "Baseline_CNN", "Baseline", None),
    ("shufflenetv2-1.5x", "Baseline_CNN", "Baseline", None),
]

OUT_DIR = "results/stage3_validation"
CSV_OUT = os.path.join(OUT_DIR, "stage3_one_epoch_validation_results.csv")


def load_best_hyperparams(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("best_hyperparameters", None)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    millions = total / 1e6
    return total, trainable, non_trainable, millions


def run_one(model_entry, base_config):
    model_name, model_type, optimizer, summary_path = model_entry
    run_name_safe = model_name.replace("/", "_")
    run_dir = os.path.join(OUT_DIR, run_name_safe)
    ensure_dir(run_dir)
    logger = setup_logging(os.path.join(run_dir, "run.log"))

    seed = base_config.random_seed
    set_global_seed(seed)

    # Determine batch size: optimized models use their best_hyperparameters batch_size
    best_hp = load_best_hyperparams(summary_path)
    if best_hp is not None and "batch_size" in best_hp:
        batch_size = best_hp["batch_size"]
    else:
        batch_size = base_config.batch_size

    logger.info("Running %s | batch_size=%s", model_name, batch_size)

    bundle = build_data_bundle(
        dataset=base_config.dataset,
        data_dir=base_config.data_dir,
        batch_size=batch_size,
        train_size=base_config.train_size,
        val_size=base_config.val_size,
        seed=seed,
        num_workers=base_config.num_workers,
    )

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # Prepare model_config for build_model: optimized => pass hyperparams dict; baseline => pass {'model_name': name}
    if best_hp is not None:
        model_config = best_hp
    else:
        model_config = {"model_name": model_name}

    try:
        model = build_model(model_config, bundle.in_channels, bundle.img_size, bundle.num_classes)
    except Exception as e:
        logger.exception("Failed to build model %s: %s", model_name, e)
        raise

    model = model.to(device)

    total, trainable, non_trainable, millions = count_params(model)
    write_json(
        os.path.join(run_dir, "parameter_count.json"),
        {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "non_trainable_parameters": int(non_trainable),
            "parameters_millions": float(millions),
        },
    )

    # Train for 1 epoch
    start = time.perf_counter()
    model, history, best_val_acc, best_epoch, train_time = train_model(
        model,
        bundle.train_loader,
        bundle.val_loader,
        device,
        learning_rate=(best_hp.get("learning_rate") if best_hp and "learning_rate" in best_hp else base_config.learning_rate),
        epochs=1,
        patience=None,
        logger=logger,
        checkpoint_path=None,
    )
    total_time = time.perf_counter() - start

    test_metrics = evaluate_model(model, bundle.test_loader, device)
    confusion = build_confusion_matrix(test_metrics["targets"], test_metrics["predictions"], bundle.num_classes)
    summary_scores = summarize_confusion_matrix(confusion)

    result = {
        "model": model_name,
        "model_type": model_type,
        "optimizer": optimizer,
        "run": "stage3_validation",
        "parameter_count": {"total": total, "trainable": trainable, "non_trainable": non_trainable, "millions": millions},
        "history": history,
        "train_time_seconds": train_time,
        "total_time_seconds": total_time,
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "summary_scores": summary_scores,
    }

    write_json(os.path.join(run_dir, "result.json"), result)
    write_json(os.path.join(run_dir, "confusion_matrix.json"), {"matrix": confusion.tolist()})

    # Return line for CSV
    return {
        "Model": model_name,
        "Model_Type": model_type,
        "Optimizer": optimizer,
        "Run": "stage3_validation",
        "Total_Parameters": int(total),
        "Trainable_Parameters": int(trainable),
        "Parameters_Millions": f"{millions:.6f}",
        "Train_Loss": history["train_loss"][0] if history["train_loss"] else None,
        "Train_Accuracy": history["train_accuracy"][0] if history["train_accuracy"] else None,
        "Val_Loss": history["val_loss"][0] if history["val_loss"] else None,
        "Val_Accuracy": history["val_accuracy"][0] if history["val_accuracy"] else None,
        "Test_Loss": test_metrics["loss"],
        "Test_Accuracy": test_metrics["accuracy"],
        "Precision": float(summary_scores.get("precision", 0.0)),
        "Recall": float(summary_scores.get("recall", 0.0)),
        "Macro_F1": float(summary_scores.get("f1", 0.0)),
        "Training_Time": float(train_time),
    }


def main():
    ensure_dir(OUT_DIR)
    config = load_config(None)

    rows = []
    for m in MODELS:
        try:
            row = run_one(m, config)
        except Exception as e:
            print(f"Error running model {m[0]}: {e}")
            raise
        rows.append(row)

    # write CSV
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Model",
            "Model_Type",
            "Optimizer",
            "Run",
            "Total_Parameters",
            "Trainable_Parameters",
            "Parameters_Millions",
            "Train_Loss",
            "Train_Accuracy",
            "Val_Loss",
            "Val_Accuracy",
            "Test_Loss",
            "Test_Accuracy",
            "Precision",
            "Recall",
            "Macro_F1",
            "Training_Time",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Stage3 one-epoch validation complete. CSV:", CSV_OUT)


if __name__ == "__main__":
    main()

import os
import sys
import time
import json
import csv
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from utils import ensure_dir, set_global_seed, write_json, read_json
from data_loader import build_data_bundle
from model_factory import create_model
from experiment_runner import build_model
from metrics import evaluate_model, build_confusion_matrix

# Configuration for final experiments
RESULTS_ROOT = "results/final_experiments"
DATASETS_ROOT = r"C:\Users\emirh\Desktop\Projects\datasets"

# Dataset paths (will validate existence)
DATASETS = {
    "CIFAR10": None,  # use internal loader
    "ISIC2019": os.path.join(DATASETS_ROOT, "input_sk"),
    "BrainTumor": os.path.join(DATASETS_ROOT, "BT"),
}

MODELS = [
    "HybridCNN_GWO_Run1",
    "HybridCNN_GWO_Run3",
    "HybridCNN_WOA_Run3",
    "efficientnet-b0",
    "mobilenetv3-large",
    "shufflenetv2-1.5x",
]

MASTER_CSV = os.path.join(RESULTS_ROOT, "master_results.csv")
MASTER_JSON = os.path.join(RESULTS_ROOT, "master_results.json")
FINAL_SUMMARY = os.path.join(RESULTS_ROOT, "final_summary.json")

# Smoke-test flags
SMOKE_ONLY = True


def ensure_dataset_paths():
    missing = []
    for name, path in DATASETS.items():
        if path is None:
            continue
        if not os.path.exists(path):
            missing.append((name, path))
    return missing


def check_brain_tumor_structure(bt_path):
    # Expect TRAIN and TEST directories
    train = os.path.join(bt_path, "TRAIN")
    test = os.path.join(bt_path, "TEST")
    return os.path.isdir(train), os.path.isdir(test)


def infer_num_classes_from_folder(path):
    # assume structure: path/<split>/<class> or path/<class> for test dirs
    for root, dirs, files in os.walk(path):
        # look for class directories directly under root
        if dirs:
            return len(dirs)
    return None


def import_existing_cifar_hybrid_results(target_dir):
    # copy existing JSON results for the 3 hybrid runs into target_dir/CIFAR10/Hybrid...
    src_base = os.path.join("results", "CIFAR10")
    hybrids = [
        ("HybridCNN_GWO_Run1", os.path.join(src_base, "GWO", "run_01", "summary.json")),
        ("HybridCNN_GWO_Run3", os.path.join(src_base, "GWO", "run_03", "summary.json")),
        ("HybridCNN_WOA_Run3", os.path.join(src_base, "WOA", "run_03", "summary.json")),
    ]
    imported = []
    for name, src in hybrids:
        if os.path.exists(src):
            dst_dir = os.path.join(target_dir, "CIFAR10", name)
            ensure_dir(dst_dir)
            try:
                data = read_json(src)
                write_json(os.path.join(dst_dir, "result.json"), data)
                imported.append(name)
            except Exception:
                pass
    return imported


def smoke_test_dataset_and_models(config):
    summary = {"datasets": {}, "models": {}}
    set_global_seed(config.random_seed)

    # datasets
    missing = ensure_dataset_paths()
    for name, path in DATASETS.items():
        ds_info = {"path": path, "exists": True, "details": {}}
        if path is None:
            ds_info["exists"] = True
            ds_info["details"]["note"] = "CIFAR10 uses internal loader"
        else:
            if not os.path.exists(path):
                ds_info["exists"] = False
            else:
                if name == "BrainTumor":
                    train_ok, test_ok = check_brain_tumor_structure(path)
                    ds_info["details"]["train_dir_exists"] = train_ok
                    ds_info["details"]["test_dir_exists"] = test_ok
                    if train_ok:
                        ds_info["details"]["train_num_classes"] = infer_num_classes_from_folder(os.path.join(path, "TRAIN"))
                    if test_ok:
                        ds_info["details"]["test_num_classes"] = infer_num_classes_from_folder(os.path.join(path, "TEST"))
                elif name == "ISIC2019":
                    ds_info["details"]["num_classes"] = infer_num_classes_from_folder(path)
        summary["datasets"][name] = ds_info

    # models: create and dummy forward with sample input sizes per dataset
    for model_name in MODELS:
        try:
            m_cpu = create_model(model_name, num_classes=10, device="cpu")
            # dummy forward for CIFAR10 shape
            import torch
            x = torch.randn(2, 3, 32, 32)
            y = m_cpu(x)
            out_shape = list(y.shape)
            summary["models"][model_name] = {"created": True, "dummy_output_shape": out_shape}
        except Exception as e:
            summary["models"][model_name] = {"created": False, "error": str(e)}

    return summary


def write_master_files(master_rows, master_json):
    ensure_dir(RESULTS_ROOT)
    # write csv
    with open(MASTER_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset",
            "model",
            "seed",
            "total_parameters",
            "trainable_parameters",
            "best_epoch",
            "best_val_accuracy",
            "test_accuracy",
            "test_precision_macro",
            "test_recall_macro",
            "test_f1_macro",
            "validation_accuracy",
            "validation_precision_macro",
            "validation_recall_macro",
            "validation_f1_macro",
            "training_time_seconds",
            "total_wall_time_seconds",
            "best_model_path",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in master_rows:
            writer.writerow(r)

    # write json
    write_json(master_json, {"rows": master_rows})


def main():
    config = load_config(None)
    ensure_dir(RESULTS_ROOT)

    print("Running smoke tests (no heavy training)...")
    smoke = smoke_test_dataset_and_models(config)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    smoke_path = os.path.join(RESULTS_ROOT, f"smoke_{timestamp}.json")
    write_json(smoke_path, smoke)
    print("Smoke test written:", smoke_path)

    # Import existing CIFAR hybrids into master structure
    imported = import_existing_cifar_hybrid_results(RESULTS_ROOT)
    print("Imported existing CIFAR HybridCNN results:", imported)

    # Build minimal master rows for smoke summary (mark SKIPPED)
    master_rows = []
    for dataset in ["CIFAR10", "ISIC2019", "BrainTumor"]:
        for model in MODELS:
            row = {
                "dataset": dataset,
                "model": model,
                "seed": config.random_seed,
                "total_parameters": None,
                "trainable_parameters": None,
                "best_epoch": None,
                "best_val_accuracy": None,
                "test_accuracy": None,
                "test_precision_macro": None,
                "test_recall_macro": None,
                "test_f1_macro": None,
                "validation_accuracy": None,
                "validation_precision_macro": None,
                "validation_recall_macro": None,
                "validation_f1_macro": None,
                "training_time_seconds": None,
                "total_wall_time_seconds": None,
                "best_model_path": None,
                "status": "SKIPPED",
            }
            # If CIFAR hybrids present and dataset CIFAR10 and model is hybrid, try to import
            if dataset == "CIFAR10" and model.startswith("HybridCNN"):
                # look for existing result.json in results/CIFAR10/*
                existing_paths = [
                    os.path.join("results", "CIFAR10", "GWO", "run_01", "summary.json"),
                    os.path.join("results", "CIFAR10", "GWO", "run_03", "summary.json"),
                    os.path.join("results", "CIFAR10", "WOA", "run_03", "summary.json"),
                ]
                for p in existing_paths:
                    if os.path.exists(p):
                        try:
                            data = read_json(p)
                            # map to basic fields
                            row["best_val_accuracy"] = data.get("best_hyperparameters", {}).get("best_val_accuracy") or data.get("best_val_accuracy")
                            row["best_epoch"] = data.get("best_epoch")
                            row["status"] = "IMPORTED"
                        except Exception:
                            pass
            master_rows.append(row)

    write_master_files(master_rows, MASTER_JSON)
    final_summary = {
        "total_experiments": len(master_rows),
        "successful_experiments": 0,
        "failed_experiments": 0,
        "skipped_experiments": len(master_rows),
        "total_runtime": 0,
        "dataset_status": {k: (v is not None) for k, v in DATASETS.items()},
        "model_status": {m: True for m in MODELS},
        "master_results_csv": MASTER_CSV,
        "master_results_json": MASTER_JSON,
    }
    write_json(FINAL_SUMMARY, final_summary)
    print("Smoke run complete. Master files:", MASTER_CSV, MASTER_JSON)


if __name__ == '__main__':
    main()

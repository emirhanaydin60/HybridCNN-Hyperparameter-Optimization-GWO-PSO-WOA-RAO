import json
import csv
import os
import math
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import HybridCNN

RUNS = [
    ("GWO", "run_01", "results/CIFAR10/GWO/run_01/summary.json"),
    ("GWO", "run_03", "results/CIFAR10/GWO/run_03/summary.json"),
    ("WOA", "run_03", "results/CIFAR10/WOA/run_03/summary.json"),
]

OUT_CSV = "results/optimized_hybridcnn_parameter_counts.csv"


def load_hyperparams(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    hp = data.get("best_hyperparameters") or data.get("best_params") or {}
    return hp


def build_model_from_hp(hp):
    # Map JSON keys to HybridCNN constructor
    params = {}
    params["shared_conv_kernel_size"] = hp.get("shared_conv_kernel_size", 3)
    params["base_filters"] = hp.get("base_filters", 32)
    params["dilation"] = hp.get("dilation", 1)
    params["final_neurons"] = hp.get("final_neurons", 256)
    params["dropout"] = hp.get("dropout", 0.25)
    params["se_ratio"] = hp.get("se_ratio", 8)
    # defaults
    params["in_channels"] = 3
    params["img_size"] = 32
    params["num_classes"] = 10

    model = HybridCNN(**params)
    return model, params


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    millions = total / 1e6
    return total, trainable, non_trainable, millions


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    rows = []

    print("==================================================")
    print("OPTIMIZED HYBRIDCNN PARAMETER ANALYSIS")
    print("==================================================")

    for optimizer, run_name, summary_path in RUNS:
        if not os.path.exists(summary_path):
            print(f"WARNING: summary file not found: {summary_path}")
            continue
        hp = load_hyperparams(summary_path)
        model, used_params = build_model_from_hp(hp)
        total, trainable, non_trainable, millions = count_parameters(model)

        print(f"\n{optimizer} {run_name}")
        print(f"Parameters: {total}")
        print(f"Trainable Parameters: {trainable} ({trainable/1e6:.4f} M)")

        rows.append(
            {
                "Model": "HybridCNN",
                "Optimizer": optimizer,
                "Run": run_name,
                "Total_Parameters": total,
                "Trainable_Parameters": trainable,
                "Parameters_Millions": f"{millions:.6f}",
            }
        )

    # summary stats
    millions_list = [float(r["Parameters_Millions"]) for r in rows]
    if millions_list:
        mn = min(millions_list)
        mx = max(millions_list)
        mean = sum(millions_list) / len(millions_list)
        std = math.sqrt(sum((x - mean) ** 2 for x in millions_list) / len(millions_list))

        print("\n--------------------------------------------------")
        print("SUMMARY")
        print("--------------------------------------------------")
        print(f"Minimum: {mn:.6f} M")
        print(f"Maximum: {mx:.6f} M")
        print(f"Mean: {mean:.6f} M")
        print(f"Std: {std:.6f} M")
        print("--------------------------------------------------")

    # write CSV
    if rows:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["Model", "Optimizer", "Run", "Total_Parameters", "Trainable_Parameters", "Parameters_Millions"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"\nSaved CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()

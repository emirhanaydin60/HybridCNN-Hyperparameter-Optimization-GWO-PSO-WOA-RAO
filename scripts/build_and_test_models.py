import os
import csv
import time
import json
import torch
import math

from model_factory import create_model


RUN_MODELS = [
    ("HybridCNN_GWO_Run1", "Optimized_HybridCNN", "GWO", "GWO Run 1"),
    ("HybridCNN_GWO_Run3", "Optimized_HybridCNN", "GWO", "GWO Run 3"),
    ("HybridCNN_WOA_Run3", "Optimized_HybridCNN", "WOA", "WOA Run 3"),
    ("efficientnet-b0", "Baseline_CNN", "Baseline", "Baseline"),
    ("mobilenetv3-large", "Baseline_CNN", "Baseline", "Baseline"),
    ("shufflenetv2-1.5x", "Baseline_CNN", "Baseline", "Baseline"),
]

OUT_BASELINE_CSV = "results/baseline_model_parameter_counts.csv"
OUT_COMPARISON_CSV = "results/model_parameter_comparison.csv"


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    millions = total / 1e6
    return total, trainable, non_trainable, millions


def dummy_forward_test(model, device="cpu", batch_size=4, img_size=32, in_channels=3, num_classes=10):
    model.eval()
    x = torch.randn(batch_size, in_channels, img_size, img_size, device=device)
    with torch.no_grad():
        out = model(x)
    return out.shape


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    comparison_rows = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name, model_type, optimizer, run in RUN_MODELS:
        print(f"Building {model_name} on {device}")
        try:
            model = create_model(model_name, num_classes=10, device=device)
        except Exception as e:
            print(f"Failed to create {model_name}: {e}")
            continue

        total, trainable, non_trainable, millions = count_params(model)
        print(f"{model_name}: {millions:.6f} M parameters")

        # dummy forward
        try:
            out_shape = dummy_forward_test(model, device=device, batch_size=2, img_size=32, in_channels=3, num_classes=10)
        except Exception as e:
            out_shape = f"forward failed: {e}"

        rows.append({
            "Model": model_name,
            "Model_Type": model_type,
            "Optimizer": optimizer,
            "Run": run,
            "Total_Parameters": total,
            "Trainable_Parameters": trainable,
            "Non_Trainable_Parameters": non_trainable,
            "Parameters_Millions": f"{millions:.6f}",
            "Dummy_Output_Shape": str(out_shape),
        })

        comparison_rows.append({
            "Model": model_name,
            "Model_Type": model_type,
            "Optimizer": optimizer,
            "Run": run,
            "Total_Parameters": total,
            "Trainable_Parameters": trainable,
            "Non_Trainable_Parameters": non_trainable,
            "Parameters_Millions": f"{millions:.6f}",
        })

    # write baseline CSV (only baseline models rows)
    with open(OUT_BASELINE_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Model", "Model_Type", "Optimizer", "Run", "Total_Parameters", "Trainable_Parameters", "Non_Trainable_Parameters", "Parameters_Millions", "Dummy_Output_Shape"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # write comparison CSV (all models)
    with open(OUT_COMPARISON_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Model", "Model_Type", "Optimizer", "Run", "Total_Parameters", "Trainable_Parameters", "Non_Trainable_Parameters", "Parameters_Millions"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in comparison_rows:
            writer.writerow(r)

    print(f"Wrote: {OUT_BASELINE_CSV}")
    print(f"Wrote: {OUT_COMPARISON_CSV}")


if __name__ == "__main__":
    main()

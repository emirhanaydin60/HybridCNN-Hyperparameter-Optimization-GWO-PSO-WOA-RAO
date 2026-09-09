"""Generate a grid of confusion matrices for selected runs.

Usage:
  python -m tools.plot_confusion_grid --out results/CIFAR10/confusion_grid.png

This script rebuilds numeric confusion matrices (by loading each run's saved model)
and uses a grid plotting helper to render a combined image.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np

# Local helpers
from finalize_results import rebuild_confusion_matrix, CIFAR10_CLASS_NAMES


def _panel_prefix(index):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if index < len(alphabet):
        return f"({alphabet[index]})"
    return f"({index + 1})"


def plot_confusion_matrix_grid(models_data, out_path, class_names=None, grid_cols=3, grid_rows=None):
    # This is a lightweight inline copy of the plotting code you provided,
    # adapted to avoid adding new package-level dependencies.
    import math
    import matplotlib.pyplot as plt

    def _resolve_grid_shape(count, grid_cols=3, grid_rows=None):
        cols = max(1, int(grid_cols) if grid_cols is not None else 1)
        rows = max(1, int(grid_rows) if grid_rows is not None else math.ceil(count / cols))
        if rows * cols < count:
            rows = math.ceil(count / cols)
        return rows, cols

    count = len(models_data)
    if count == 0:
        return

    rows, cols = _resolve_grid_shape(count, grid_cols=grid_cols, grid_rows=grid_rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.8))
    axes = np.array(axes).reshape(-1)

    matrices = [np.asarray(item.get("confusion_matrix")) for item in models_data if item.get("confusion_matrix") is not None]
    matrices = [cm for cm in matrices if cm.size > 0]
    if not matrices:
        plt.close(fig)
        return

    vmax = max(float(np.max(cm)) for cm in matrices)
    if vmax <= 0:
        vmax = 1.0

    cmap = plt.get_cmap("Blues")
    for idx, item in enumerate(models_data):
        ax = axes[idx]
        cm = item.get("confusion_matrix")
        if cm is None:
            ax.set_axis_off()
            continue

        cm = np.asarray(cm)
        if cm.size == 0:
            ax.set_axis_off()
            continue

        item_class_names = item.get("class_names") or class_names
        if not item_class_names or len(item_class_names) != cm.shape[0]:
            item_class_names = [str(i) for i in range(cm.shape[0])]

        ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax)
        for row_idx in range(cm.shape[0]):
            for col_idx in range(cm.shape[1]):
                value = cm[row_idx, col_idx]
                text_color = "white" if value > (vmax * 0.55) else "black"
                ax.text(col_idx, row_idx, f"{int(value)}", ha="center", va="center", fontsize=6.5, color=text_color)

        display_label = item.get("display_label") or item.get("label")
        ax.set_title(f"({_panel_prefix(idx)[1:-1]}) {display_label}", fontsize=10, fontweight="bold")
        ax.set_xticks(np.arange(len(item_class_names)))
        ax.set_yticks(np.arange(len(item_class_names)))
        ax.set_xticklabels(item_class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(item_class_names, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)

        caption_parts = []
        if item.get("accuracy") is not None:
            caption_parts.append(f"acc={item['accuracy']:.3f}")
        if item.get("macro_f1") is not None:
            caption_parts.append(f"mf1={item['macro_f1']:.3f}")
        if caption_parts:
            ax.text(0.02, 0.02, " | ".join(caption_parts), transform=ax.transAxes, fontsize=8, va="bottom")

    for idx in range(count, len(axes)):
        axes[idx].set_axis_off()

    fig.suptitle("Test confusion matrices by model", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_run_data(run_path: str):
    cfg_path = os.path.join(run_path, "config_used.json")
    summary_path = os.path.join(run_path, "summary.json")
    if not os.path.exists(cfg_path) or not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing run artifacts in {run_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    with open(summary_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)

    best_config = summary.get("best_hyperparameters") or summary.get("best_config") or {}
    return config, best_config, summary


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/CIFAR10/confusion_grid.png")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[
            "results/CIFAR10/GWO/run_01",
            "results/CIFAR10/WOA/run_03",
            "results/CIFAR10/GWO/run_03",
        ],
    )
    args = parser.parse_args(argv)

    models_data = []
    for run_path in args.runs:
        if not os.path.exists(run_path):
            print(f"Run path not found: {run_path}")
            continue
        try:
            config, best_config, summary = load_run_data(run_path)
            cm, cm_path = rebuild_confusion_matrix(run_path, config, best_config)
            if cm is None:
                print(f"Failed to rebuild confusion for {run_path}")
                continue

            accuracy = None
            if summary.get("best_test_accuracy") is not None:
                accuracy = float(summary["best_test_accuracy"])
            label = os.path.basename(os.path.dirname(run_path)) + "/" + os.path.basename(run_path)
            display = f"{label} → {accuracy * 100:.2f}%" if accuracy is not None else label

            models_data.append(
                {
                    "label": label,
                    "display_label": display,
                    "confusion_matrix": cm,
                    "class_names": CIFAR10_CLASS_NAMES,
                    "accuracy": accuracy,
                    "macro_f1": float(summary.get("f1")) if summary.get("f1") is not None else None,
                }
            )
        except Exception as exc:
            print(f"Error processing {run_path}: {exc}")

    if not models_data:
        print("No confusion matrices collected.")
        return 2

    plot_confusion_matrix_grid(models_data, args.out, class_names=CIFAR10_CLASS_NAMES, grid_cols=3)
    print(f"Wrote grid to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

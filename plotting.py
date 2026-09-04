import os

import matplotlib.pyplot as plt
import numpy as np

from metrics import normalize_confusion_matrix


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_global(global_bests, out_path, title="Global Best Fitness"):
    _ensure_parent_dir(out_path)
    plt.figure(figsize=(8, 4))
    plt.plot(global_bests, marker="o", label="Global Best")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_locals(local_bests, out_path, title="Local Bests", individual_label="Wolf"):
    _ensure_parent_dir(out_path)
    arr = np.array(local_bests)
    plt.figure(figsize=(10, 5))
    for i in range(arr.shape[1]):
        plt.plot(arr[:, i], label=f"{individual_label}-{i + 1}")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Personal Best Validation Accuracy")
    plt.legend(fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_curves(history, out_path, title_suffix=""):
    _ensure_parent_dir(out_path)
    epochs = range(1, len(history["train_accuracy"]) + 1)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(epochs, history["train_accuracy"], label="Training Accuracy")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title(f"Accuracy Curve{title_suffix}")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_loss"], label="Training Loss")
    axes[1].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[1].set_title(f"Loss Curve{title_suffix}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)


def plot_confusion_matrix(confusion_matrix, class_names, out_path, title_suffix=""):
    _ensure_parent_dir(out_path)
    normalized_matrix = normalize_confusion_matrix(confusion_matrix)

    plt.figure(figsize=(9, 7))
    plt.imshow(normalized_matrix, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix{title_suffix}")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    # Hücrelere değer yaz
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):

            value = confusion_matrix[i, j]
            ratio = normalized_matrix[i, j] * 100

            color = "white" if normalized_matrix[i, j] > 0.5 else "black"

            plt.text(j, i - 0.12, f"{value}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")

            plt.text(j, i + 0.18, f"{ratio:.1f}%", ha="center", va="center", color=color, fontsize=6, fontweight="bold")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

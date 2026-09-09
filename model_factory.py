import os
import json
import torch
import torch.nn as nn
from typing import Tuple

from model import HybridCNN


def _load_hyperparams_from_summary(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("best_hyperparameters", {})


def _replace_last_linear(model: nn.Module, out_features: int) -> Tuple[nn.Module, str]:
    # Find last nn.Linear module and replace it with new Linear(out_features)
    last_name = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_name = name
    if last_name is None:
        raise RuntimeError("No Linear layer found to replace")

    parts = last_name.split('.')
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)

    last_part = parts[-1]
    old_linear = None
    if last_part.isdigit():
        idx = int(last_part)
        old_linear = parent[idx]
        parent[idx] = nn.Linear(old_linear.in_features, out_features)
    else:
        old_linear = getattr(parent, last_part)
        setattr(parent, last_part, nn.Linear(old_linear.in_features, out_features))

    return model, last_name


def create_model(model_name: str, num_classes: int = 10, device: str = "cpu"):
    name = model_name.lower()
    # Optimized HybridCNN variants
    if name in ("hybridcnn_gwo_run1", "hybridcnn_gwo_run3", "hybridcnn_woa_run3"):
        # Map to summary paths
        mapping = {
            "hybridcnn_gwo_run1": "results/CIFAR10/GWO/run_01/summary.json",
            "hybridcnn_gwo_run3": "results/CIFAR10/GWO/run_03/summary.json",
            "hybridcnn_woa_run3": "results/CIFAR10/WOA/run_03/summary.json",
        }
        summary_path = mapping[name]
        hp = _load_hyperparams_from_summary(summary_path)
        params = {
            "in_channels": hp.get("in_channels", 3),
            "img_size": hp.get("img_size", 32),
            "shared_conv_kernel_size": hp.get("shared_conv_kernel_size", 3),
            "base_filters": hp.get("base_filters", 32),
            "dilation": hp.get("dilation", 1),
            "final_neurons": hp.get("final_neurons", 256),
            "dropout": hp.get("dropout", 0.25),
            "se_ratio": hp.get("se_ratio", 8),
            "num_classes": num_classes,
        }
        model = HybridCNN(**params)
        return model.to(device)

    # Baseline CNNs: EfficientNet-B0, MobileNetV3-Large, ShuffleNetV2 1.5x
    import torchvision.models as tvmodels

    if name == "efficientnet-b0" or name == "efficientnet_b0":
        # Newer torchvision uses weights=None; older use pretrained=False
        try:
            model = tvmodels.efficientnet_b0(weights=None)
            weights_source = "weights=None"
        except TypeError:
            model = tvmodels.efficientnet_b0(pretrained=False)
            weights_source = "pretrained=False"
        model, last = _replace_last_linear(model, num_classes)
        return model.to(device)

    if name in ("mobilenetv3-large", "mobilenetv3_large", "mobilenet_v3_large"):
        try:
            model = tvmodels.mobilenet_v3_large(weights=None)
            weights_source = "weights=None"
        except TypeError:
            model = tvmodels.mobilenet_v3_large(pretrained=False)
            weights_source = "pretrained=False"
        model, last = _replace_last_linear(model, num_classes)
        return model.to(device)

    if name in ("shufflenetv2-1.5x", "shufflenet_v2_x1_5", "shufflenetv2_1.5x"):
        try:
            model = tvmodels.shufflenet_v2_x1_5(weights=None)
            weights_source = "weights=None"
        except TypeError:
            model = tvmodels.shufflenet_v2_x1_5(pretrained=False)
            weights_source = "pretrained=False"
        model, last = _replace_last_linear(model, num_classes)
        return model.to(device)

    raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # quick smoke test
    m = create_model("efficientnet-b0", num_classes=10)
    print(m)

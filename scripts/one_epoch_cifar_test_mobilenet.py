import os
import time
import json
import sys
import torch

# ensure repo root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import load_config
from data_loader import build_data_bundle
from utils import setup_logging, set_global_seed, write_json, ensure_dir
from model_factory import create_model
from experiment_runner import train_model
from metrics import evaluate_model, build_confusion_matrix


def main():
    # Load base config and override for this quick test
    config = load_config(None)
    config.dataset = "cifar10"
    config.batch_size = 128
    config.random_seed = 42
    config.final_epochs = 50
    config.search_epochs = 6

    # prepare run dir
    run_dir = os.path.join("results", "CIFAR10", "ONE_EPOCH_TEST", "MobileNetV3_run01")
    ensure_dir(run_dir)

    logger = setup_logging(os.path.join(run_dir, "run.log"))
    logger.info("Starting 1-epoch CIFAR-10 test: MobileNetV3-Large")

    # Seed
    set_global_seed(config.random_seed)

    # Data
    bundle = build_data_bundle(
        dataset=config.dataset,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        train_size=config.train_size,
        val_size=config.val_size,
        seed=config.random_seed,
        num_workers=config.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Model
    model = create_model("mobilenetv3-large", num_classes=bundle.num_classes, device=device)

    # Parameter counts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    params_m = total / 1e6

    write_json(
        os.path.join(run_dir, "parameter_count.json"),
        {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "non_trainable_parameters": int(non_trainable),
            "parameters_millions": float(params_m),
        },
    )

    # Train for 3 epochs to validate pipeline
    start = time.perf_counter()
    model, history, best_val_acc, best_epoch, train_time = train_model(
        model,
        bundle.train_loader,
        bundle.val_loader,
        device,
        learning_rate=config.learning_rate,
        epochs=3,
        patience=None,
        logger=logger,
    )
    total_time = time.perf_counter() - start

    # Test evaluation
    test_metrics = evaluate_model(model, bundle.test_loader, device)
    confusion = build_confusion_matrix(test_metrics["targets"], test_metrics["predictions"], bundle.num_classes)

    result = {
        "dataset": config.dataset,
        "model": "mobilenetv3-large",
        "seed": config.random_seed,
        "device": str(device),
        "history": history,
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "train_time_seconds": train_time,
        "total_time_seconds": total_time,
        "test_metrics": test_metrics,
    }

    write_json(os.path.join(run_dir, "one_epoch_result.json"), result)
    # Save confusion matrix as JSON
    write_json(
        os.path.join(run_dir, "confusion_matrix.json"),
        {
            "matrix": confusion.tolist(),
        },
    )

    logger.info("1-epoch MobileNetV3 test complete. Results saved to %s", run_dir)


if __name__ == "__main__":
    main()

"""Orquestra treino, validação e persistência dos resultados do experimento."""

import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import Config
from src.data import get_dataloaders
from src.evaluate import confusion_matrix_from, evaluate
from src.train import resolve_device, train_model
from src.utils import (
    count_parameters,
    get_hardware_info,
    measure_inference_time,
    save_run,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "runs"


def run_experiment(
    model: nn.Module,
    config: Config,
    *,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Executa um experimento usando apenas loaders de treino e validação."""
    if (train_loader is None) != (val_loader is None):
        raise ValueError("Forneça train_loader e val_loader juntos, ou nenhum dos dois.")

    set_seed(config.seed)
    device = resolve_device(config.device)
    if train_loader is None:
        train_loader, val_loader = get_dataloaders(config)
    assert val_loader is not None

    # Passa ao loop de treino o dispositivo já resolvido sem alterar a configuração
    # solicitada pelo usuário, que é preservada no arquivo JSON.
    resolved_config = replace(config, device=device.type)
    model.to(device)
    training = train_model(model, resolved_config, train_loader, val_loader)

    checkpoint_path = Path(training["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint do melhor modelo não encontrado: {checkpoint_path}")
    best_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state)

    validation_result = evaluate(model, val_loader, device)
    matrix = confusion_matrix_from(
        validation_result["y_true"], validation_result["y_pred"]
    )
    inference_seconds_per_image = measure_inference_time(model, val_loader, device)
    hardware = get_hardware_info()
    hardware["device_type"] = device.type
    hardware["device_name"] = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or platform.machine()
    )

    result: dict[str, Any] = {
        "config": asdict(config),
        "history": training["history"],
        "best_val_acc": training["best_val_acc"],
        "best_epoch": training["best_epoch"],
        "validation": {
            "loss": validation_result["loss"],
            "accuracy": validation_result["accuracy"],
            "confusion_matrix": matrix.tolist(),
        },
        "parameters": count_parameters(model),
        "timings": {
            "train_seconds": training["train_time_seconds"],
            "inference_seconds_per_image": inference_seconds_per_image,
        },
        "hardware": hardware,
    }
    result_path = RESULTS_DIR / f"{config.experiment_name}.json"
    save_run(result, result_path)
    return result

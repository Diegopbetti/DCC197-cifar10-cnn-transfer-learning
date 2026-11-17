"""Loop configurável de treinamento e validação para modelos PyTorch."""

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.evaluate import evaluate


CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def resolve_device(device_name: str) -> torch.device:
    """Resolve o dispositivo configurado e dá erro claro se CUDA não existir."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("O dispositivo CUDA foi solicitado, mas não está disponível.")
    if device_name not in ("cpu", "cuda"):
        raise ValueError("device deve ser 'auto', 'cpu' ou 'cuda'.")
    return torch.device(device_name)


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_optimizer(config: Config, parameters: list[nn.Parameter]):
    if config.optimizer == "sgd":
        return SGD(
            parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adam":
        return Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)
    raise ValueError(f"Otimizador não suportado: {config.optimizer}.")


def _make_scheduler(config: Config, optimizer):
    if config.scheduler == "none":
        return None
    if config.scheduler == "step":
        return StepLR(
            optimizer,
            step_size=max(1, config.epochs // 3),
            gamma=0.1,
        )
    if config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    raise ValueError(f"Agendador não suportado: {config.scheduler}.")


def train_model(
    model: nn.Module,
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> dict[str, Any]:
    """Treina o modelo e salva os pesos da época com melhor acurácia de validação."""
    try:
        if len(train_loader) == 0:
            raise ValueError("Não é possível treinar com um loader de treino vazio.")
    except TypeError:
        pass

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("O modelo não possui parâmetros treináveis.")

    device = resolve_device(config.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(config, trainable_parameters)
    scheduler = _make_scheduler(config, optimizer)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epoch_time_seconds": [],
    }
    best_val_acc = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    checkpoint_path = CHECKPOINT_DIR / f"{config.experiment_name}_best.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    for epoch in range(config.epochs):
        _synchronize_if_cuda(device)
        epoch_start = time.perf_counter()
        model.train()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        progress = tqdm(
            train_loader,
            desc=f"Época {epoch + 1}/{config.epochs}",
            leave=False,
        )
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.ndim != 2 or logits.shape != (labels.shape[0], 10):
                raise ValueError("O modelo deve retornar logits com shape [batch, 10].")
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        if total_examples == 0:
            raise ValueError("Não é possível treinar com um loader de treino vazio.")

        train_loss = total_loss / total_examples
        train_accuracy = total_correct / total_examples
        validation = evaluate(model, val_loader, device)
        _synchronize_if_cuda(device)
        epoch_seconds = time.perf_counter() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(float(validation["loss"]))
        history["val_acc"].append(float(validation["accuracy"]))
        history["lr"].append(learning_rate)
        history["epoch_time_seconds"].append(epoch_seconds)

        if validation["accuracy"] > best_val_acc:
            best_val_acc = float(validation["accuracy"])
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            torch.save(best_state, checkpoint_path)

        if scheduler is not None:
            scheduler.step()

    _synchronize_if_cuda(device)
    train_seconds = time.perf_counter() - total_start
    if best_state is None:
        raise ValueError("O treinamento não produziu métricas de validação.")

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_path),
        "train_time_seconds": train_seconds,
    }

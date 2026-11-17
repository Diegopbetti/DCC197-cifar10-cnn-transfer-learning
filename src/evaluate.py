"""Métricas de validação e matriz de confusão para CIFAR-10."""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> dict[str, Any]:
    """Calcula perda média, acurácia, predições e rótulos de um loader."""
    resolved_device = torch.device(device)
    criterion = nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    predictions: list[int] = []
    labels_seen: list[int] = []

    try:
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(resolved_device)
                labels = labels.to(resolved_device)
                logits = model(images)
                if logits.ndim != 2 or logits.shape != (labels.shape[0], 10):
                    raise ValueError(
                        "O modelo deve retornar logits com shape [batch, 10]."
                    )

                loss = criterion(logits, labels)
                batch_size = labels.shape[0]
                total_loss += loss.item() * batch_size
                batch_predictions = logits.argmax(dim=1)
                total_correct += (batch_predictions == labels).sum().item()
                total_examples += batch_size
                predictions.extend(batch_predictions.cpu().tolist())
                labels_seen.extend(labels.cpu().tolist())
    finally:
        model.train(was_training)

    if total_examples == 0:
        raise ValueError("Não é possível avaliar um loader vazio.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "y_pred": predictions,
        "y_true": labels_seen,
    }


def confusion_matrix_from(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> np.ndarray:
    """Retorna matriz 10 x 10, preservando classes sem exemplos no lote."""
    return confusion_matrix(y_true, y_pred, labels=list(range(10)))

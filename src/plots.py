"""Geração de curvas de treinamento em PDF."""

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


def save_training_curves(
    history: Mapping[str, Sequence[float]],
    experiment_name: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Salva curvas de perda e acurácia de treino/validação em PDF."""
    required_keys = ("train_loss", "val_loss", "train_acc", "val_acc")
    missing = [key for key in required_keys if key not in history]
    if missing:
        raise ValueError(f"Histórico sem as métricas necessárias: {', '.join(missing)}.")

    lengths = {key: len(history[key]) for key in required_keys}
    if not lengths["train_loss"] or len(set(lengths.values())) != 1:
        raise ValueError("As métricas de treino e validação devem ter o mesmo número de épocas.")

    epochs = list(range(1, lengths["train_loss"] + 1))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    curves = {
        "loss": ("train_loss", "val_loss", "Perda"),
        "accuracy": ("train_acc", "val_acc", "Acurácia"),
    }
    saved_files: dict[str, Path] = {}

    for curve_name, (train_key, val_key, metric_label) in curves.items():
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.plot(epochs, history[train_key], label="Treino", linewidth=2)
        axis.plot(epochs, history[val_key], label="Validação", linewidth=2)
        axis.set_title(f"{metric_label} por época — {experiment_name}")
        axis.set_xlabel("Época")
        axis.set_ylabel(metric_label)
        axis.set_xticks(epochs)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend()

        try:
            file_path = output_path / f"{experiment_name}_{curve_name}.pdf"
            figure.savefig(file_path, format="pdf")
            saved_files[f"{curve_name}_pdf"] = file_path
        finally:
            plt.close(figure)

    return saved_files

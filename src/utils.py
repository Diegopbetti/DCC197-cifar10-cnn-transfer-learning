"""Utilitários comuns para experimentos reproduzíveis."""

import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Fixa as sementes de Python, NumPy e PyTorch, incluindo CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Conta parâmetros totais e aqueles que participarão do treinamento."""
    parameters = list(model.parameters())
    return {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
    }


def _synchronize_if_cuda(device: torch.device) -> None:
    """Espera a GPU terminar o trabalho pendente antes de medir o tempo."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> float:
    """Retorna segundos por imagem após aquecimento e uma passagem cronometrada."""
    resolved_device = torch.device(device)
    first_batch = next(iter(loader), None)
    if first_batch is None:
        raise ValueError("Não é possível medir inferência com um loader vazio.")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            warmup_images = first_batch[0].to(resolved_device)
            model(warmup_images)
            _synchronize_if_cuda(resolved_device)

            image_count = 0
            start_time = time.perf_counter()
            for batch in loader:
                images = batch[0].to(resolved_device)
                model(images)
                image_count += images.shape[0]
            _synchronize_if_cuda(resolved_device)
            elapsed = time.perf_counter() - start_time
    finally:
        model.train(was_training)

    if image_count == 0:
        raise ValueError("O loader não forneceu imagens para medir inferência.")
    return elapsed / image_count


def get_hardware_info() -> dict[str, Any]:
    """Retorna versões do ambiente e identifica o dispositivo disponível."""
    if torch.cuda.is_available():
        device_type = "cuda"
        device_name = torch.cuda.get_device_name(0)
    else:
        device_type = "cpu"
        device_name = platform.processor() or platform.machine()
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_type": device_type,
        "device_name": device_name,
    }


def save_run(result: dict[str, Any], path: str | Path) -> None:
    """Salva o resultado de um experimento como JSON UTF-8 formatado."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

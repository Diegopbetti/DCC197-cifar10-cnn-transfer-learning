"""Utilitários comuns para experimentos reproduzíveis."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fixa as sementes de Python, NumPy e PyTorch, incluindo CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

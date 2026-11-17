"""Configuração comum dos experimentos do trabalho."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    """Hiperparâmetros e opções usados pela infraestrutura de dados e treino."""

    experiment_name: str
    seed: int = 42
    optimizer: Literal["sgd", "adam"] = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 20
    scheduler: Literal["none", "step", "cosine"] = "none"
    augmentation: bool = True
    resize: int | None = None
    normalization: Literal["cifar10", "imagenet"] = "cifar10"
    num_workers: int = 2
    device: str = "auto"

    def __post_init__(self) -> None:
        """Falha cedo quando uma opção básica está fora do domínio esperado."""
        if not self.experiment_name.strip():
            raise ValueError("experiment_name não pode ser vazio.")
        if self.lr <= 0:
            raise ValueError("lr deve ser maior que zero.")
        if self.batch_size <= 0:
            raise ValueError("batch_size deve ser maior que zero.")
        if self.epochs <= 0:
            raise ValueError("epochs deve ser maior que zero.")
        if self.num_workers < 0:
            raise ValueError("num_workers não pode ser negativo.")
        if self.resize is not None and self.resize <= 0:
            raise ValueError("resize deve ser um inteiro positivo ou None.")
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError("optimizer deve ser 'sgd' ou 'adam'.")
        if self.scheduler not in ("none", "step", "cosine"):
            raise ValueError("scheduler deve ser 'none', 'step' ou 'cosine'.")
        if self.normalization not in ("cifar10", "imagenet"):
            raise ValueError("normalization deve ser 'cifar10' ou 'imagenet'.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device deve ser 'auto', 'cpu' ou 'cuda'.")

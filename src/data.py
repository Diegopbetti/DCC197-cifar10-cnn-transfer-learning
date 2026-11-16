"""Datasets e dataloaders do CIFAR-10 para treino e validação."""

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import Config


CLASS_NAMES = (
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_PATH = PROJECT_ROOT / "splits" / "splits.npz"

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(config: Config, training: bool) -> transforms.Compose:
    """Monta transforms; augmentation é exclusiva do conjunto de treino."""
    steps = []
    if training and config.augmentation:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
            ]
        )
    # Aumenta a imagem CIFAR-10 antes do resize opcional para que, por exemplo,
    # um modelo pré-treinado continue recebendo imagens 224 x 224.
    if config.resize is not None:
        steps.append(transforms.Resize((config.resize, config.resize)))
    steps.append(transforms.ToTensor())
    if config.normalization == "imagenet":
        mean, std = _IMAGENET_MEAN, _IMAGENET_STD
    else:
        mean, std = _CIFAR10_MEAN, _CIFAR10_STD
    steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


def _load_split() -> tuple[list[int], list[int]]:
    """Carrega o split fixo ou informa como gerá-lo uma única vez."""
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(
            f"Split não encontrado em {SPLIT_PATH}. Gere-o uma vez com "
            "'python scripts/make_split.py' antes de carregar os dados."
        )
    with np.load(SPLIT_PATH) as split:
        if "train_idx" not in split or "val_idx" not in split:
            raise ValueError(f"O arquivo {SPLIT_PATH} não contém train_idx e val_idx.")
        train_idx = split["train_idx"].astype(np.int64).tolist()
        val_idx = split["val_idx"].astype(np.int64).tolist()
    if len(train_idx) != 45_000 or len(val_idx) != 5_000:
        raise ValueError(
            "O split precisa conter 45.000 índices de treino e 5.000 de validação."
        )
    if set(train_idx).intersection(val_idx):
        raise ValueError("O arquivo de split contém índices em treino e validação.")
    if min(train_idx + val_idx) < 0 or max(train_idx + val_idx) >= 50_000:
        raise ValueError("O arquivo de split contém índices fora do CIFAR-10 de treino.")
    if len(set(train_idx).union(val_idx)) != 50_000:
        raise ValueError("Treino e validação devem cobrir os 50.000 índices originais.")
    return train_idx, val_idx


def get_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    """Retorna somente loaders de treino e validação; nunca acessa o teste."""
    train_idx, val_idx = _load_split()
    train_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=_build_transform(config, training=True),
    )
    val_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=_build_transform(config, training=False),
    )
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device in ("auto", "cuda"),
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device in ("auto", "cuda"),
    )
    return train_loader, val_loader


def get_test_loader(config: Config) -> DataLoader:
    """Retorna loader de teste separado para uso na avaliação final (Sprint 3)."""
    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=_build_transform(config, training=False),
    )
    return DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device in ("auto", "cuda"),
    )

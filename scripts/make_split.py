"""Gera uma única divisão estratificada e fixa para treino e validação."""

import argparse
import sys
from pathlib import Path
from urllib.error import URLError

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
SPLIT_PATH = PROJECT_ROOT / "splits" / "splits.npz"
SEED = 42
NUM_CLASSES = 10
VALIDATION_PER_CLASS = 500
DOWNLOAD_URLS = (
    # URL atual divulgado na página oficial do CIFAR-10, sem o redirecionamento
    # do endereço antigo usado por algumas versões do torchvision.
    "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz",
    # Espelho alternativo do mesmo arquivo; o torchvision valida o MD5 esperado.
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
)


def _download_dataset():
    """Tenta a origem oficial atual e um espelho com o mesmo checksum."""
    errors = []
    for url in DOWNLOAD_URLS:
        CIFAR10.url = url
        try:
            return CIFAR10(root=str(DATA_DIR), train=True, download=True)
        except URLError as error:
            errors.append(f"{url}: {error}")
    details = "\n".join(errors)
    raise RuntimeError(
        "Não foi possível baixar o CIFAR-10. Verifique sua conexão/DNS ou "
        "baixe o arquivo cifar-10-python.tar.gz manualmente para a pasta data/.\n"
        f"Tentativas:\n{details}"
    )


def create_split(force: bool = False) -> Path:
    """Baixa os dados de treino, cria e salva o split estratificado."""
    if SPLIT_PATH.exists() and not force:
        raise FileExistsError(
            f"{SPLIT_PATH} já existe e não será sobrescrito. "
            "Use --force somente se o grupo decidiu recriar o split."
        )

    dataset = _download_dataset()
    labels = np.asarray(dataset.targets)
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=5_000,
        stratify=labels,
        random_state=SEED,
    )

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    val_counts = np.bincount(labels[val_idx], minlength=NUM_CLASSES)
    overlap = np.intersect1d(train_idx, val_idx)

    if len(train_idx) != 45_000 or len(val_idx) != 5_000:
        raise RuntimeError("O split gerado não tem os tamanhos esperados.")
    if not np.all(val_counts == VALIDATION_PER_CLASS):
        raise RuntimeError(f"A validação não tem 500 imagens por classe: {val_counts}.")
    if overlap.size:
        raise RuntimeError(f"O split contém {overlap.size} índices em comum.")

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SPLIT_PATH, train_idx=train_idx, val_idx=val_idx)

    print(f"Split salvo em: {SPLIT_PATH}")
    print(f"Treino: {len(train_idx)} imagens")
    print(f"Validação: {len(val_idx)} imagens")
    print(f"Imagens de validação por classe: {val_counts.tolist()}")
    print(f"Interseção entre conjuntos: {overlap.size}")
    return SPLIT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="sobrescreve o split existente (use apenas com decisão do grupo)",
    )
    args = parser.parse_args()
    create_split(force=args.force)


if __name__ == "__main__":
    main()

"""Confere o fluxo de experimento com uma CNN pequena e dados reduzidos."""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from torch import nn
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.data import get_dataloaders
from src.experiment import RESULTS_DIR, run_experiment


class SimpleSmokeCNN(nn.Module):
    """CNN pequena apenas para verificar o pipeline; não é a CNN do trabalho."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, images):
        features = self.features(images).flatten(start_dim=1)
        return self.classifier(features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="usa 512 imagens de treino, 256 de validação e uma época",
    )
    args = parser.parse_args()

    train_count, val_count, batch_size, epochs = (
        (512, 256, 64, 1) if args.fast else (2048, 512, 128, 2)
    )
    name = "verificacao_rapida" if args.fast else "verificacao_completa"
    config = Config(
        experiment_name=name,
        seed=42,
        batch_size=batch_size,
        epochs=epochs,
        num_workers=0,
        device="auto",
    )

    full_train_loader, full_val_loader = get_dataloaders(config)
    train_subset = Subset(full_train_loader.dataset, range(train_count))
    val_subset = Subset(full_val_loader.dataset, range(val_count))
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    run_experiment(
        SimpleSmokeCNN(), config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    result_path = RESULTS_DIR / f"{name}.json"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / f"{name}_best.pt"
    if not result_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError("A verificação não gerou o JSON e o checkpoint esperados.")
    print(f"Verificação concluída. Resultados: {result_path}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

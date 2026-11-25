"""Uma época curta da MobileNetV2 congelada, só para testar o encaixe."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.experiment import run_experiment
from src.data import get_dataloaders
from src.transfer import MobileNetV2CIFAR
from src.utils import count_parameters, set_seed


def main() -> None:
    config = Config(
        experiment_name="p5_mobilenet_smoke",
        seed=42,
        batch_size=16,
        epochs=1,
        resize=224,
        normalization="imagenet",
        num_workers=0,
        device="auto",
    )
    set_seed(config.seed)
    model = MobileNetV2CIFAR(strategy="frozen", pretrained=True)
    print(count_parameters(model))

    full_train, full_val = get_dataloaders(config)
    train_loader = DataLoader(
        Subset(full_train.dataset, range(64)),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(full_val.dataset, range(32)),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    result = run_experiment(
        model, config, train_loader=train_loader, val_loader=val_loader
    )
    print(result["validation"]["accuracy"])
    print(result["parameters"])


if __name__ == "__main__":
    main()
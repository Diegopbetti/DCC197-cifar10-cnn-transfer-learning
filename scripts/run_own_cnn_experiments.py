from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.own_cnn_config import own_cnn_config
from src.experiment import run_experiment
from src.models.cnn import BaselineCNN
from src.utils import set_seed

def main():
    experimentos = {
        "cnn_baseline": {
            "base_width": 32,
            "depth": 3,
            "dropout": 0.5,
            "use_batch_norm": True,
        },
        "cnn_deep": {
            "base_width": 32,
            "depth": 4,
            "dropout": 0.5,
            "use_batch_norm": True,
        },
        "cnn_wide": {
            "base_width": 64,
            "depth": 3,
            "dropout": 0.5,
            "use_batch_norm": True,
        },
        "cnn_dropout": {
            "base_width": 32,
            "depth": 3,
            "dropout": 0.2,
            "use_batch_norm": True,
        },
    }

    for nome, architecture in experimentos.items():
        config = own_cnn_config(nome)
        set_seed(config.seed)
        model = BaselineCNN(**architecture)
        result = run_experiment(model, config)

        print(f"\nExperimento: {nome}")
        print("Acurácia:", result["validation"]["accuracy"])
        print("Parâmetros:", result["parameters"]["total"])
        print("Melhor época:", result["best_epoch"])


if __name__ == "__main__":
    main()
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.own_cnn_config import own_cnn_config
from src.experiment import run_experiment
from src.models.cnn import BaselineCNN
from src.utils import set_seed


def main():
    config = own_cnn_config("cnn_baseline")

    set_seed(config.seed)
    model = BaselineCNN(base_width=32, depth=3)

    result = run_experiment(model, config)
    print("Acurácia de validação:", result["validation"]["accuracy"])
    print("Parâmetros:", result["parameters"]["total"])
    print("Melhor época:", result["best_epoch"])


if __name__ == "__main__":
    main()
"""Treina a MobileNetV2 nas estratégias de transfer learning.

Usa os mesmos hiperparâmetros da CNN própria. A foto em 224 pixels e a
normalização ImageNet são o ajuste necessário para os pesos do ImageNet.
"""

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.own_cnn_config import own_cnn_config
from src.experiment import run_experiment
from src.transfer import STRATEGIES, MobileNetV2CIFAR
from src.utils import set_seed


def main() -> None:
    strategy = sys.argv[1] if len(sys.argv) > 1 else None
    if strategy is not None and strategy not in STRATEGIES:
        raise SystemExit(f"Estratégia desconhecida: {strategy}. Use uma de {STRATEGIES}.")

    chosen = (strategy,) if strategy is not None else STRATEGIES
    for name in chosen:
        config = replace(
            own_cnn_config(f"p5_mobilenetv2_{name}"),
            resize=224,
            normalization="imagenet",
        )
        set_seed(config.seed)
        model = MobileNetV2CIFAR(strategy=name, pretrained=True)
        result = run_experiment(model, config)

        print(f"\nExperimento: {config.experiment_name}")
        print("Acurácia de validação:", result["validation"]["accuracy"])
        print("Parâmetros:", result["parameters"])
        print("Melhor época:", result["best_epoch"])
        print("Tempo de treino (s):", result["timings"]["train_seconds"])
        print("Tempo de inferência por imagem (s):", result["timings"]["inference_seconds_per_image"])


if __name__ == "__main__":
    main()
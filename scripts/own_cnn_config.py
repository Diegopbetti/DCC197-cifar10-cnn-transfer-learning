"""Hiperparâmetros definidos para os experimentos de CNN própria.

A loss usada pelo loop compartilhado é CrossEntropyLoss.
"""

from src.config import Config


def own_cnn_config(experiment_name: str = "cnn_baseline") -> Config:
    """Retorna a configuração fixada para o baseline da CNN própria."""
    return Config(
        experiment_name=experiment_name,
        seed=42,
        optimizer="adam",
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        epochs=2,
        scheduler="cosine",
        augmentation=True,
        resize=None,
        normalization="cifar10",
        num_workers=2,
        device="auto",
    )

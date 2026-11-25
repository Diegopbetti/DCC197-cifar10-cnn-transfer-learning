"""MobileNetV2 adaptada às 10 classes do CIFAR-10.

As estratégias descrevem quais camadas recebem gradiente. O loop de treino
compartilhado só otimiza parâmetros com requires_grad=True e chama model.train()
a cada época, então o BatchNorm de camada congelada precisa permanecer em eval.
"""

from typing import Literal

from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


Strategy = Literal["frozen", "partial", "finetune", "scratch"]
STRATEGIES: tuple[Strategy, ...] = ("frozen", "partial", "finetune", "scratch")

NUM_CLASSES = 10
# features[14:] é o estágio de 160 canais, o bloco de 320 e a convolução 1x1 final.
PARTIAL_FROM = 14


class MobileNetV2CIFAR(nn.Module):
    """MobileNetV2 com classificador de 10 classes e uma estratégia de treino."""

    def __init__(self, strategy: Strategy = "frozen", *, pretrained: bool = True):
        super().__init__()
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Estratégia desconhecida: {strategy}. Use uma de {STRATEGIES}."
            )

        weights = None
        if strategy != "scratch" and pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1

        self.strategy = strategy
        self.backbone = mobilenet_v2(weights=weights)
        self._replace_classifier()
        self._apply_strategy(strategy)
        self._frozen_batchnorm_to_eval()

    def _replace_classifier(self) -> None:
        head = self.backbone.classifier
        if not isinstance(head, nn.Sequential) or not isinstance(head[-1], nn.Linear):
            raise TypeError("A cabeça da MobileNetV2 não tem o formato esperado.")
        head[-1] = nn.Linear(head[-1].in_features, NUM_CLASSES)

    def _apply_strategy(self, strategy: Strategy) -> None:
        if strategy in ("finetune", "scratch"):
            return
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False
        if strategy == "partial":
            for block in self.backbone.features[PARTIAL_FROM:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    def _frozen_batchnorm_to_eval(self) -> None:
        for module in self.modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            params = list(module.parameters())
            if params and not any(parameter.requires_grad for parameter in params):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._frozen_batchnorm_to_eval()
        return self

    def forward(self, images):
        return self.backbone(images)
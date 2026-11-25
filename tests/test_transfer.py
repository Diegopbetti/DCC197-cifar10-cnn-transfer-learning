"""Conferência da MobileNetV2 sem download de pesos e sem treino."""

import unittest

import torch
from torch import nn

from src.transfer import PARTIAL_FROM, MobileNetV2CIFAR
from src.utils import count_parameters


def _batchnorm_modules(root: nn.Module) -> list[nn.BatchNorm2d]:
    return [module for module in root.modules() if isinstance(module, nn.BatchNorm2d)]


class MobileNetV2TransferTests(unittest.TestCase):
    def test_strategy_desconhecida_falha(self) -> None:
        with self.assertRaisesRegex(ValueError, "Estratégia"):
            MobileNetV2CIFAR(strategy="lora", pretrained=False)  # type: ignore[arg-type]

    def test_saida_tem_dez_classes(self) -> None:
        model = MobileNetV2CIFAR(strategy="scratch", pretrained=False)
        logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_congelado_treina_somente_a_cabeca(self) -> None:
        model = MobileNetV2CIFAR(strategy="frozen", pretrained=False)
        counts = count_parameters(model)
        head = model.backbone.classifier[-1]
        head_params = head.weight.numel() + head.bias.numel()

        self.assertEqual(counts["trainable"], head_params)
        self.assertLess(counts["trainable"], counts["total"])
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.backbone.features.parameters())
        )

        model.train()
        self.assertTrue(all(not module.training for module in _batchnorm_modules(model.backbone.features)))

    def test_parcial_descongela_somente_o_final(self) -> None:
        model = MobileNetV2CIFAR(strategy="partial", pretrained=False)
        early = list(model.backbone.features[:PARTIAL_FROM].parameters())
        late = list(model.backbone.features[PARTIAL_FROM:].parameters())

        self.assertTrue(all(not parameter.requires_grad for parameter in early))
        self.assertTrue(all(parameter.requires_grad for parameter in late))

        counts = count_parameters(model)
        self.assertLess(counts["trainable"], counts["total"])
        self.assertGreater(counts["trainable"], model.backbone.classifier[-1].weight.numel())

        model.train()
        self.assertTrue(
            all(not module.training for module in _batchnorm_modules(model.backbone.features[:PARTIAL_FROM]))
        )
        self.assertTrue(
            all(module.training for module in _batchnorm_modules(model.backbone.features[PARTIAL_FROM:]))
        )

    def test_finetune_e_scratch_treinam_tudo(self) -> None:
        for strategy in ("finetune", "scratch"):
            with self.subTest(strategy=strategy):
                model = MobileNetV2CIFAR(strategy=strategy, pretrained=False)
                counts = count_parameters(model)
                self.assertEqual(counts["trainable"], counts["total"])
                self.assertGreater(counts["total"], 2_000_000)


if __name__ == "__main__":
    unittest.main()
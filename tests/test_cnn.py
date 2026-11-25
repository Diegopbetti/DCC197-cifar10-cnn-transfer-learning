"""Testes de formato de saída da CNN própria."""

import unittest

import torch

from src.models.cnn import BaselineCNN


class BaselineCNNTests(unittest.TestCase):
    def test_baseline_and_architectural_variants_return_ten_logits(self) -> None:
        images = torch.randn(4, 3, 32, 32)
        configurations = (
            {"base_width": 32, "depth": 3},
            {"base_width": 32, "depth": 2},
            {"base_width": 64, "depth": 3},
            {"base_width": 32, "depth": 3, "use_batch_norm": False},
        )

        for configuration in configurations:
            with self.subTest(configuration=configuration):
                model = BaselineCNN(**configuration).eval()
                with torch.inference_mode():
                    logits = model(images)

                self.assertEqual(logits.shape, (4, 10))

    def test_batch_norm_flag_controls_batch_norm_layers(self) -> None:
        model_with_batch_norm = BaselineCNN(depth=1, use_batch_norm=True)
        model_without_batch_norm = BaselineCNN(depth=1, use_batch_norm=False)

        self.assertEqual(
            sum(isinstance(layer, torch.nn.BatchNorm2d) for layer in model_with_batch_norm.modules()),
            2,
        )
        self.assertEqual(
            sum(isinstance(layer, torch.nn.BatchNorm2d) for layer in model_without_batch_norm.modules()),
            0,
        )


if __name__ == "__main__":
    unittest.main()

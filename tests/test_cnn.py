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
        )

        for configuration in configurations:
            with self.subTest(configuration=configuration):
                model = BaselineCNN(**configuration).eval()
                with torch.inference_mode():
                    logits = model(images)

                self.assertEqual(logits.shape, (4, 10))


if __name__ == "__main__":
    unittest.main()

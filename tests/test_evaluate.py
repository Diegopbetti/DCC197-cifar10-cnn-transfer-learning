"""Testes das métricas de avaliação e da matriz de confusão."""

import importlib
import importlib.util
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _evaluator_module(test_case: unittest.TestCase):
    module_spec = importlib.util.find_spec("src.evaluate")
    test_case.assertIsNotNone(module_spec, "src.evaluate deve fornecer as métricas")
    return importlib.import_module("src.evaluate")


class EvaluationTests(unittest.TestCase):
    def test_evaluate_returns_ordered_predictions_and_exact_accuracy(self) -> None:
        module = _evaluator_module(self)
        logits = torch.eye(10)[[0, 1, 2, 3]] * 10
        labels = torch.tensor([0, 1, 2, 0])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=3)

        result = module.evaluate(nn.Identity(), loader, torch.device("cpu"))

        self.assertEqual(result["y_pred"], [0, 1, 2, 3])
        self.assertEqual(result["y_true"], [0, 1, 2, 0])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertGreaterEqual(result["loss"], 0.0)

    def test_confusion_matrix_has_ten_classes_and_expected_cells(self) -> None:
        module = _evaluator_module(self)

        matrix = module.confusion_matrix_from(
            y_true=[0, 1, 1, 3],
            y_pred=[0, 0, 1, 3],
        )

        self.assertEqual(matrix.shape, (10, 10))
        self.assertEqual(int(matrix[0, 0]), 1)
        self.assertEqual(int(matrix[1, 0]), 1)
        self.assertEqual(int(matrix[1, 1]), 1)
        self.assertEqual(int(matrix[3, 3]), 1)
        self.assertEqual(int(matrix.sum()), 4)

    def test_evaluate_rejects_empty_loader(self) -> None:
        module = _evaluator_module(self)
        loader = DataLoader(
            TensorDataset(torch.empty(0, 10), torch.empty(0, dtype=torch.long)),
            batch_size=2,
        )

        with self.assertRaisesRegex(ValueError, "vazio"):
            module.evaluate(nn.Identity(), loader, "cpu")

    def test_evaluate_rejects_logits_without_ten_classes(self) -> None:
        module = _evaluator_module(self)
        loader = DataLoader(
            TensorDataset(torch.ones(2, 10), torch.tensor([0, 1])),
            batch_size=2,
        )

        with self.assertRaisesRegex(ValueError, "10"):
            module.evaluate(nn.Linear(10, 2), loader, "cpu")


if __name__ == "__main__":
    unittest.main()

"""Testes do loop de treinamento e do salvamento do melhor checkpoint."""

import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config


def _training_module(test_case: unittest.TestCase):
    module_spec = importlib.util.find_spec("src.train")
    test_case.assertIsNotNone(module_spec, "src.train deve fornecer o loop de treino")
    return importlib.import_module("src.train")


class TrainTests(unittest.TestCase):
    def _loaders(self, labels: list[int] | None = None, batch_size: int = 2):
        if labels is None:
            labels = [0, 1, 2, 3]
        images = torch.ones(len(labels), 10)
        targets = torch.tensor(labels, dtype=torch.long)
        loader = DataLoader(TensorDataset(images, targets), batch_size=batch_size)
        return loader, loader

    def test_history_has_one_entry_per_epoch_for_all_metrics(self) -> None:
        module = _training_module(self)
        train_loader, val_loader = self._loaders()
        model = nn.Linear(10, 10)
        config = Config(
            experiment_name="history-test", epochs=2, batch_size=2, device="cpu"
        )

        with tempfile.TemporaryDirectory() as directory, patch.object(
            module, "CHECKPOINT_DIR", Path(directory)
        ):
            result = module.train_model(model, config, train_loader, val_loader)
            self.assertTrue(Path(result["checkpoint_path"]).is_file())

        for key in (
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "lr",
            "epoch_time_seconds",
        ):
            self.assertEqual(len(result["history"][key]), 2, key)

    def test_checkpoint_contains_weights_from_highest_validation_accuracy(self) -> None:
        module = _training_module(self)
        train_loader, val_loader = self._loaders()
        model = nn.Linear(10, 10)
        config = Config(
            experiment_name="best-test", epochs=3, batch_size=2, device="cpu"
        )
        observed_weights = []
        accuracies = iter([0.25, 0.75, 0.50])

        def fake_evaluate(current_model, loader, device):
            observed_weights.append(
                {name: value.detach().clone() for name, value in current_model.state_dict().items()}
            )
            accuracy = next(accuracies)
            return {"loss": 1.0, "accuracy": accuracy, "y_pred": [], "y_true": []}

        with tempfile.TemporaryDirectory() as directory, patch.object(
            module, "CHECKPOINT_DIR", Path(directory)
        ), patch.object(module, "evaluate", side_effect=fake_evaluate):
            result = module.train_model(model, config, train_loader, val_loader)
            checkpoint = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)

        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["best_val_acc"], 0.75)
        for name, value in checkpoint.items():
            self.assertTrue(torch.equal(value, observed_weights[1][name]), name)

    def test_rejects_model_without_trainable_parameters_before_training(self) -> None:
        module = _training_module(self)
        train_loader, val_loader = self._loaders()
        model = nn.Linear(10, 10)
        model.requires_grad_(False)
        config = Config(experiment_name="frozen-test", epochs=1, device="cpu")

        with self.assertRaisesRegex(ValueError, "treináveis"):
            module.train_model(model, config, train_loader, val_loader)

    def test_training_loss_weights_final_partial_batch_by_example_count(self) -> None:
        module = _training_module(self)

        class FixedLogits(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.tensor(0.0))

            def forward(self, inputs):
                logits = torch.zeros(inputs.shape[0], 10, device=inputs.device)
                logits[:, 0] = 2.0 + self.anchor * 0.0
                return logits

        train_loader, val_loader = self._loaders([0, 0, 1], batch_size=2)
        model = FixedLogits()
        config = Config(
            experiment_name="partial-batch-test", epochs=1, batch_size=2,
            lr=0.01, device="cpu"
        )
        expected_loss = nn.CrossEntropyLoss()(model(torch.ones(3, 10)), torch.tensor([0, 0, 1])).item()

        with tempfile.TemporaryDirectory() as directory, patch.object(
            module, "CHECKPOINT_DIR", Path(directory)
        ):
            result = module.train_model(model, config, train_loader, val_loader)

        self.assertAlmostEqual(result["history"]["train_loss"][0], expected_loss, places=6)

    def test_empty_training_loader_fails_before_first_epoch(self) -> None:
        module = _training_module(self)
        empty_loader = DataLoader(
            TensorDataset(torch.empty(0, 10), torch.empty(0, dtype=torch.long)),
            batch_size=2,
        )
        _, val_loader = self._loaders()
        config = Config(experiment_name="empty-train-test", epochs=1, device="cpu")

        with self.assertRaisesRegex(ValueError, "vazio"):
            module.train_model(nn.Linear(10, 10), config, empty_loader, val_loader)


if __name__ == "__main__":
    unittest.main()

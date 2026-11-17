"""Testes da orquestração de experimentos sem acesso ao conjunto de teste."""

import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config


def _experiment_module(test_case: unittest.TestCase):
    module_spec = importlib.util.find_spec("src.experiment")
    test_case.assertIsNotNone(module_spec, "src.experiment deve orquestrar experimentos")
    return importlib.import_module("src.experiment")


class ExperimentTests(unittest.TestCase):
    def _loaders(self):
        images = torch.eye(10)[:4]
        labels = torch.tensor([0, 1, 2, 3])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        return loader, loader

    def test_optional_loaders_must_be_supplied_as_a_pair(self) -> None:
        module = _experiment_module(self)
        config = Config(experiment_name="pair-test", epochs=1, device="cpu")
        train_loader, _ = self._loaders()

        with self.assertRaisesRegex(ValueError, "juntos"):
            module.run_experiment(nn.Linear(10, 10), config, train_loader=train_loader)

    def test_injected_loaders_save_checkpoint_and_complete_json_without_test_loader(self) -> None:
        module = _experiment_module(self)
        train_loader, val_loader = self._loaders()
        config = Config(
            experiment_name="orchestration-test", epochs=1,
            batch_size=2, device="cpu", augmentation=False,
        )
        model = nn.Linear(10, 10)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / "checkpoints"
            results_dir = root / "results" / "runs"
            with patch.object(module, "RESULTS_DIR", results_dir), patch(
                "src.train.CHECKPOINT_DIR", checkpoint_dir
            ), patch("src.data.get_test_loader", side_effect=AssertionError("teste acessado")):
                result = module.run_experiment(
                    model, config, train_loader=train_loader, val_loader=val_loader
                )

            output_path = results_dir / "orchestration-test.json"
            checkpoint_path = checkpoint_dir / "orchestration-test_best.pt"
            self.assertTrue(output_path.is_file())
            self.assertTrue(checkpoint_path.is_file())
            serialized = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            set(result),
            {
                "config", "history", "best_val_acc", "best_epoch", "validation",
                "parameters", "timings", "hardware",
            },
        )
        self.assertEqual(result, serialized)
        self.assertEqual(len(result["validation"]["confusion_matrix"]), 10)
        self.assertEqual(len(result["validation"]["confusion_matrix"][0]), 10)
        self.assertIn("loss", result["validation"])
        self.assertIn("accuracy", result["validation"])
        self.assertIn("total", result["parameters"])
        self.assertIn("trainable", result["parameters"])
        self.assertIn("train_seconds", result["timings"])
        self.assertIn("inference_seconds_per_image", result["timings"])
        self.assertIn("python_version", result["hardware"])

    def test_failed_run_does_not_create_a_complete_result_json(self) -> None:
        module = _experiment_module(self)
        train_loader, val_loader = self._loaders()
        config = Config(experiment_name="failed-run", epochs=1, device="cpu")

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory) / "results"
            with patch.object(module, "RESULTS_DIR", results_dir), patch.object(
                module, "train_model", side_effect=RuntimeError("falha simulada")
            ):
                with self.assertRaisesRegex(RuntimeError, "simulada"):
                    module.run_experiment(
                        nn.Linear(10, 10), config,
                        train_loader=train_loader, val_loader=val_loader,
                    )
            self.assertFalse((results_dir / "failed-run.json").exists())

    def test_default_loaders_are_created_after_seed_is_set(self) -> None:
        module = _experiment_module(self)
        train_loader, val_loader = self._loaders()
        config = Config(experiment_name="seed-order", epochs=1, device="cpu")
        sequence = []

        def seed(seed_value):
            sequence.append(("seed", seed_value))

        def loaders(received_config):
            sequence.append(("loaders", received_config.seed))
            return train_loader, val_loader

        with tempfile.TemporaryDirectory() as directory, patch.object(
            module, "RESULTS_DIR", Path(directory) / "results"
        ), patch("src.train.CHECKPOINT_DIR", Path(directory) / "checkpoints"), patch.object(
            module, "set_seed", side_effect=seed
        ), patch.object(module, "get_dataloaders", side_effect=loaders):
            module.run_experiment(nn.Linear(10, 10), config)

        self.assertEqual(sequence, [("seed", config.seed), ("loaders", config.seed)])


if __name__ == "__main__":
    unittest.main()

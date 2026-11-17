"""Testes dos utilitários de resultados e validação da configuração."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import src.utils as utils
from src.config import Config


class ConfigAndUtilityTests(unittest.TestCase):
    def test_config_rejects_unknown_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "device"):
            Config(experiment_name="teste", device="quantum")

    def test_count_parameters_separates_frozen_and_trainable(self) -> None:
        model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        for parameter in model[0].parameters():
            parameter.requires_grad = False

        self.assertTrue(hasattr(utils, "count_parameters"))
        result = utils.count_parameters(model)

        self.assertEqual(result, {"total": 13, "trainable": 4})

    def test_measure_inference_time_returns_seconds_per_image(self) -> None:
        model = nn.Linear(2, 2)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )

        self.assertTrue(hasattr(utils, "measure_inference_time"))
        elapsed = utils.measure_inference_time(model, loader, torch.device("cpu"))

        self.assertTrue(math.isfinite(elapsed))
        self.assertGreaterEqual(elapsed, 0.0)

    def test_hardware_info_reports_runtime_versions_and_device(self) -> None:
        self.assertTrue(hasattr(utils, "get_hardware_info"))
        result = utils.get_hardware_info()

        self.assertTrue(
            {"python_version", "torch_version", "cuda_version", "device_type", "device_name"}
            <= result.keys()
        )

    def test_save_run_creates_parent_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runs" / "result.json"
            result = {"accuracy": 0.75, "epoch": 3}

            self.assertTrue(hasattr(utils, "save_run"))
            utils.save_run(result, path)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), result)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA não disponível neste ambiente")
    def test_measure_inference_time_supports_cuda(self) -> None:
        model = nn.Linear(2, 2).cuda()
        loader = DataLoader(
            TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )

        elapsed = utils.measure_inference_time(model, loader, torch.device("cuda"))

        self.assertTrue(math.isfinite(elapsed))
        self.assertGreaterEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()

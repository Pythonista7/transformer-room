from __future__ import annotations

import unittest
from unittest import mock

import torch

from src.train import _read_peak_cuda_memory_gib


class TrainPeakMemoryGatingTests(unittest.TestCase):
    def test_skips_cuda_queries_when_capture_disabled(self) -> None:
        with (
            mock.patch("torch.cuda.max_memory_allocated") as allocated_mock,
            mock.patch("torch.cuda.max_memory_reserved") as reserved_mock,
        ):
            peak_gib, reserved_gib = _read_peak_cuda_memory_gib(
                device=torch.device("cuda"),
                should_capture=False,
            )

        self.assertIsNone(peak_gib)
        self.assertIsNone(reserved_gib)
        allocated_mock.assert_not_called()
        reserved_mock.assert_not_called()

    def test_skips_cuda_queries_on_non_cuda_device(self) -> None:
        with (
            mock.patch("torch.cuda.max_memory_allocated") as allocated_mock,
            mock.patch("torch.cuda.max_memory_reserved") as reserved_mock,
        ):
            peak_gib, reserved_gib = _read_peak_cuda_memory_gib(
                device=torch.device("cpu"),
                should_capture=True,
            )

        self.assertIsNone(peak_gib)
        self.assertIsNone(reserved_gib)
        allocated_mock.assert_not_called()
        reserved_mock.assert_not_called()

    def test_reads_cuda_queries_only_when_capture_enabled(self) -> None:
        with (
            mock.patch(
                "torch.cuda.max_memory_allocated",
                return_value=3 * (1024**3),
            ) as allocated_mock,
            mock.patch(
                "torch.cuda.max_memory_reserved",
                return_value=5 * (1024**3),
            ) as reserved_mock,
        ):
            peak_gib, reserved_gib = _read_peak_cuda_memory_gib(
                device=torch.device("cuda"),
                should_capture=True,
            )

        self.assertEqual(peak_gib, 3.0)
        self.assertEqual(reserved_gib, 5.0)
        allocated_mock.assert_called_once_with(torch.device("cuda"))
        reserved_mock.assert_called_once_with(torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()

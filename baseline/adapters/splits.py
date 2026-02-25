from __future__ import annotations

import random

from torch.utils.data import Dataset, Subset

from baseline.core.config import HoldoutSplitConfig
from baseline.core.registry import register_split_adapter


class HoldoutSplitAdapter:
    def split(self, dataset: Dataset, cfg: HoldoutSplitConfig) -> tuple[Dataset, Dataset]:
        sample_count = len(dataset)
        if sample_count < 2:
            raise ValueError(
                f"Need at least 2 dataset samples for train/val split, got {sample_count}."
            )

        indices = list(range(sample_count))
        if cfg.shuffle:
            rng = random.Random(cfg.seed)
            rng.shuffle(indices)

        train_size = int(sample_count * cfg.train_fraction)
        train_size = max(1, min(sample_count - 1, train_size))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        return Subset(dataset, train_indices), Subset(dataset, val_indices)


register_split_adapter("holdout", HoldoutSplitAdapter())

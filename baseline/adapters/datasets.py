from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from baseline.core.config import HFTextDatasetConfig, LocalTextDatasetConfig
from baseline.core.registry import register_dataset_adapter
from baseline.core.types import TextCorpus


def _normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if item is None:
                continue
            normalized = item.strip() if isinstance(item, str) else str(item).strip()
            if normalized:
                parts.append(normalized)
        return "\n".join(parts).strip()
    return str(value).strip()


def _infer_text_field_from_sample(sample: Mapping[str, Any]) -> str:
    preferred_fields = ("text", "content", "article", "body", "document", "sentence")

    for field in preferred_fields:
        value = sample.get(field)
        if isinstance(value, str):
            return field
        if isinstance(value, list) and (
            not value or all(isinstance(item, str) for item in value)
        ):
            return field

    for field, value in sample.items():
        if isinstance(value, str):
            return field
        if isinstance(value, list) and (
            not value or all(isinstance(item, str) for item in value)
        ):
            return field

    raise ValueError(
        "Could not infer a text field from Hugging Face dataset rows. "
        "Set dataset.text_field in config."
    )


class LocalTextDatasetAdapter:
    def load(self, cfg: LocalTextDatasetConfig) -> TextCorpus:
        dataset_path = Path(cfg.path).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Local dataset file not found: {dataset_path}")

        corpus = dataset_path.read_text(encoding="utf-8")
        if not corpus:
            raise ValueError(f"Local dataset is empty: {dataset_path}")

        delimiter = cfg.segment_delimiter
        segments = corpus.split(delimiter) if delimiter else [corpus]
        segments = [segment for segment in segments if segment.strip()]
        if not segments:
            raise ValueError(f"No non-empty segments found in local dataset: {dataset_path}")

        print(f"Loaded local dataset from: {dataset_path}")
        return TextCorpus(
            full_text=corpus,
            segments=segments,
            source_description=f"local_text:{dataset_path}",
        )


class HFTextDatasetAdapter:
    def load(self, cfg: HFTextDatasetConfig) -> TextCorpus:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Hugging Face dataset support requires the `datasets` package. "
                "Install it with `pip install datasets`."
            ) from exc

        load_kwargs: dict[str, Any] = {
            "path": cfg.dataset_name,
            "split": cfg.split,
            "streaming": cfg.streaming,
        }
        if cfg.dataset_config:
            load_kwargs["name"] = cfg.dataset_config

        dataset = load_dataset(**load_kwargs)
        iterator = iter(dataset)

        try:
            first_row = next(iterator)
        except StopIteration as exc:
            raise ValueError(
                f"Hugging Face dataset '{cfg.dataset_name}' split '{cfg.split}' is empty."
            ) from exc

        if not isinstance(first_row, Mapping):
            raise ValueError("Expected Hugging Face dataset rows to be dict-like objects.")

        text_field = cfg.text_field or _infer_text_field_from_sample(first_row)
        if text_field not in first_row:
            available = ", ".join(first_row.keys())
            raise ValueError(
                f"dataset.text_field '{text_field}' not found in rows. "
                f"Available fields: {available}"
            )

        segments: list[str] = []
        first_text = _normalize_text_value(first_row.get(text_field))
        if first_text:
            segments.append(first_text)

        for row in iterator:
            if cfg.max_rows > 0 and len(segments) >= cfg.max_rows:
                break
            if not isinstance(row, Mapping):
                continue
            text = _normalize_text_value(row.get(text_field))
            if text:
                segments.append(text)

        if cfg.max_rows > 0:
            segments = segments[: cfg.max_rows]

        if not segments:
            raise ValueError(
                f"No usable text found in field '{text_field}' for "
                f"dataset '{cfg.dataset_name}' split '{cfg.split}'."
            )

        descriptor = cfg.dataset_name
        if cfg.dataset_config:
            descriptor = f"{cfg.dataset_name}/{cfg.dataset_config}"
        print(
            f"Loaded Hugging Face dataset: {descriptor} | "
            f"split={cfg.split} | text_field={text_field} | rows={len(segments):,}"
        )
        return TextCorpus(
            full_text="\n\n".join(segments),
            segments=segments,
            source_description=f"hf_text:{descriptor}:{cfg.split}:{text_field}",
        )


register_dataset_adapter("local_text", LocalTextDatasetAdapter())
register_dataset_adapter("hf_text", HFTextDatasetAdapter())

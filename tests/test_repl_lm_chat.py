from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.components.models.baseline_model import BaselineModel
from src.core.types import SpecialTokenIds, VocabInfo


def load_repl_module():
    module_path = Path(__file__).resolve().parents[1] / "repl-lm-chat.py"
    spec = importlib.util.spec_from_file_location("repl_lm_chat", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeNotFoundError(Exception):
    pass


class FakeArtifact:
    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path

    def download(self, root: str) -> str:
        target_dir = Path(root) / self.source_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source_path, target_dir / self.source_path.name)
        return str(target_dir)


class FakeRun:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeApi:
    def __init__(
        self,
        *,
        default_entity: str,
        runs: dict[str, FakeRun],
        artifacts: dict[str, FakeArtifact],
    ) -> None:
        self.default_entity = default_entity
        self._runs = runs
        self._artifacts = artifacts
        self.artifact_requests: list[str] = []

    def run(self, ref: str) -> FakeRun:
        if ref not in self._runs:
            raise FakeNotFoundError(f"Run not found: {ref}")
        return self._runs[ref]

    def artifact(self, ref: str) -> FakeArtifact:
        self.artifact_requests.append(ref)
        if ref not in self._artifacts:
            raise FakeNotFoundError(f"Artifact not found: {ref}")
        return self._artifacts[ref]


class FakeHFTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        if not text:
            return {"input_ids": []}
        return {"input_ids": [0]}

    def decode(self, ids: list[int], **_: object) -> str:
        return " ".join(str(token_id) for token_id in ids)


def build_fake_hf_bundle() -> SimpleNamespace:
    vocab = VocabInfo(
        token_to_id={
            "tok0": 0,
            "tok1": 1,
            "tok2": 2,
            "tok3": 3,
            "<EOS>": 4,
            "<PAD>": 5,
            "<UNK>": 6,
        },
        id_to_token=["tok0", "tok1", "tok2", "tok3", "<EOS>", "<PAD>", "<UNK>"],
        special=SpecialTokenIds(
            vocab_size=7,
            eos_id=4,
            pad_id=5,
            unk_id=6,
            base_vocab_size=4,
            num_special_tokens=3,
        ),
    )
    return SimpleNamespace(
        tokenizer=FakeHFTokenizer(),
        vocab=vocab,
    )


def write_phase1_like_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = BaselineModel(
        vocab_size=7,
        d_model=8,
        n_heads=2,
        layers=1,
        dropout=0.25,
        attention_impl="sdpa",
        norm_placement="pre",
        enable_weight_tying=True,
        pad_id=5,
    )

    model_path = tmp_path / "baseline_model.pt"
    torch.save(model.state_dict(), model_path)

    run_config_path = tmp_path / "run_config.json"
    run_config_path.write_text(
        json.dumps(
            {
                "model": {
                    "name": "baseline_decoder",
                    "d_model": 8,
                    "n_heads": 2,
                    "layers": 1,
                    "dropout": 0.25,
                    "attention_impl": "sdpa",
                    "norm_placement": "pre",
                    "enable_weight_tying": True,
                },
                "tokenizer": {
                    "name": "hf_pretrained",
                    "pretrained_name_or_path": "gpt2",
                    "use_fast": True,
                    "revision": "main",
                    "trust_remote_code": False,
                },
            }
        ),
        encoding="utf-8",
    )

    inference_config_path = tmp_path / "inference_config.json"
    inference_config_path.write_text(
        json.dumps(
            {
                "model_name": "baseline_decoder",
                "tokenizer_name": "hf_pretrained",
                "tokenizer_source": "gpt2",
                "tokenizer_revision": "main",
                "base_vocab_size": 4,
                "num_special_tokens": 3,
                "vocab_size": 7,
                "eos_id": 4,
                "pad_id": 5,
                "unk_id": 6,
                "d_model": 8,
                "n_heads": 2,
                "layers": 1,
                "attention_impl": "sdpa",
                "training_seq_len": 64,
            }
        ),
        encoding="utf-8",
    )

    return model_path, run_config_path, inference_config_path


class ReplLmChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repl = load_repl_module()

    def test_resolve_wandb_run_reference_accepts_full_path(self) -> None:
        api = SimpleNamespace(default_entity="fallback-entity")
        resolved = self.repl.resolve_wandb_run_reference(
            "demo-entity/demo-project/abc12345",
            api=api,
            explicit_entity=None,
            explicit_project=None,
        )
        self.assertEqual(("demo-entity", "demo-project", "abc12345"), resolved)

    def test_resolve_wandb_run_reference_accepts_bare_id_with_defaults(self) -> None:
        api = SimpleNamespace(default_entity="default-entity")
        resolved = self.repl.resolve_wandb_run_reference(
            "abc12345",
            api=api,
            explicit_entity=None,
            explicit_project="project-x",
        )
        self.assertEqual(("default-entity", "project-x", "abc12345"), resolved)

    def test_resolve_wandb_run_reference_rejects_invalid_shape(self) -> None:
        api = SimpleNamespace(default_entity="default-entity")
        with self.assertRaises(ValueError):
            self.repl.resolve_wandb_run_reference(
                "entity/project",
                api=api,
                explicit_entity=None,
                explicit_project=None,
            )

    def test_resolve_wandb_artifacts_downloads_and_falls_back_to_latest_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path, run_config_path, inference_config_path = write_phase1_like_artifacts(tmp_path)
            api = FakeApi(
                default_entity="default-entity",
                runs={
                    "default-entity/transformer-room-baseline/run-123": FakeRun("ph-1-stg-1")
                },
                artifacts={
                    "default-entity/transformer-room-baseline/ph-1-stg-1-model:latest": FakeArtifact(model_path),
                    "default-entity/transformer-room-baseline/ph-1-stg-1-run-config:latest": FakeArtifact(run_config_path),
                    "default-entity/transformer-room-baseline/ph-1-stg-1-inference-config:latest": FakeArtifact(inference_config_path),
                },
            )
            args = self.repl.parse_args(
                [
                    "--wandb-run",
                    "run-123",
                    "--wandb-cache-dir",
                    str(tmp_path / "cache"),
                ]
            )

            with patch.object(self.repl, "get_wandb_api", return_value=api):
                resolved_model_path, run_config, inference_config = self.repl.resolve_wandb_artifacts(args)

            self.assertTrue(resolved_model_path.exists())
            self.assertEqual("baseline_decoder", run_config["model"]["name"])
            self.assertEqual(64, inference_config["training_seq_len"])
            self.assertIn(
                "default-entity/transformer-room-baseline/ph-1-stg-1-model:final",
                api.artifact_requests,
            )
            self.assertIn(
                "default-entity/transformer-room-baseline/ph-1-stg-1-model:latest",
                api.artifact_requests,
            )

    def test_resolve_wandb_artifacts_errors_when_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path, _, _ = write_phase1_like_artifacts(tmp_path)
            api = FakeApi(
                default_entity="default-entity",
                runs={
                    "default-entity/transformer-room-baseline/run-123": FakeRun("ph-1-stg-1")
                },
                artifacts={
                    "default-entity/transformer-room-baseline/ph-1-stg-1-model:final": FakeArtifact(model_path),
                },
            )
            args = self.repl.parse_args(
                [
                    "--wandb-run",
                    "run-123",
                    "--wandb-cache-dir",
                    str(tmp_path / "cache"),
                ]
            )

            with patch.object(self.repl, "get_wandb_api", return_value=api):
                with self.assertRaises(FileNotFoundError):
                    self.repl.resolve_wandb_artifacts(args)

    def test_load_model_and_tokenizer_from_wandb_rehydrates_phase1_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path, run_config_path, inference_config_path = write_phase1_like_artifacts(tmp_path)
            api = FakeApi(
                default_entity="default-entity",
                runs={
                    "default-entity/transformer-room-baseline/run-123": FakeRun("ph-1-stg-1")
                },
                artifacts={
                    "default-entity/transformer-room-baseline/ph-1-stg-1-model:final": FakeArtifact(model_path),
                    "default-entity/transformer-room-baseline/ph-1-stg-1-run-config:latest": FakeArtifact(run_config_path),
                    "default-entity/transformer-room-baseline/ph-1-stg-1-inference-config:latest": FakeArtifact(inference_config_path),
                },
            )
            args = self.repl.parse_args(
                [
                    "--wandb-run",
                    "run-123",
                    "--wandb-cache-dir",
                    str(tmp_path / "cache"),
                ]
            )

            with patch.object(self.repl, "get_wandb_api", return_value=api):
                with patch.object(
                    self.repl,
                    "build_hf_pretrained_tokenizer_bundle",
                    return_value=build_fake_hf_bundle(),
                ):
                    model, tokenizer, token_to_id, id_to_token, config, _, eos_id, pad_id, unk_id = (
                        self.repl.load_model_and_tokenizer_from_wandb(args)
                    )

            self.assertIsInstance(model, BaselineModel)
            self.assertEqual("pre", model.norm_placement)
            self.assertTrue(model.enable_weight_tying)
            self.assertEqual("sdpa", model.attention_impl)
            self.assertEqual(0.25, model.dropout)
            self.assertEqual(64, config["training_seq_len"])
            self.assertEqual(4, eos_id)
            self.assertEqual(5, pad_id)
            self.assertEqual(6, unk_id)
            self.assertEqual("gpt2", config["tokenizer_source"])
            self.assertIsInstance(tokenizer, FakeHFTokenizer)
            self.assertEqual(7, len(id_to_token))
            self.assertEqual(0, token_to_id["tok0"])

    def test_main_wandb_repl_smoke_runs_without_local_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path, run_config_path, inference_config_path = write_phase1_like_artifacts(tmp_path)
            api = FakeApi(
                default_entity="default-entity",
                runs={
                    "default-entity/transformer-room-baseline/run-123": FakeRun("ph-1-stg-1")
                },
                artifacts={
                    "default-entity/transformer-room-baseline/ph-1-stg-1-model:final": FakeArtifact(model_path),
                    "default-entity/transformer-room-baseline/ph-1-stg-1-run-config:latest": FakeArtifact(run_config_path),
                    "default-entity/transformer-room-baseline/ph-1-stg-1-inference-config:latest": FakeArtifact(inference_config_path),
                },
            )

            with patch.object(self.repl, "get_wandb_api", return_value=api):
                with patch.object(
                    self.repl,
                    "build_hf_pretrained_tokenizer_bundle",
                    return_value=build_fake_hf_bundle(),
                ):
                    with patch.object(self.repl, "generate", return_value="hello back") as mock_generate:
                        with patch("builtins.input", side_effect=["hello", "/quit"]):
                            result = self.repl.main(
                                [
                                    "--wandb-run",
                                    "run-123",
                                    "--wandb-cache-dir",
                                    str(tmp_path / "cache"),
                                    "--prompt-format",
                                    "chat",
                                ]
                            )

            self.assertEqual(0, result)
            self.assertEqual(1, mock_generate.call_count)


if __name__ == "__main__":
    unittest.main()

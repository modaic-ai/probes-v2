from pathlib import Path

import pytest
import torch
from datasets import Dataset

from probes2.finetuning.linear_head import load_linear_head_checkpoint
import probes2.finetuning.tune as tune_module
from probes2.finetuning.tune import (
    build_stage_rows,
    default_tune_config_path,
    derive_annotated_target,
    load_tune_config,
    main,
    resolve_stage_config,
)


def test_default_tune_config_loads_correctly() -> None:
    path = default_tune_config_path()
    assert path.exists()
    config = load_tune_config(str(path))

    self_consistency = resolve_stage_config(config, "self_consistency")
    cross_consistency = resolve_stage_config(config, "cross_consistency")
    annotated = resolve_stage_config(config, "annotated")

    assert self_consistency.lr == pytest.approx(1e-5)
    assert self_consistency.dropout == pytest.approx(0.1)
    assert self_consistency.train_pct == pytest.approx(1.0)

    assert cross_consistency.lr == pytest.approx(1e-4)
    assert cross_consistency.dropout == pytest.approx(0.0)
    assert cross_consistency.train_pct == pytest.approx(1.0)

    assert annotated.lr == pytest.approx(1e-3)
    assert annotated.dropout == pytest.approx(0.0)
    assert annotated.train_pct == pytest.approx(0.8)


def test_stage_config_overrides_global_defaults() -> None:
    config = {
        "lr": "1e-5",
        "dropout": 0.3,
        "epochs": 2,
        "batch_size": 4,
        "seed": 99,
        "annotated": {
            "lr": "5e-4",
            "dropout": 0.0,
        },
    }

    stage = resolve_stage_config(config, "annotated")

    assert stage.lr == pytest.approx(5e-4)
    assert stage.dropout == pytest.approx(0.0)
    assert stage.epochs == 2
    assert stage.batch_size == 4
    assert stage.seed == 99


def test_derive_annotated_target_prefers_correct_and_falls_back_to_ground_truth() -> None:
    assert derive_annotated_target({"correct": True}) == pytest.approx(1.0)

    matched = derive_annotated_target(
        {
            "ground_truth": '{"is_spam": "spam"}',
            "prediction": '{"is_spam": "spam"}',
        }
    )
    assert matched == pytest.approx(1.0)

    mismatched = derive_annotated_target(
        {
            "ground_truth": '{"is_spam": "spam"}',
            "is_spam": "not spam",
        }
    )
    assert mismatched == pytest.approx(0.0)


def test_build_stage_rows_collects_all_three_stages() -> None:
    dataset = Dataset.from_dict(
        {
            "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "sc_confidence": [0.9, None, None],
            "cc_confidence": [None, 0.25, None],
            "correct": [None, None, 1],
        }
    )

    stage_rows, input_dim = build_stage_rows(dataset)

    assert input_dim == 2
    assert stage_rows["self_consistency"].targets == [pytest.approx(0.9)]
    assert stage_rows["cross_consistency"].targets == [pytest.approx(0.25)]
    assert stage_rows["annotated"].targets == [pytest.approx(1.0)]


def test_tune_main_writes_linear_head_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset.from_dict(
        {
            "embeddings": [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
                [0.2, 0.8],
            ],
            "sc_confidence": [0.95, 0.05, None, None],
            "cc_confidence": [None, None, 0.85, 0.15],
            "correct": [1, 0, 1, 0],
        }
    )
    dataset_path = tmp_path / "reflected.parquet"
    checkpoint_path = tmp_path / "head.pt"
    config_path = str(default_tune_config_path())
    monkeypatch.setattr(tune_module, "load_input_dataset", lambda _: dataset)

    main(
        [
            "--model",
            "modaic/test-probe",
            "--dataset",
            str(dataset_path),
            "--config",
            config_path,
            "--checkpoint",
            str(checkpoint_path),
            "--device",
            "cpu",
        ]
    )

    payload = load_linear_head_checkpoint(checkpoint_path)
    assert checkpoint_path.exists()
    assert payload["base_model"] == "modaic/test-probe"
    assert payload["input_dim"] == 2
    assert set(payload["state_dict"]) == {"weight", "bias"}
    assert torch.is_tensor(payload["state_dict"]["weight"])

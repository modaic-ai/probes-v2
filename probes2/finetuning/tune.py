from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score

from probes2.finetuning.linear_head import LinearProbeHead, save_linear_head_checkpoint
from probes2.inference.predict import load_input_dataset


SPECIAL_STAGE_KEYS = ("self_consistency", "cross_consistency", "annotated")
TRAINING_DEFAULTS = {
    "lr": 1e-5,
    "dropout": 0.0,
    "epochs": 5,
    "batch_size": 256,
    "weight_decay": 0.0,
    "seed": 0,
    "train_pct": 1.0,
}


@dataclass(frozen=True)
class StageConfig:
    lr: float
    dropout: float
    epochs: int
    batch_size: int
    weight_decay: float
    seed: int
    train_pct: float


@dataclass
class StageRows:
    embeddings: list[list[float]]
    targets: list[float]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a probe linear head from reflected embeddings"
    )
    parser.add_argument("--model", required=True, help="Base probe model ID or path")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to reflected dataset (overrides config YAML 'dataset' key)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Output path for the tuned linear head checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to tune config YAML",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Training device (auto-detected if omitted)",
    )
    return parser.parse_args(argv)


def default_tune_config_path() -> Path:
    return Path(__file__).with_name("default_tune.yaml")


def validate_args(args: argparse.Namespace) -> None:
    if Path(args.checkpoint).suffix.lower() != ".pt":
        raise ValueError(
            f"--checkpoint must point to a .pt file, got '{args.checkpoint}'"
        )


def resolve_device(device_arg: str | None) -> str:
    if device_arg is not None:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tune_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    with path.open() as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Tune config at {path} must be a YAML mapping")
    return config


NON_TRAINING_KEYS = ("dataset",)


def split_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    global_defaults: dict[str, Any] = {}
    stage_overrides: dict[str, dict[str, Any]] = {
        stage: {} for stage in SPECIAL_STAGE_KEYS
    }
    for key, value in config.items():
        if key in SPECIAL_STAGE_KEYS:
            if value is None:
                stage_overrides[key] = {}
            elif isinstance(value, dict):
                stage_overrides[key] = dict(value)
            else:
                raise ValueError(f"Stage override '{key}' must be a mapping")
        elif key not in NON_TRAINING_KEYS:
            global_defaults[key] = value
    return global_defaults, stage_overrides


def _as_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc


def _as_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc


def resolve_stage_config(config: dict[str, Any], stage_name: str) -> StageConfig:
    global_defaults, stage_overrides = split_config(config)
    merged: dict[str, Any] = dict(TRAINING_DEFAULTS)
    merged.update(global_defaults)
    merged.update(stage_overrides.get(stage_name, {}))

    lr = _as_float(merged["lr"], f"{stage_name}.lr")
    dropout = _as_float(merged["dropout"], f"{stage_name}.dropout")
    epochs = _as_int(merged["epochs"], f"{stage_name}.epochs")
    batch_size = _as_int(merged["batch_size"], f"{stage_name}.batch_size")
    weight_decay = _as_float(merged["weight_decay"], f"{stage_name}.weight_decay")
    seed = _as_int(merged["seed"], f"{stage_name}.seed")
    train_pct = _as_float(merged["train_pct"], f"{stage_name}.train_pct")

    if lr <= 0:
        raise ValueError(f"{stage_name}.lr must be > 0, got {lr}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"{stage_name}.dropout must be in [0, 1), got {dropout}")
    if epochs < 1:
        raise ValueError(f"{stage_name}.epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"{stage_name}.batch_size must be >= 1, got {batch_size}")
    if weight_decay < 0:
        raise ValueError(
            f"{stage_name}.weight_decay must be >= 0, got {weight_decay}"
        )
    if not 0.0 < train_pct <= 1.0:
        raise ValueError(
            f"{stage_name}.train_pct must be in (0, 1], got {train_pct}"
        )

    return StageConfig(
        lr=lr,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        seed=seed,
        train_pct=train_pct,
    )


def resolve_global_seed(config: dict[str, Any]) -> int:
    global_defaults, _ = split_config(config)
    return _as_int(global_defaults.get("seed", TRAINING_DEFAULTS["seed"]), "seed")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _coerce_probability(value: Any, field_name: str) -> float | None:
    if _is_missing(value):
        return None
    probability = _as_float(value, field_name)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {probability}")
    return probability


def _coerce_correct_label(value: Any) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = _as_float(value, "correct")
        if numeric in (0.0, 1.0):
            return numeric
        raise ValueError(f"correct must be binary, got {value!r}")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "correct"}:
            return 1.0
        if normalized in {"0", "false", "no", "n", "incorrect"}:
            return 0.0
    raise ValueError(f"correct must be binary, got {value!r}")


def _parse_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if _is_missing(value):
        raise ValueError(f"{field_name} is missing")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{field_name} must be valid JSON when stored as a string"
            ) from exc
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{field_name} must be a mapping, got {type(value).__name__}")


def _normalize_label_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [_normalize_label_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_label_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def _prediction_mapping_for_row(
    row: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    if not _is_missing(row.get("prediction")):
        return _parse_mapping(row["prediction"], "prediction")
    return {key: row.get(key) for key in ground_truth}


def derive_annotated_target(row: dict[str, Any]) -> float | None:
    correct = _coerce_correct_label(row.get("correct"))
    if correct is not None:
        return correct

    if _is_missing(row.get("ground_truth")):
        return None

    ground_truth = _parse_mapping(row["ground_truth"], "ground_truth")
    if not ground_truth:
        raise ValueError("ground_truth must include at least one output field")

    prediction = _prediction_mapping_for_row(row, ground_truth)
    matches = all(
        _normalize_label_value(prediction.get(field))
        == _normalize_label_value(expected)
        for field, expected in ground_truth.items()
    )
    return float(matches)


def _extract_embedding(row: dict[str, Any], row_index: int) -> list[float]:
    raw_embedding = row.get("embeddings")
    if _is_missing(raw_embedding):
        raise ValueError(f"Row {row_index} is missing embeddings")
    if isinstance(raw_embedding, (str, bytes)):
        raise ValueError(
            f"Row {row_index} embeddings must be a numeric sequence, not {type(raw_embedding).__name__}"
        )
    try:
        embedding = [float(value) for value in raw_embedding]
    except TypeError as exc:
        raise ValueError(
            f"Row {row_index} embeddings must be a numeric sequence"
        ) from exc
    if not embedding:
        raise ValueError(f"Row {row_index} embeddings cannot be empty")
    return embedding


def build_stage_rows(dataset: Dataset) -> tuple[dict[str, StageRows], int]:
    if "embeddings" not in dataset.column_names:
        raise ValueError(
            f"Input dataset must contain an 'embeddings' column. Found: {dataset.column_names}"
        )

    rows_by_stage = {
        stage: StageRows(embeddings=[], targets=[])
        for stage in SPECIAL_STAGE_KEYS
    }
    input_dim: int | None = None
    columns = set(dataset.column_names)

    for row_index, raw_row in enumerate(dataset):
        row = dict(raw_row)
        embedding = _extract_embedding(row, row_index)
        if input_dim is None:
            input_dim = len(embedding)
        elif len(embedding) != input_dim:
            raise ValueError(
                f"Row {row_index} embedding width {len(embedding)} does not match expected {input_dim}"
            )

        if "sc_confidence" in columns:
            sc_target = _coerce_probability(row.get("sc_confidence"), "sc_confidence")
            if sc_target is not None:
                rows_by_stage["self_consistency"].embeddings.append(embedding)
                rows_by_stage["self_consistency"].targets.append(sc_target)

        if "cc_confidence" in columns:
            cc_target = _coerce_probability(row.get("cc_confidence"), "cc_confidence")
            if cc_target is not None:
                rows_by_stage["cross_consistency"].embeddings.append(embedding)
                rows_by_stage["cross_consistency"].targets.append(cc_target)

        annotated_target = derive_annotated_target(row)
        if annotated_target is not None:
            rows_by_stage["annotated"].embeddings.append(embedding)
            rows_by_stage["annotated"].targets.append(annotated_target)

    if input_dim is None:
        raise ValueError("Input dataset is empty")
    if not any(stage_rows.targets for stage_rows in rows_by_stage.values()):
        raise ValueError(
            "Dataset does not contain any tunable rows. Provide sc_confidence, cc_confidence, correct, or ground_truth."
        )

    return rows_by_stage, input_dim


def _split_train_eval(
    stage_rows: StageRows,
    train_pct: float,
    seed: int,
) -> tuple[StageRows, StageRows | None]:
    if train_pct >= 1.0:
        return stage_rows, None

    n = len(stage_rows.targets)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = max(1, int(n * train_pct))

    train_idx = indices[:split]
    eval_idx = indices[split:]
    if not eval_idx:
        return stage_rows, None

    train_rows = StageRows(
        embeddings=[stage_rows.embeddings[i] for i in train_idx],
        targets=[stage_rows.targets[i] for i in train_idx],
    )
    eval_rows = StageRows(
        embeddings=[stage_rows.embeddings[i] for i in eval_idx],
        targets=[stage_rows.targets[i] for i in eval_idx],
    )
    return train_rows, eval_rows


def train_stage(
    head: LinearProbeHead,
    stage_name: str,
    stage_rows: StageRows,
    stage_config: StageConfig,
    device: str,
    eval_rows: StageRows | None = None,
) -> None:
    if not stage_rows.targets:
        return

    features = torch.tensor(stage_rows.embeddings, dtype=torch.float32)
    targets = torch.tensor(stage_rows.targets, dtype=torch.float32)
    tensor_dataset = TensorDataset(features, targets)
    generator = torch.Generator()
    generator.manual_seed(stage_config.seed)
    loader = DataLoader(
        tensor_dataset,
        batch_size=min(stage_config.batch_size, len(tensor_dataset)),
        shuffle=True,
        generator=generator,
    )

    if eval_rows is not None and eval_rows.targets:
        eval_features = torch.tensor(eval_rows.embeddings, dtype=torch.float32).to(device)
        eval_targets = torch.tensor(eval_rows.targets, dtype=torch.float32).to(device)
    else:
        eval_features = None
        eval_targets = None

    optimizer = torch.optim.AdamW(
        head.linear.parameters(),
        lr=stage_config.lr,
        weight_decay=stage_config.weight_decay,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    head.dropout.p = stage_config.dropout
    head.train()

    for epoch in range(stage_config.epochs):
        total_loss = 0.0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_features)
            loss = loss_fn(logits, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_targets.size(0)

        average_loss = total_loss / len(tensor_dataset)
        eval_msg = ""
        if eval_features is not None:
            head.eval()
            with torch.no_grad():
                eval_logits = head(eval_features)
                eval_loss = loss_fn(eval_logits, eval_targets).item()
                eval_probs = torch.sigmoid(eval_logits).cpu().numpy()
                eval_labels = eval_targets.cpu().numpy()
            head.train()
            head.dropout.p = stage_config.dropout
            eval_msg = f" eval_loss={eval_loss:.6f}"
            if len(set(eval_labels)) >= 2:
                auroc = roc_auc_score(eval_labels, eval_probs)
                eval_msg += f" eval_auroc={auroc:.4f}"
        print(
            f"[{stage_name}] epoch {epoch + 1}/{stage_config.epochs} loss={average_loss:.6f}{eval_msg}"
        )


def resolve_dataset_path(args: argparse.Namespace, config: dict[str, Any]) -> str:
    if args.dataset is not None:
        return args.dataset
    dataset_path = config.get("dataset")
    if dataset_path is None:
        raise ValueError(
            "No dataset specified. Provide --dataset or set 'dataset' in the config YAML."
        )
    return str(dataset_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    config = load_tune_config(args.config)
    device = resolve_device(args.device)
    dataset_path = resolve_dataset_path(args, config)

    seed = resolve_global_seed(config)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config_path = Path(args.config)
    print(f"Loading dataset: {dataset_path}")
    dataset = load_input_dataset(dataset_path)
    stage_rows, input_dim = build_stage_rows(dataset)
    print(f"  {len(dataset)} rows loaded")
    print(f"Using config: {config_path}")
    print(f"Training device: {device}")
    print(f"Embedding width: {input_dim}")

    head = LinearProbeHead(input_dim=input_dim).to(device)

    for stage_name in SPECIAL_STAGE_KEYS:
        current_rows = stage_rows[stage_name]
        if not current_rows.targets:
            print(f"Skipping {stage_name}: no rows available")
            continue

        stage_config = resolve_stage_config(config, stage_name)
        train_rows, eval_rows = _split_train_eval(
            current_rows, stage_config.train_pct, stage_config.seed
        )

        eval_info = ""
        if eval_rows is not None:
            eval_info = f" eval={len(eval_rows.targets)}"
        print(
            f"Training {stage_name}: train={len(train_rows.targets)}{eval_info} "
            f"lr={stage_config.lr} dropout={stage_config.dropout} "
            f"epochs={stage_config.epochs}"
        )

        use_eval = eval_rows if stage_name == "annotated" else None
        train_stage(head, stage_name, train_rows, stage_config, device, eval_rows=use_eval)

    head.dropout.p = 0.0
    head.eval()
    save_linear_head_checkpoint(args.checkpoint, head, args.model)
    print(f"Checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()

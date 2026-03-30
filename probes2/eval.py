"""CLI tool for evaluating probe confidence quality with AUROC and ECE.

Usage:
    uv run eval --input <dataset>
    uv run eval --input <dataset> --checkpoint <tuned_head.pt> --device cuda
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score

from probes2.finetuning.linear_head import build_linear_head_from_checkpoint
from probes2.finetuning.tune import derive_annotated_target
from probes2.inference.predict import load_input_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate probe confidence with AUROC and ECE"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to dataset with confidence columns and ground truth (.jsonl, .parquet, .arrow)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Optional tuned linear head checkpoint (.pt) to recompute confidence from embeddings",
    )
    parser.add_argument(
        "--device", choices=["mps", "cuda", "cpu"], default=None,
        help="Compute device for checkpoint inference (auto-detected if omitted)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Number of bins for ECE (default: 10)",
    )
    return parser.parse_args(argv)


def resolve_device(device_arg: str | None) -> str:
    if device_arg is not None:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def ece(confidences: list[float], labels: list[float], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    total = len(confidences)
    if total == 0:
        return 0.0

    calibration_error = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins

        bin_confs: list[float] = []
        bin_labels: list[float] = []
        for conf, label in zip(confidences, labels):
            in_bin = (lo <= conf <= hi) if i == n_bins - 1 else (lo <= conf < hi)
            if in_bin:
                bin_confs.append(conf)
                bin_labels.append(label)

        if not bin_confs:
            continue

        avg_conf = sum(bin_confs) / len(bin_confs)
        avg_acc = sum(bin_labels) / len(bin_labels)
        calibration_error += abs(avg_acc - avg_conf) * len(bin_confs) / total

    return calibration_error


def recompute_confidence_from_checkpoint(
    dataset: Dataset,
    checkpoint_path: str,
    device: str,
) -> list[float]:
    if "embeddings" not in dataset.column_names:
        raise ValueError(
            "Dataset must have an 'embeddings' column to use --checkpoint. "
            "Re-run reflect with --with-embeddings."
        )

    head, payload = build_linear_head_from_checkpoint(checkpoint_path, device=device)
    print(
        f"  Loaded checkpoint: {checkpoint_path} "
        f"(base_model={payload['base_model']})"
    )

    embeddings = [[float(v) for v in row["embeddings"]] for row in dataset]
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)

    with torch.no_grad():
        confidences = torch.sigmoid(head(emb_tensor)).cpu().tolist()
    return confidences


def _collect_scores_and_labels(
    dataset: Dataset,
    column: str,
    valid_indices: list[int],
    labels: list[float],
) -> tuple[list[float], list[float]] | None:
    """Return (scores, labels) pairs for non-null entries in *column*, or None."""
    if column not in dataset.column_names:
        return None

    col_scores: list[float] = []
    col_labels: list[float] = []
    for idx, label in zip(valid_indices, labels):
        value = dataset[idx][column]
        if not _is_missing(value):
            col_scores.append(float(value))
            col_labels.append(label)

    return (col_scores, col_labels) if col_scores else None


def _print_metrics(
    name: str,
    scores: list[float],
    labels: list[float],
    n_bins: int,
) -> None:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  {name} ({len(scores)} rows scored)")

    if n_pos == 0 or n_neg == 0:
        print(f"    AUROC: N/A (need both positive and negative labels in scored rows)")
    else:
        print(f"    AUROC: {roc_auc_score(labels, scores):.4f}")

    print(f"    ECE:   {ece(scores, labels, n_bins):.4f}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)

    print(f"Loading dataset: {args.input}")
    dataset = load_input_dataset(args.input)
    print(f"  {len(dataset)} rows loaded")

    # Derive ground truth labels
    labels: list[float] = []
    valid_indices: list[int] = []
    for i, row in enumerate(dataset):
        target = derive_annotated_target(dict(row))
        if target is not None:
            labels.append(target)
            valid_indices.append(i)

    if not labels:
        raise ValueError(
            "No ground truth found. Dataset needs a 'correct' column "
            "or 'ground_truth'/'prediction' columns."
        )

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    accuracy = n_pos / len(labels) * 100
    print(f"  {len(labels)}/{len(dataset)} rows have ground truth")
    print(f"  Labels: {int(n_pos)} positive, {int(n_neg)} negative")
    print(f"  Label accuracy: {accuracy:.2f}%")

    # -- confidence -----------------------------------------------------------
    if args.checkpoint:
        all_conf = recompute_confidence_from_checkpoint(dataset, args.checkpoint, device)
        conf_scores = [all_conf[i] for i in valid_indices]
    elif "confidence" in dataset.column_names:
        conf_scores = [float(dataset[i]["confidence"]) for i in valid_indices]
    else:
        raise ValueError(
            "Dataset has no 'confidence' column and no --checkpoint was provided"
        )

    # -- sc_confidence --------------------------------------------------------
    sc_pair = _collect_scores_and_labels(dataset, "sc_confidence", valid_indices, labels)

    # -- results --------------------------------------------------------------
    print("\n--- Evaluation Results ---\n")
    _print_metrics("confidence", conf_scores, labels, args.n_bins)

    if sc_pair is not None:
        sc_scores, sc_labels = sc_pair
        _print_metrics("sc_confidence", sc_scores, sc_labels, args.n_bins)
    elif "sc_confidence" in dataset.column_names:
        print("  sc_confidence: all values are null -- skipped")
    else:
        print("  sc_confidence: column not found -- skipped")


if __name__ == "__main__":
    main()

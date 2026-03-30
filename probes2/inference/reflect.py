"""CLI tool for probe inference: adds confidence estimates to a dataset.

Usage:
    uv run reflect --input <dataset> --output <dataset> --model <hf_model> --device cuda
    uv run reflect --input <dataset> --output <dataset> --model <hf_model> --device cuda --n-gpus 4
    uv run reflect --input <dataset> --output <dataset> --model <hf_model> --device mps --with-embeddings
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from probes2.finetuning.linear_head import build_linear_head_from_checkpoint
import probes2.registry  # noqa: F401 — registers probe architectures

from probes2.inference.predict import load_input_dataset, save_output_dataset

CONFIDENCE_TEMPERATURE = 3.0
PLATT_BIAS = 0.3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run probe inference to produce confidence estimates"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input dataset (.jsonl, .parquet, .csv, .arrow)"
    )
    parser.add_argument("--output", required=True, help="Path to output dataset")
    parser.add_argument(
        "--model", required=True, help="HuggingFace model ID or local PEFT checkpoint path"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional tuned linear head checkpoint (.pt) produced by the tune CLI",
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cuda", "cpu"],
        default=None,
        help="Compute device (auto-detected if omitted)",
    )
    parser.add_argument(
        "--n-gpus", type=int, default=1, help="Number of GPUs for data-parallel inference (default: 1)"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length (default: 4096)")
    parser.add_argument(
        "--with-embeddings", action="store_true", help="Also extract and store embeddings"
    )
    # Internal flag: set when re-launched under accelerate for multi-GPU
    parser.add_argument("--_distributed", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    ext = Path(args.output).suffix.lower()
    if args.with_embeddings and ext not in (".parquet", ".arrow"):
        raise ValueError(
            f"--with-embeddings requires output format .parquet or .arrow, got '{ext}'. "
            "Embeddings are high-dimensional vectors that don't fit well in CSV/JSONL."
        )
    if args.n_gpus > 1 and args.device in ("mps", "cpu"):
        raise ValueError(
            f"Multi-GPU (--n-gpus {args.n_gpus}) is not supported with --device {args.device}. "
            "Use --device cuda for multi-GPU inference."
        )
    if args.device == "cpu":
        warnings.warn(
            "Running on CPU — inference will be very slow. "
            "Consider using --device cuda or --device mps if available.",
            stacklevel=2,
        )
    if args.checkpoint is not None and Path(args.checkpoint).suffix.lower() != ".pt":
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
    warnings.warn(
        "No GPU detected, falling back to CPU. Inference will be very slow.",
        stacklevel=2,
    )
    return "cpu"


def load_model_and_tokenizer(
    model_path: str,
    device: str,
    device_index: int = 0,
) -> tuple[Any, Any]:
    from peft import AutoPeftModelForSequenceClassification, PeftConfig
    from transformers import AutoTokenizer

    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path

    # Load tokenizer — try from adapter first, fall back to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Resolve device_map
    if device == "cuda":
        device_map = {"": device_index}
    elif device == "mps":
        device_map = {"": "mps"}
    else:
        device_map = {"": "cpu"}

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": device_map,
        "attn_implementation": "eager",
    }

    # MPS may not support bfloat16 — fall back to float16
    try:
        model = AutoPeftModelForSequenceClassification.from_pretrained(
            model_path, **load_kwargs
        )
    except Exception:
        if device == "mps":
            warnings.warn(
                "bfloat16 not supported on MPS, falling back to float16.", stacklevel=2
            )
            load_kwargs["torch_dtype"] = torch.float16
            model = AutoPeftModelForSequenceClassification.from_pretrained(
                model_path, **load_kwargs
            )
        else:
            raise

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return model, tokenizer


def tokenize_messages(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int,
) -> Dataset:
    def _tokenize_row(row: dict) -> dict:
        raw = row["messages"]
        parsed = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
        messages = parsed["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer(
            text, max_length=max_length, truncation=True, padding=False
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    return dataset.map(_tokenize_row, desc="Tokenizing messages")


def _collate_fn(
    batch: list[dict[str, Any]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    input_ids_list = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    attention_mask_list = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]

    max_len = max(ids.size(0) for ids in input_ids_list)

    padded_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
        length = ids.size(0)
        padded_ids[i, :length] = ids
        padded_mask[i, :length] = mask

    return {"input_ids": padded_ids, "attention_mask": padded_mask}


def compute_confidence_margin(logits: torch.Tensor) -> torch.Tensor:
    """Return the gap between the top two logits as a class-agnostic confidence margin."""
    logits = logits.float()
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError(
            f"Expected logits with shape [batch, num_labels>=2], got {tuple(logits.shape)}"
        )
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def calibrate_confidences(
    logits: torch.Tensor,
    temperature: float = CONFIDENCE_TEMPERATURE,
    bias: float = PLATT_BIAS,
) -> torch.Tensor:
    """Apply a light Platt-style transform so confidence is less extreme."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    margin = compute_confidence_margin(logits)
    return torch.sigmoid((margin / temperature) + bias)


def extract_last_token_embeddings(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError(
            "Expected hidden states with shape [batch, seq, hidden], "
            f"got {tuple(hidden_states.shape)}"
        )
    if attention_mask.ndim != 2:
        raise ValueError(
            f"Expected attention mask with shape [batch, seq], got {tuple(attention_mask.shape)}"
        )
    last_positions = attention_mask.sum(dim=1) - 1
    if torch.any(last_positions < 0):
        raise ValueError("Cannot extract embeddings from an empty attention mask")
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, last_positions]


def run_inference(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    batch_size: int,
    with_embeddings: bool,
    linear_head: Any = None,
) -> tuple[list[float], list[list[float]] | None]:
    collate = partial(_collate_fn, pad_token_id=tokenizer.pad_token_id)
    dataset.set_format("python")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    device = next(model.parameters()).device
    all_confidences: list[float] = []
    all_embeddings: list[list[float]] | None = [] if with_embeddings else None

    total_batches = (len(dataset) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=with_embeddings or linear_head is not None,
            )

            batch_embeddings = None
            if linear_head is not None:
                if outputs.hidden_states is None:
                    raise ValueError(
                        "Model did not return hidden states required for --checkpoint inference"
                    )
                batch_embeddings = extract_last_token_embeddings(
                    outputs.hidden_states[-1],
                    attention_mask,
                )
                confidences = torch.sigmoid(linear_head(batch_embeddings)).cpu().tolist()
            else:
                # Confidence should reflect the model's winning label, not a fixed class index.
                # We use the logit gap between the top two classes and a light Platt-style
                # calibration to keep outputs from collapsing to ~0 or ~1.
                confidences = calibrate_confidences(outputs.logits).cpu().tolist()
            all_confidences.extend(confidences)

            if with_embeddings:
                if outputs.hidden_states is None:
                    raise ValueError(
                        "Model did not return hidden states required for --with-embeddings"
                    )
                if batch_embeddings is None:
                    batch_embeddings = extract_last_token_embeddings(
                        outputs.hidden_states[-1],
                        attention_mask,
                    )
                for embedding in batch_embeddings.float().cpu().tolist():
                    all_embeddings.append(embedding)

            print(f"  Batch {batch_idx + 1}/{total_batches} done", flush=True)

    return all_confidences, all_embeddings


def _launch_distributed(args: argparse.Namespace) -> None:
    """Re-launch the script under accelerate for multi-GPU inference."""
    import subprocess

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", str(args.n_gpus),
        "-m", "probes2.inference.reflect",
        "--input", args.input,
        "--output", args.output,
        "--model", args.model,
        "--device", "cuda",
        "--n-gpus", str(args.n_gpus),
        "--batch-size", str(args.batch_size),
        "--max-length", str(args.max_length),
        "--_distributed",
    ]
    if args.with_embeddings:
        cmd.append("--with-embeddings")
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def _run_distributed(args: argparse.Namespace) -> None:
    """Entry point for each accelerate worker process."""
    from accelerate import PartialState
    from accelerate.utils import gather_object

    distributed_state = PartialState()
    rank = distributed_state.process_index
    world_size = distributed_state.num_processes

    print(f"[rank {rank}/{world_size}] Loading model on GPU {rank}")
    model, tokenizer = load_model_and_tokenizer(args.model, "cuda", device_index=rank)
    linear_head = None
    if args.checkpoint:
        linear_head, payload = build_linear_head_from_checkpoint(args.checkpoint, device=f"cuda:{rank}")
        print(
            f"[rank {rank}] Loaded tuned head from {args.checkpoint} "
            f"(base_model={payload['base_model']})"
        )

    dataset = load_input_dataset(args.input)
    if "messages" not in dataset.column_names:
        raise ValueError(
            f"Input dataset must have a 'messages' column. Found: {dataset.column_names}"
        )

    # Shard the dataset indices across processes
    all_indices = list(range(len(dataset)))
    with distributed_state.split_between_processes(all_indices, apply_padding=True) as shard_indices:
        # Track how many are real vs padded
        n_real = min(len(shard_indices), len(dataset))
        real_indices = [idx for idx in shard_indices if idx < len(dataset)]
        shard = dataset.select(real_indices)

        print(f"[rank {rank}] Processing {len(shard)} samples")
        tokenized = tokenize_messages(shard, tokenizer, args.max_length)
        confidences, embeddings = run_inference(
            model, tokenizer, tokenized, args.batch_size, args.with_embeddings, linear_head
        )

    # Gather results from all processes
    gathered_conf = gather_object([{"indices": real_indices, "confidences": confidences}])

    gathered_emb = None
    if args.with_embeddings:
        gathered_emb = gather_object([{"indices": real_indices, "embeddings": embeddings}])

    # Only rank 0 saves the final dataset
    if distributed_state.is_main_process:
        # Reconstruct in original order
        all_confidences = [0.0] * len(dataset)
        all_embeddings = [None] * len(dataset) if args.with_embeddings else None

        for item in gathered_conf:
            for idx, conf in zip(item["indices"], item["confidences"]):
                all_confidences[idx] = conf

        if args.with_embeddings and gathered_emb is not None:
            for item in gathered_emb:
                for idx, emb in zip(item["indices"], item["embeddings"]):
                    all_embeddings[idx] = emb

        dataset = dataset.add_column("confidence", all_confidences)
        if args.with_embeddings:
            dataset = dataset.add_column("embeddings", all_embeddings)

        save_output_dataset(dataset, args.output)
        print(f"Output saved to {args.output} ({len(dataset)} rows)")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    args.device = device
    validate_args(args)

    # Multi-GPU: re-launch under accelerate or run as distributed worker
    if args.n_gpus > 1 and not args._distributed:
        _launch_distributed(args)
        return

    if args._distributed:
        _run_distributed(args)
        return

    # Single-device path
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    print(f"  Model loaded on {device}")
    linear_head = None
    if args.checkpoint:
        linear_head, payload = build_linear_head_from_checkpoint(args.checkpoint, device=device)
        print(
            f"  Loaded tuned head from {args.checkpoint} "
            f"(base_model={payload['base_model']})"
        )

    print(f"Loading dataset: {args.input}")
    dataset = load_input_dataset(args.input)
    if "messages" not in dataset.column_names:
        raise ValueError(
            f"Input dataset must have a 'messages' column. Found: {dataset.column_names}"
        )
    print(f"  {len(dataset)} rows loaded")

    print("Tokenizing messages...")
    tokenized = tokenize_messages(dataset, tokenizer, args.max_length)

    print("Running inference...")
    confidences, embeddings = run_inference(
        model, tokenizer, tokenized, args.batch_size, args.with_embeddings, linear_head
    )

    dataset = dataset.add_column("confidence", confidences)
    if args.with_embeddings and embeddings is not None:
        dataset = dataset.add_column("embeddings", embeddings)

    save_output_dataset(dataset, args.output)
    print(f"Output saved to {args.output} ({len(dataset)} rows)")


if __name__ == "__main__":
    main()

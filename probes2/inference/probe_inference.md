# Probe Inference Guide

This guide walks through running the probe model to get confidence estimates for your LLM judge's predictions. You should have already completed the [LLM Inference Guide](./llm_inference.md) and have a dataset with a `messages` column.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- A GPU (NVIDIA CUDA or Apple Silicon MPS) — CPU is supported but very slow

## How It Works

The probe is a small classifier model that reads the full LLM conversation (stored in the `messages` column) and outputs a confidence score between 0 and 1. Higher scores mean the probe is more confident that the LLM's prediction is correct.

## Running the Probe

The `reflect` command loads the probe model, tokenizes the conversation from each row's `messages` column, and appends a `confidence` column to the output.

**On Mac (Apple Silicon):**
```bash
uv run reflect --input results.jsonl --output results_with_confidence.jsonl --model modaic/Qwen3.5-4B-probe --device mps
```

**On a single NVIDIA GPU:**
```bash
uv run reflect --input results.jsonl --output results_with_confidence.jsonl --model modaic/Qwen3.5-4B-probe --device cuda
```

**On multiple NVIDIA GPUs (data parallelism):**
```bash
uv run reflect --input results.jsonl --output results_with_confidence.jsonl --model modaic/Qwen3.5-4B-probe --device cuda --n-gpus 4
```

If you omit `--device`, it will be auto-detected (CUDA > MPS > CPU).

### CLI Reference

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--input` | Yes | — | Path to input dataset (`.jsonl`, `.parquet`, `.csv`, `.arrow`). Must contain a `messages` column from the `predict` step. |
| `--output` | Yes | — | Path to output dataset (same format options) |
| `--model` | Yes | — | HuggingFace model ID (e.g. `modaic/Qwen3.5-4B-probe`) or path to a local PEFT checkpoint |
| `--checkpoint` | No | — | Path to a tuned linear head `.pt` checkpoint from `tune`. If omitted, the probe's built-in head is used. |
| `--device` | No | auto | Compute device: `cuda`, `mps`, or `cpu`. Auto-detected if omitted. |
| `--n-gpus` | No | 1 | Number of GPUs for data-parallel inference. Only works with `--device cuda`. |
| `--batch-size` | No | 8 | Number of rows per batch. Increase if you have plenty of GPU memory, decrease if you run into OOM errors. |
| `--max-length` | No | 4096 | Maximum token sequence length. Conversations longer than this are truncated. |
| `--with-embeddings` | No | — | Extract and store the internal embeddings alongside confidence scores. Requires output format `.parquet` or `.arrow`. |

## View Results

The output dataset is a copy of the input with an added `confidence` column (a float between 0 and 1).

| subject | body | is_spam | messages | confidence |
| --- | --- | --- | --- | --- |
| Meeting tomorrow | Hi, can we sync at 10am to review the roadmap? | not spam | `{"messages": [...], ...}` | 0.83 |
| Limited time offer | Click here now for 90% off luxury watches!!! | spam | `{"messages": [...], ...}` | 0.92 |
| Re: invoice #4421 | Attached is the corrected PDF. Let me know if anything looks off. | not spam | `{"messages": [...], ...}` | 0.63 |

## Extracting Embeddings

You can also extract the internal embeddings used to produce the confidence scores. This is useful if you later want to [fine-tune](../finetuning/fine-tuning.md) a probe on your own data.

Add the `--with-embeddings` flag. The output format **must** be `.parquet` or `.arrow` since embeddings are high-dimensional vectors that don't fit well in CSV or JSONL.

```bash
uv run reflect --input results.parquet --output results_with_embeddings.parquet --model modaic/Qwen3.5-4B-probe --device cuda --with-embeddings
```

This adds an `embeddings` column containing the embedding vector for each row.

## Using A Tuned Head

If you fine-tuned a linear head with the `tune` CLI, you can load it during probe inference:

```bash
uv run reflect --input results.jsonl --output results_with_confidence.jsonl --model modaic/Qwen3.5-4B-probe --checkpoint tuned_head.pt --device cuda
```

When `--checkpoint` is provided, `reflect` still uses the base probe model to produce embeddings, but it computes the final confidence score with the tuned linear head from the checkpoint.

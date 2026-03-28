# LLM Inference Guide

This guide walks through running an LLM judge on your dataset to produce classifications. The output of this step is a labeled dataset ready for [Probe Inference](./probe_inference.md) (confidence estimation).

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Access to `Qwen/Qwen3.5-4B` through a local batch client or a legacy LLM server setup

## Step 1: Define Your Dataset

Create a dataset where each column is a separate input field for the judge. Supported formats: `.jsonl`, `.parquet`, or `.arrow`.

**Example** (`data.jsonl`):
```json
{"subject": "Meeting tomorrow", "body": "Hi, can we sync at 10am to review the roadmap?"}
{"subject": "Limited time offer", "body": "Click here now for 90% off luxury watches!!!"}
{"subject": "Re: invoice #4421", "body": "Attached is the corrected PDF. Let me know if anything looks off."}
```

## Step 2: Create Your Arbiter Config

An **arbiter** is a YAML file that defines how the LLM should judge each row in your dataset. It specifies the model, instructions, inputs, and outputs.

```yaml
lm:
  model: huggingface/Qwen/Qwen3.5-4B

instructions: Classify the email as spam or not spam

inputs:                          # maps to columns in your dataset
  - name: subject                # column name
    type: string                 # python type: string, float, int, or bool
  - name: body
    type: string

outputs:                         # new columns appended to your dataset
  - name: is_spam                # output column name
    type: string
    options:                     # constrain the LLM to these values
      - spam
      - not spam
```

### Model Identifier Format

For the recommended batch-client workflow, set:

| Use Case | Model |
| --- | --- |
| Local batch client (recommended) | `huggingface/Qwen/Qwen3.5-4B` |
| Ollama server | `ollama/qwen3.5:4b` |
| Legacy hosted vLLM server | `hosted_vllm/Qwen/Qwen3.5-4B` |

Use `api_base` only for server-backed paths such as Ollama or a legacy hosted vLLM endpoint. The batch-client path does not need `api_base`.

## Step 3: Choose An Execution Path

### Option A: Modal Batch Client (Recommended)

This is the recommended path for large runs. It submits the dataset directly through `modaic.Predict.abatch` using Modaic's `ModalBatchClient`.

```bash
uv run predict --arbiter arbiter.yaml --input data.jsonl --output results.jsonl --batch-client modal
```

### Option B: Local vLLM Batch Client

This also uses `modaic.Predict.abatch`, but runs vLLM locally through Modaic's `VLLMBatchClient`.

```bash
uv run predict --arbiter arbiter.yaml --input data.jsonl --output results.jsonl --batch-client vllm
```

### Option C: Ollama Server

Ollama remains the easiest server-backed option on macOS, Linux, or Windows. It is slower than the batch-client paths for large datasets.

```bash
ollama pull qwen3.5:4b
ollama serve
```

Set the arbiter config with an `api_base`:
```yaml
lm:
  model: ollama/qwen3.5:4b
  api_base: http://localhost:11434
```

### Option D: Legacy Hosted vLLM Server

This remains supported for compatibility, but it is no longer the recommended path. If you still want to run a separate vLLM server, keep using a `hosted_vllm/...` model plus `api_base`.

```yaml
lm:
  model: hosted_vllm/Qwen/Qwen3.5-4B
  api_base: http://localhost:8000/v1
```

## Step 4: Run Inference

```bash
uv run predict --arbiter arbiter.yaml --input data.jsonl --output results.jsonl --batch-client modal
```

### CLI Reference

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `-a`, `--arbiter` | Yes | — | Path to your arbiter YAML config |
| `-i`, `--input` | Yes | — | Path to input dataset (`.jsonl`, `.parquet`, `.arrow`) |
| `-o`, `--output` | Yes | — | Path to output dataset (same format options) |
| `-t`, `--threads` | No | 64 | Number of concurrent requests for the regular DSPy/server-backed path. Ignored for the main arbiter pass and self-consistency when `--batch-client` is set. |
| `--batch-client` | No | — | Batch execution backend: `modal` or `vllm`. When set, the main arbiter pass and self-consistency use `modaic.Predict.abatch`. |
| `-sc`, `--self-consistency` | No | — | Fraction of rows (0.0–1.0) to run self-consistency scoring on. Adds an `sc_confidence` column. |
| `-cc`, `--cross-consistency` | No | — | Fraction of remaining rows (after SC) to run cross-consistency scoring on. Requires `--council`. Adds a `cc_confidence` column. |
| `--council` | No | — | Path to a council YAML file listing frontier models for cross-consistency. Required when using `-cc`. |
| `-v` | No | — | Increase logging verbosity. Stack for more detail: `-v` = WARNING, `-vv` = INFO, `-vvv` = DEBUG |

### Example

```bash
# Recommended: Modal batch execution
uv run predict --arbiter arbiter.yaml --input data.jsonl --output results.jsonl --batch-client modal

# Local vLLM batch execution
uv run predict --arbiter arbiter.yaml --input data.jsonl --output results.jsonl --batch-client vllm

# Using Ollama (fewer threads since it's sequential)
uv run predict --arbiter arbiter.yaml --input data.jsonl --output results.jsonl --threads 4
```

## Advanced Options

### Self-Consistency (`-s`)

Self-consistency measures how confident the arbiter is in its own answer by asking it the same question multiple times and checking how often it gives the same response.

**How it works:**

1. After the initial inference pass, a random subset of rows is selected (controlled by the `-s` fraction).
2. Each selected row is re-run through the same arbiter 10 additional times.
3. For each row, `sc_confidence` = (number of re-runs that match the original answer) / 10.
4. Rows not selected for self-consistency have `sc_confidence` set to `null`.

A score of `1.0` means the model gave the same answer every time — high confidence. A score of `0.1` means only 1 out of 10 re-runs agreed — the model is uncertain.

```bash
# Score 20% of rows with self-consistency
uv run predict --arbiter arbiter.yaml --input data.jsonl --output results.jsonl -sc 0.2
```

### Cross-Consistency (`-cc` + `--council`)

Cross-consistency measures confidence by asking a council of different frontier models the same question and checking how many agree with the original arbiter's answer.

**How it works:**

1. After self-consistency (if enabled), the remaining unscored rows are the candidate pool.
2. A random subset of those remaining rows is selected (controlled by the `-cc` fraction).
3. Each selected row is sent to every model listed in the council YAML file.
4. For each row, `cc_confidence` = (number of council models that match the original answer) / (number of council models).
5. Rows not selected for cross-consistency have `cc_confidence` set to `null`.

No row will receive both an `sc_confidence` and a `cc_confidence` score — the two methods cover different slices of the dataset.

#### Council Config

The council YAML file is a list of LM specs. Each entry requires a `model` field and optionally accepts any additional parameters supported by the model backend (e.g., `api_base`, `api_key`, `temperature`).

```yaml
# council.yaml
- model: openai/gpt-4o
- model: anthropic/claude-sonnet-4-20250514
- model: openai/gpt-4o-mini
  temperature: 0.3
```

#### Example Usage

```bash
# Cross-consistency only: score 30% of rows using a council
uv run predict --arbiter arbiter.yaml --input data.jsonl --output results.jsonl \
  -cc 0.3 --council council.yaml

# Combined: self-consistency on 20% of rows, then cross-consistency on 30% of the rest
uv run predict --arbiter arbiter.yaml --input data.jsonl --output results.jsonl \
  -s 0.2 -cc 0.3 --council council.yaml
```

### Consistency Scoring Summary

| Flag | Method | Column added | What it measures |
| --- | --- | --- | --- |
| `-sc 0.2` | Self-consistency | `sc_confidence` | Agreement of the same model across 10 repeated runs |
| `-cc 0.3 --council council.yaml` | Cross-consistency | `cc_confidence` | Agreement of different frontier models with the original answer |

## Step 5: View Results

The output dataset contains all original columns plus:
- One new column per entry in the `outputs` section of your arbiter config
- A `prediction` column containing a JSON object of all output fields (e.g., `{"is_spam": "not spam"}`)
- A `messages` column containing the full LLM conversation (system prompt, user message, and model response)
- `sc_confidence` — self-consistency score (if `-sc` was used)
- `cc_confidence` — cross-consistency score (if `-cc` was used)

**Example output:**

| subject | body | is_spam | prediction | messages | sc_confidence | cc_confidence |
| --- | --- | --- | --- | --- | --- | --- |
| Meeting tomorrow | Hi, can we sync at 10am... | not spam | `{"is_spam": "not spam"}` | `{"messages": [...], ...}` | 0.9 | |
| Limited time offer | Click here now for 90%... | spam | `{"is_spam": "spam"}` | `{"messages": [...], ...}` | | 1.0 |
| Re: invoice #4421 | Attached is the corrected... | not spam | `{"is_spam": "not spam"}` | `{"messages": [...], ...}` | | |

This dataset is now ready for confidence estimation. Continue to the [Probe Inference Guide](./probe_inference.md).

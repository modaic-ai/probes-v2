# LLM Inference Guide

This guide walks through running an LLM judge on your dataset to produce classifications. The output of this step is a labeled dataset ready for [Probe Inference](./probe_inference.md) (confidence estimation).

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- An LLM server running locally or remotely (see [Serve The LLM](#serve-the-llm) below)

## Step 1: Define Your Dataset

Create a dataset where each column is a separate input field for the judge. Supported formats: `.jsonl`, `.parquet`, `.csv`, or `.arrow`.

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
  model: hosted_vllm/Qwen/Qwen3.5-4B      # model identifier (see below)
  api_base: http://localhost:8000/v1         # URL of your LLM server

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

The `model` field uses a prefix that tells the system which backend you are using:

| Backend | Prefix | Example |
| --- | --- | --- |
| vLLM | `hosted_vllm/` | `hosted_vllm/Qwen/Qwen3.5-4B` |
| Ollama | `ollama/` | `ollama/qwen3.5:4b` |

The part after the prefix is the model name as known by that backend.

## Step 3: Serve The LLM

You need an LLM server running before you can call `predict`. Below are instructions for the two supported backends.

### Option A: vLLM (Recommended)

[vLLM](https://docs.vllm.ai/) is a high-throughput LLM serving engine. It requires an NVIDIA GPU.

**Install vLLM:**
```bash
pip install vllm
```

**Start the server:**
```bash
vllm serve Qwen/Qwen3.5-4B --gpu-memory-utilization 0.9
```

By default vLLM serves on `http://localhost:8000`. Set the arbiter config to point at it:
```yaml
lm:
  model: hosted_vllm/Qwen/Qwen3.5-4B
  api_base: http://localhost:8000/v1
```

**Useful vLLM flags:**

| Flag | Description |
| --- | --- |
| `--tensor-parallel-size N` | Spread the model across N GPUs (tensor parallelism). Use this when the model doesn't fit on a single GPU. |
| `--gpu-memory-utilization 0.9` | Fraction of GPU memory vLLM is allowed to use. |
| `--enforce-eager` | Disables CUDA graph capture for faster startup. Good for testing, but reduces throughput — turn it off for large datasets. |

**Example — 4 GPUs with tensor parallelism:**
```bash
vllm serve Qwen/Qwen3.5-4B --tensor-parallel-size 4 --gpu-memory-utilization 0.9
```

For the full list of options see the [vLLM docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/).

### Option B: Ollama (Easy Setup, But Slow)

> **Warning:** Ollama processes requests sequentially and is **significantly slower** than vLLM for large datasets. We strongly recommend using vLLM with data parallelism (`--tensor-parallel-size`) for any dataset larger than a few hundred rows.

[Ollama](https://ollama.com/) is the easiest way to get started locally on macOS, Linux, or Windows. It does not require an NVIDIA GPU.

**Install Ollama:**
Download from [ollama.com](https://ollama.com/) and follow the install instructions.

**Pull the model and start the server:**
```bash
ollama pull qwen3.5:4b
ollama serve     # starts the server on http://localhost:11434
```

Set the arbiter config:
```yaml
lm:
  model: ollama/qwen3.5:4b
  api_base: http://localhost:11434
```

## Step 4: Run Inference

```bash
uv run predict --arbiter arbiter.yaml --input data.jsonl --output results.jsonl
```

### CLI Reference

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--arbiter` | Yes | — | Path to your arbiter YAML config |
| `--input` | Yes | — | Path to input dataset (`.jsonl`, `.parquet`, `.csv`, `.arrow`) |
| `--output` | Yes | — | Path to output dataset (same format options) |
| `--threads` | No | 64 | Number of concurrent requests to the LLM server. Increase for higher throughput if your server can handle it. |
| `-v` | No | — | Increase logging verbosity. Stack for more detail: `-v` = WARNING, `-vv` = INFO, `-vvv` = DEBUG |

### Example

```bash
# Using vLLM with 128 threads for high throughput
uv run predict --arbiter arbiter.yaml --input data.jsonl --output results.jsonl --threads 128

# Using Ollama (fewer threads since it's sequential)
uv run predict --arbiter arbiter.yaml --input data.jsonl --output results.jsonl --threads 4
```

## Step 5: View Results

The output dataset contains all original columns plus:
- One new column per entry in the `outputs` section of your arbiter config
- A `messages` column containing the full LLM conversation (system prompt, user message, and model response)

**Example output:**

| subject | body | is_spam | messages |
| --- | --- | --- | --- |
| Meeting tomorrow | Hi, can we sync at 10am to review the roadmap? | not spam | `{"messages": [...], "outputs": {"text": ..., "reasoning": ...}}` |
| Limited time offer | Click here now for 90% off luxury watches!!! | spam | `{"messages": [...], "outputs": {"text": ..., "reasoning": ...}}` |
| Re: invoice #4421 | Attached is the corrected PDF. Let me know if anything looks off. | not spam | `{"messages": [...], "outputs": {"text": ..., "reasoning": ...}}` |

This dataset is now ready for confidence estimation. Continue to the [Probe Inference Guide](./probe_inference.md).

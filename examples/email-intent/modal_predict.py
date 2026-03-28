from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "email-intent-predict"
DATA_VOLUME_NAME = "probes-email-intent-data"
HF_CACHE_VOLUME_NAME = "huggingface-cache"
VLLM_CACHE_VOLUME_NAME = "vllm-cache"

REMOTE_EXAMPLE_DIR = "/root/examples/email-intent"
REMOTE_DATA_DIR = "/data/email-intent"
REMOTE_INPUT_PATH = f"{REMOTE_DATA_DIR}/multiclass-email-classification.parquet"
REMOTE_DEFAULT_OUTPUT_PATH = (
    f"{REMOTE_DATA_DIR}/multiclass-email-classification.predictions.parquet"
)
REMOTE_ARBITER_PATH = f"{REMOTE_EXAMPLE_DIR}/arbiter.yaml"

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name(
    VLLM_CACHE_VOLUME_NAME, create_if_missing=True
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["cu128"])
    .pip_install("vllm>=0.17.0")
    .add_local_python_source("probes2")
    .add_local_python_source("modaic")
    .add_local_dir("examples/email-intent", remote_path=REMOTE_EXAMPLE_DIR)
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="H100:4",
    timeout=60 * 60,
    volumes={
        "/data": data_volume,
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
)
def run_predict(
    input_path: str = REMOTE_INPUT_PATH,
    output_path: str = REMOTE_DEFAULT_OUTPUT_PATH,
    arbiter_path: str = REMOTE_ARBITER_PATH,
) -> str:
    print(
        f"[modal_predict] starting remote predict input={input_path} output={output_path} arbiter={arbiter_path}",
        flush=True,
    )
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["MODAIC_BATCH_TENSOR_PARALLEL_SIZE"] = "4"
    os.environ["MODAIC_BATCH_GPU_MEMORY_UTILIZATION"] = "0.9"
    # Keep our INFO logs visible without enabling verbose per-request HTTP traces.
    os.environ["LOGLEVEL"] = "INFO"
    cmd = [
        "python",
        "-m",
        "probes2.inference.predict",
        "--arbiter",
        arbiter_path,
        "--input",
        input_path,
        "--output",
        output_path,
        "--batch-client",
        "vllm",
        "-vv",
    ]
    print(f"[modal_predict] command: {' '.join(cmd)}", flush=True)
    start = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    env["LOGLEVEL"] = "INFO"
    env["PYTHONWARNINGS"] = "default"
    env["DSPY_LOG_LEVEL"] = "INFO"
    env["MODAIC_BATCH_TENSOR_PARALLEL_SIZE"] = "4"
    env["MODAIC_BATCH_GPU_MEMORY_UTILIZATION"] = "0.9"
    subprocess.run(
        cmd,
        check=True,
        env=env,
    )
    print(
        f"[modal_predict] predict finished in {time.time() - start:.1f}s, committing data volume",
        flush=True,
    )
    data_volume.commit()
    print(f"[modal_predict] data volume committed, output at {output_path}", flush=True)
    return output_path


@app.local_entrypoint()
def main(
    output_filename: str = "multiclass-email-classification.predictions.parquet",
) -> None:
    local_dataset = Path.cwd() / "data" / "multiclass-email-classification.parquet"
    if not local_dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {local_dataset}")

    remote_output_path = f"{REMOTE_DATA_DIR}/{output_filename}"
    print(
        f"[modal_predict] uploading local dataset {local_dataset} to Modal volume as {REMOTE_INPUT_PATH}",
        flush=True,
    )

    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(
            str(local_dataset), "email-intent/multiclass-email-classification.parquet"
        )
    print(
        "[modal_predict] dataset upload complete, starting remote function", flush=True
    )

    output_path = run_predict.remote(
        input_path=REMOTE_INPUT_PATH,
        output_path=remote_output_path,
        arbiter_path=REMOTE_ARBITER_PATH,
    )
    print(f"Prediction output saved to Modal volume at {output_path}")

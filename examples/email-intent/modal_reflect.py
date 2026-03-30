from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "email-intent-reflect"
DATA_VOLUME_NAME = "probes-email-intent-data"
HF_CACHE_VOLUME_NAME = "huggingface-cache"

REMOTE_EXAMPLE_DIR = "/root/examples/email-intent"
REMOTE_DATA_DIR = "/data/email-intent"
REMOTE_INPUT_PATH = (
    f"{REMOTE_DATA_DIR}/multiclass-email-classification.predictions.parquet"
)
REMOTE_DEFAULT_OUTPUT_PATH = (
    f"{REMOTE_DATA_DIR}/multiclass-email-classification.reflections.parquet"
)
REMOTE_MODEL = "modaic/Qwen3.5-4B-probe"

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["cu128"])
    .add_local_python_source("probes2")
    .add_local_python_source("modaic")
    .add_local_dir("examples/email-intent", remote_path=REMOTE_EXAMPLE_DIR)
)

app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="H100:4",
    timeout=60 * 60,
    volumes={
        "/data": data_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
class Reflector:
    @modal.method()
    def run_reflect(
        self,
        input_path: str = REMOTE_INPUT_PATH,
        output_path: str = REMOTE_DEFAULT_OUTPUT_PATH,
        model: str = REMOTE_MODEL,
    ) -> str:
        print(
            f"[modal_reflect] starting remote reflect input={input_path} "
            f"output={output_path} model={model}",
            flush=True,
        )
        cmd = [
            "python",
            "-m",
            "probes2.inference.reflect",
            "--input",
            input_path,
            "--output",
            output_path,
            "--model",
            model,
            "--device",
            "cuda",
            "--n-gpus",
            "4",
            "--with-embeddings",
        ]
        print(f"[modal_reflect] command: {' '.join(cmd)}", flush=True)
        start = time.time()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        subprocess.run(cmd, check=True, env=env)
        print(
            f"[modal_reflect] reflect finished in {time.time() - start:.1f}s",
            flush=True,
        )
        data_volume.commit()
        print("[modal_reflect] data volume committed", flush=True)
        return output_path

    @modal.exit()
    def commit_volume(self):
        data_volume.commit()
        print("[modal_reflect] data volume committed (exit hook)", flush=True)


@app.local_entrypoint()
def main(
    output_filename: str = "multiclass-email-classification.reflections.parquet",
) -> None:
    local_input = (
        Path.cwd()
        / "data"
        / "multiclass-email-classification.predictions.parquet"
    )
    if not local_input.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {local_input}. Run modal_predict.py first."
        )

    remote_output_path = f"{REMOTE_DATA_DIR}/{output_filename}"

    print(
        f"[modal_reflect] uploading predictions {local_input} to Modal volume "
        f"as {REMOTE_INPUT_PATH}",
        flush=True,
    )
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(
            str(local_input),
            "email-intent/multiclass-email-classification.predictions.parquet",
        )
    print(
        "[modal_reflect] upload complete, starting remote function",
        flush=True,
    )

    output_path = Reflector().run_reflect.remote(
        input_path=REMOTE_INPUT_PATH,
        output_path=remote_output_path,
        model=REMOTE_MODEL,
    )
    print(f"Reflection output saved to Modal volume at {output_path}")

    local_output = Path.cwd() / "data" / output_filename
    print(
        f"[modal_reflect] downloading {output_path} to {local_output}",
        flush=True,
    )
    data = b"".join(data_volume.read_file(f"email-intent/{output_filename}"))
    local_output.write_bytes(data)
    print(f"[modal_reflect] saved local copy to {local_output}", flush=True)

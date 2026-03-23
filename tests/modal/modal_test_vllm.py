import time
import urllib.error
import urllib.request

import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm>=0.17.0", "huggingface-hub", "transformers>=4.50,<5")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

MODEL_NAME = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "main"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

FAST_BOOT = False

app = modal.App("test-vllm-inference")

N_GPU = 4
MINUTES = 60
VLLM_PORT = 8000
STARTUP_TIMEOUT_SECONDS = 45 * MINUTES
FUNCTION_TIMEOUT_SECONDS = 2 * 60 * MINUTES


def wait_for_health(
    base_url: str,
    timeout_seconds: int = STARTUP_TIMEOUT_SECONDS,
    poll_seconds: int = 5,
) -> None:
    """Block until the /health endpoint reports HTTP 200."""
    deadline = time.time() + timeout_seconds
    health_url = f"{base_url.rstrip('/')}/health"

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=10) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(poll_seconds)

    raise TimeoutError(f"Timed out waiting for healthy vLLM server at {health_url}")


def _serve_impl() -> None:
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(N_GPU),
        "--gpu-memory-utilization",
        "0.9",
        "--enforce-eager",
        "--reasoning-parser",
        "qwen3"
    ]

    print(
        f"[serve] launching model_name={MODEL_NAME} model_revision={MODEL_REVISION}",
        flush=True,
    )
    print("[serve] command:", *cmd, flush=True)
    subprocess.Popen(" ".join(cmd), shell=True)


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=FUNCTION_TIMEOUT_SECONDS,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS)
def serve() -> None:
    _serve_impl()

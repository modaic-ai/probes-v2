import subprocess
import time
import urllib.error
import urllib.request

import pytest
import yaml
from datasets import Dataset


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.5:4b"


@pytest.fixture
def ollama_server():
    """Start ollama serve, pull the model, and tear down after the test."""
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for ollama to be ready
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(OLLAMA_BASE_URL, timeout=5):
                break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    else:
        proc.terminate()
        raise TimeoutError("ollama serve did not start in time")

    # Pull the model
    subprocess.run(
        ["ollama", "pull", OLLAMA_MODEL],
        check=True,
        capture_output=True,
        text=True,
    )

    yield OLLAMA_BASE_URL

    proc.terminate()
    proc.wait(timeout=10)


def test_predict_vllm(tmp_path):
    from tests.modal.modal_test_vllm import app, serve, wait_for_health

    with app.run():
        url = serve.get_web_url()
        wait_for_health(url)

        # Create a temp arbiter with the dynamic modal URL
        with open("tests/artifacts/arbiter_vllm.yaml") as f:
            spec = yaml.safe_load(f)
        spec["lm"]["api_base"] = f"{url}/v1"
        arbiter_path = tmp_path / "arbiter.yaml"
        with open(arbiter_path, "w") as f:
            yaml.safe_dump(spec, f)

        output = tmp_path / "output.jsonl"
        result = subprocess.run(
            [
                "predict",
                "--arbiter", str(arbiter_path),
                "--input", "tests/artifacts/data.jsonl",
                "--output", str(output),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"predict failed:\n{result.stderr}"

    ds = Dataset.from_json(str(output))
    assert "is_spam" in ds.column_names
    assert "messages" in ds.column_names
    assert len(ds) == 10


def test_predict_ollama(tmp_path, ollama_server):
    output = tmp_path / "output.jsonl"
    result = subprocess.run(
        [
            "predict",
            "--arbiter", "tests/artifacts/arbiter_ollama.yaml",
            "--input", "tests/artifacts/data.jsonl",
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"predict failed:\n{result.stderr}"

    ds = Dataset.from_json(str(output))
    assert "is_spam" in ds.column_names
    assert "messages" in ds.column_names
    assert len(ds) == 10

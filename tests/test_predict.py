import random
import subprocess
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
import yaml
from datasets import Dataset

from probes2.inference import predict as predict_module


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.5:4b"


def _make_prediction(is_spam: str):
    import dspy

    prediction = dspy.Prediction(is_spam=is_spam)
    prediction._messages = [{"role": "assistant", "content": is_spam}]
    prediction._outputs = {"text": is_spam}
    return prediction


class _FakeLM:
    def __init__(self, model: str = "huggingface/Qwen/Qwen3.5-4B"):
        self.model = model
        self.kwargs = {}


class _FakeArbiter:
    def __init__(self, batches: list[list]):
        self.lm = _FakeLM()
        self._batches = list(batches)
        self.calls: list[dict] = []

    async def abatch(self, inputs, show_progress, return_messages, client):
        self.calls.append(
            {
                "inputs": inputs,
                "show_progress": show_progress,
                "return_messages": return_messages,
                "client": client,
            }
        )
        predictions = self._batches.pop(0)
        return [SimpleNamespace(prediction=prediction) for prediction in predictions]


class FailedPrediction:
    def __init__(self, error: str, index: int = 0):
        self.error = error
        self.index = index


def test_process_dataset_batch_client_uses_abatch(monkeypatch):
    monkeypatch.setattr(
        predict_module,
        "_build_batch_client",
        lambda batch_client_name, lm: {"name": batch_client_name, "lm": lm},
    )
    arbiter = _FakeArbiter(
        batches=[[_make_prediction("spam"), _make_prediction("not spam")]]
    )
    dataset = Dataset.from_dict(
        {
            "subject": ["offer", "meeting"],
            "body": ["buy now", "agenda"],
        }
    )

    result = predict_module.process_dataset(
        arbiter=arbiter,
        dataset=dataset,
        input_fields=["subject", "body"],
        output_fields=["is_spam"],
        batch_client_name="modal",
    )

    assert len(arbiter.calls) == 1
    assert arbiter.calls[0]["client"]["name"] == "modal"
    assert arbiter.calls[0]["return_messages"] is True
    assert result.column_names == ["subject", "body", "is_spam", "prediction", "messages", "prediction_error"]
    assert result["is_spam"] == ["spam", "not spam"]


def test_process_dataset_batch_client_self_consistency_uses_abatch(monkeypatch):
    monkeypatch.setattr(
        predict_module,
        "_build_batch_client",
        lambda batch_client_name, lm: {"name": batch_client_name, "lm": lm},
    )
    monkeypatch.setattr(random, "sample", lambda population, k: list(population)[:k])
    arbiter = _FakeArbiter(
        batches=[
            [_make_prediction("spam"), _make_prediction("not spam")],
            [_make_prediction("spam") for _ in range(10)],
        ]
    )
    dataset = Dataset.from_dict(
        {
            "subject": ["offer", "meeting"],
            "body": ["buy now", "agenda"],
        }
    )

    result = predict_module.process_dataset(
        arbiter=arbiter,
        dataset=dataset,
        input_fields=["subject", "body"],
        output_fields=["is_spam"],
        batch_client_name="modal",
        self_consistency=0.5,
    )

    assert len(arbiter.calls) == 2
    assert len(arbiter.calls[1]["inputs"]) == 10
    assert result["sc_confidence"] == [1.0, None]


def test_process_dataset_batch_client_keeps_cross_consistency_on_parallel(monkeypatch):
    monkeypatch.setattr(
        predict_module,
        "_build_batch_client",
        lambda batch_client_name, lm: {"name": batch_client_name, "lm": lm},
    )
    monkeypatch.setattr(random, "sample", lambda population, k: list(population)[:k])

    calls = []

    def fake_run_parallel_chunks(exec_pairs, resolved_threads, label):
        calls.append(
            {
                "size": len(exec_pairs),
                "resolved_threads": resolved_threads,
                "label": label,
            }
        )
        return [
            _make_prediction("spam"),
            _make_prediction("spam"),
            _make_prediction("not spam"),
            _make_prediction("not spam"),
        ]

    monkeypatch.setattr(predict_module, "_run_parallel_chunks", fake_run_parallel_chunks)

    arbiter = _FakeArbiter(
        batches=[[_make_prediction("spam"), _make_prediction("not spam")]]
    )
    dataset = Dataset.from_dict(
        {
            "subject": ["offer", "meeting"],
            "body": ["buy now", "agenda"],
        }
    )

    result = predict_module.process_dataset(
        arbiter=arbiter,
        dataset=dataset,
        input_fields=["subject", "body"],
        output_fields=["is_spam"],
        batch_client_name="modal",
        cross_consistency=1.0,
        council_arbiters=[object(), object()],
    )

    assert len(arbiter.calls) == 1
    assert calls == [{"size": 4, "resolved_threads": 2, "label": "Cross-consistency"}]
    assert result["cc_confidence"] == [1.0, 1.0]


def test_process_dataset_batch_client_keeps_failed_predictions(monkeypatch):
    monkeypatch.setattr(
        predict_module,
        "_build_batch_client",
        lambda batch_client_name, lm: {"name": batch_client_name, "lm": lm},
    )
    arbiter = _FakeArbiter(
        batches=[[_make_prediction("spam"), FailedPrediction("parse failure", index=1)]]
    )
    dataset = Dataset.from_dict(
        {
            "subject": ["offer", "meeting"],
            "body": ["buy now", "agenda"],
        }
    )

    result = predict_module.process_dataset(
        arbiter=arbiter,
        dataset=dataset,
        input_fields=["subject", "body"],
        output_fields=["is_spam"],
        batch_client_name="vllm",
    )

    assert result.column_names == [
        "subject",
        "body",
        "is_spam",
        "prediction",
        "messages",
        "prediction_error",
    ]
    assert result["is_spam"] == ["spam", None]
    assert result["prediction_error"] == [None, "parse failure"]


def test_process_dataset_batch_client_skips_failed_predictions_for_consistency(monkeypatch):
    monkeypatch.setattr(
        predict_module,
        "_build_batch_client",
        lambda batch_client_name, lm: {"name": batch_client_name, "lm": lm},
    )
    monkeypatch.setattr(random, "sample", lambda population, k: list(population)[:k])

    arbiter = _FakeArbiter(
        batches=[
            [_make_prediction("spam"), FailedPrediction("parse failure", index=1)],
            [_make_prediction("spam") for _ in range(10)],
        ]
    )
    dataset = Dataset.from_dict(
        {
            "subject": ["offer", "meeting"],
            "body": ["buy now", "agenda"],
        }
    )

    result = predict_module.process_dataset(
        arbiter=arbiter,
        dataset=dataset,
        input_fields=["subject", "body"],
        output_fields=["is_spam"],
        batch_client_name="vllm",
        self_consistency=1.0,
    )

    assert result["prediction_error"] == [None, "parse failure"]
    assert result["sc_confidence"] == [1.0, None]


def test_build_batch_client_rejects_non_huggingface_models():
    with pytest.raises(ValueError, match="huggingface/Qwen/Qwen3.5-4B"):
        predict_module._build_batch_client("modal", _FakeLM("hosted_vllm/Qwen/Qwen3.5-4B"))


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
    assert "prediction" in ds.column_names
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
    assert "prediction" in ds.column_names
    assert "messages" in ds.column_names
    assert len(ds) == 10


def test_predict_ollama_self_consistency(tmp_path, ollama_server):
    output = tmp_path / "output.jsonl"
    result = subprocess.run(
        [
            "predict",
            "--arbiter", "tests/artifacts/arbiter_ollama.yaml",
            "--input", "tests/artifacts/data.jsonl",
            "--output", str(output),
            "-sc", "0.5",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"predict failed:\n{result.stderr}"

    ds = Dataset.from_json(str(output))
    assert "is_spam" in ds.column_names
    assert "prediction" in ds.column_names
    assert "messages" in ds.column_names
    assert "sc_confidence" in ds.column_names
    assert len(ds) == 10

    # 50% of 10 rows = 5 rows should have sc_confidence scores
    sc_scores = [row["sc_confidence"] for row in ds if row["sc_confidence"] is not None]
    assert len(sc_scores) == 5
    for score in sc_scores:
        assert 0.0 <= score <= 1.0


def test_predict_ollama_cross_consistency(tmp_path, ollama_server):
    # Create council.yaml with two ollama model entries (same model for testing)
    council_path = tmp_path / "council.yaml"
    council_spec = [
        {"model": f"ollama/{OLLAMA_MODEL}", "api_base": OLLAMA_BASE_URL},
        {"model": f"ollama/{OLLAMA_MODEL}", "api_base": OLLAMA_BASE_URL},
    ]
    with open(council_path, "w") as f:
        yaml.safe_dump(council_spec, f)

    output = tmp_path / "output.jsonl"
    result = subprocess.run(
        [
            "predict",
            "--arbiter", "tests/artifacts/arbiter_ollama.yaml",
            "--input", "tests/artifacts/data.jsonl",
            "--output", str(output),
            "-cc", "0.5",
            "--council", str(council_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"predict failed:\n{result.stderr}"

    ds = Dataset.from_json(str(output))
    assert "is_spam" in ds.column_names
    assert "prediction" in ds.column_names
    assert "messages" in ds.column_names
    assert "cc_confidence" in ds.column_names
    assert "sc_confidence" not in ds.column_names
    assert len(ds) == 10

    # 50% of 10 rows = 5 rows should have cc_confidence scores
    cc_scores = [row["cc_confidence"] for row in ds if row["cc_confidence"] is not None]
    assert len(cc_scores) == 5
    for score in cc_scores:
        assert 0.0 <= score <= 1.0


def test_predict_ollama_sc_and_cc(tmp_path, ollama_server):
    # Create council.yaml
    council_path = tmp_path / "council.yaml"
    council_spec = [
        {"model": f"ollama/{OLLAMA_MODEL}", "api_base": OLLAMA_BASE_URL},
        {"model": f"ollama/{OLLAMA_MODEL}", "api_base": OLLAMA_BASE_URL},
    ]
    with open(council_path, "w") as f:
        yaml.safe_dump(council_spec, f)

    output = tmp_path / "output.jsonl"
    result = subprocess.run(
        [
            "predict",
            "--arbiter", "tests/artifacts/arbiter_ollama.yaml",
            "--input", "tests/artifacts/data.jsonl",
            "--output", str(output),
            "-sc", "0.3",
            "-cc", "0.5",
            "--council", str(council_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"predict failed:\n{result.stderr}"

    ds = Dataset.from_json(str(output))
    assert "is_spam" in ds.column_names
    assert "prediction" in ds.column_names
    assert "messages" in ds.column_names
    assert "sc_confidence" in ds.column_names
    assert "cc_confidence" in ds.column_names
    assert len(ds) == 10

    # 30% of 10 = 3 rows with SC scores
    sc_scores = [(i, row["sc_confidence"]) for i, row in enumerate(ds) if row["sc_confidence"] is not None]
    assert len(sc_scores) == 3

    # 50% of remaining 7 = 3 rows with CC scores (int(7 * 0.5) = 3, but max(1,...))
    cc_scores = [(i, row["cc_confidence"]) for i, row in enumerate(ds) if row["cc_confidence"] is not None]
    assert len(cc_scores) == 3

    # No row should have both SC and CC scores
    sc_row_indices = {i for i, _ in sc_scores}
    cc_row_indices = {i for i, _ in cc_scores}
    assert sc_row_indices.isdisjoint(cc_row_indices), "SC and CC rows must not overlap"

    for _, score in sc_scores + cc_scores:
        assert 0.0 <= score <= 1.0

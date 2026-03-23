from pathlib import Path

from datasets import Dataset

from tests.modal.modal_test_app import Reflect, app, probes_data_vol

ARBITER_YAML = "examples/spam/arbiter.yaml"
INPUT_FILE = "tests/artifacts/predicted.jsonl"
OUTPUT_DIR = Path("tests/artifacts/tmp")


def _get_model():
    """Get probe model path from arbiter config."""
    from probes2.inference.predict import build_arbiter

    arbiter, _ = build_arbiter(ARBITER_YAML)
    return arbiter._arbiter_probe


def _run_reflect(n_gpus=1):
    model = _get_model()
    remote_input = "/data/predicted.jsonl"
    remote_output = "/data/output.parquet"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local_output = OUTPUT_DIR / f"output_{n_gpus}gpu.parquet"

    # Upload input to volume
    with probes_data_vol.batch_upload(force=True) as batch:
        batch.put_file(INPUT_FILE, "/predicted.jsonl")

    # Run reflect on Modal
    gpu = f"H100:{n_gpus}" if n_gpus > 1 else "H100"
    with app.run():
        reflect = Reflect.with_options(gpu=gpu)()
        args = [
            "--input", remote_input,
            "--output", remote_output,
            "--model", model,
            "--device", "cuda",
            "--n-gpus", str(n_gpus),
        ]
        reflect.run.remote(*args)

    # Download output from volume
    data = b""
    for chunk in probes_data_vol.read_file("/output.parquet"):
        data += chunk
    local_output.write_bytes(data)

    return local_output


def test_reflect_1gpu():
    output = _run_reflect(n_gpus=1)
    ds = Dataset.from_parquet(str(output))
    assert "confidence" in ds.column_names
    assert len(ds) == 10


def test_reflect_4gpu():
    output = _run_reflect(n_gpus=4)
    ds = Dataset.from_parquet(str(output))
    assert "confidence" in ds.column_names
    assert len(ds) == 10

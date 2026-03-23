import modal

app = modal.App("test-probes-reflect")

probes_data_vol = modal.Volume.from_name("probes-test-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["cu128"])
    .pip_install("modaic @ git+https://github.com/modaic-ai/modaic.git@09b61bc48b2378a38b960d4f7793da4dbd576b92#subdirectory=src/modaic-sdk")
    .add_local_python_source("probes2")
)


@app.cls(
    image=image,
    gpu="H100",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/data": probes_data_vol,
    },
)
class Reflect:
    @modal.method()
    def run(self, *args: str):
        from probes2.inference.reflect import main

        try:
            main(list(args))
        except SystemExit as e:
            if e.code != 0:
                raise
        # Commit volume so output files are visible to the caller
        probes_data_vol.commit()

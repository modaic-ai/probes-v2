---
name: use-modal
description: Use when working on Modal jobs, Modal images, or Python packages that are mounted into Modal containers for local development. Covers editable local packages via tool.uv.sources, uv sync, add_local_python_source, and quick alignment checks.
metadata:
  short-description: Modal local-package development
---

# Use Modal

Use this skill when editing Modal entrypoints, Modal images, or repo-local Python packages that must be visible inside a Modal container.

## Preferred Setup

For local package development, prefer a local editable source in `pyproject.toml`:

```toml
[tool.uv.sources]
modaic = { path = "../modaic/src/modaic-sdk", editable = true }
```

Then run:

```bash
TMPDIR=/tmp uv sync
```

After that, `import modaic` in this repo should resolve to the sibling checkout, not a stale installed wheel.

## Modal Packaging Rule

If the package resolves correctly in the current environment, prefer:

```python
.add_local_python_source("modaic")
```

over hardcoded sibling-path mounts such as `.add_local_dir("../modaic/...")`.

Reason:
- It follows the active Python import resolution for this repo.
- It is simpler and less fragile than hand-managed `PYTHONPATH` or sibling mount paths.
- It matches local editable-package development once `uv sync` has been run.

## Sanity Check

When Modal/package alignment is in doubt:

1. Add a tiny temporary no-op marker function to the local package.
2. Import that function in the Modal entrypoint module.
3. Add a cheap Modal probe function that prints/returns the marker before the long-running workload starts.
4. Run the probe directly with `uv run modal run path/to/file.py::probe_fn`.

If the remote output shows the marker, the container is using the intended local package source.

## Debugging Order

1. Verify `python -c "import modaic; print(modaic.__file__)"` points at the local editable checkout.
2. Verify `pyproject.toml` uses `[tool.uv.sources]` with `path = ...` and `editable = true`.
3. Run `TMPDIR=/tmp uv sync`.
4. Keep Modal images on `.add_local_python_source("modaic")` unless there is a specific reason not to.
5. Only reach for manual sibling mounts or `PYTHONPATH` overrides if the import-based approach is impossible.

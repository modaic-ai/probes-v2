# Probes V2 Agent Guide

## Modal Skill

When working on Modal in this repo, read [`skills/use-modal/SKILL.md`](/Users/tytodd/Desktop/Modaic/code/core/probes-v2/skills/use-modal/SKILL.md) first.

Preferred development setup for local Python packages:

- Use `[tool.uv.sources]` with a local editable path, for example:
  `modaic = { path = "../modaic/src/modaic-sdk", editable = true }`
- Run `TMPDIR=/tmp uv sync` after changing the source mapping or when the local package checkout changes materially.
- If the package import resolves to the local checkout, prefer `modal.Image.add_local_python_source("modaic")` over hardcoded sibling-directory mounts.

For Modal/package alignment checks, prefer a tiny temporary no-op export in the local package plus a lightweight Modal probe function before debugging long-running jobs.

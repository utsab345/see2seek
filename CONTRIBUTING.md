# Contributing to See2Seek

Thanks for your interest. This project is a research codebase; issues and pull
requests are welcome. Please keep the bar high — reviewers will hold you to it.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install            # lint + format gates on every commit
```

Data is **not** bundled. See `docs/DATASET.md` (if present) or the tools in
`see2seek/utils/` (`generate_data.py`, `augment_goal_angles.py`,
`image2vec.py`) to build a dataset, then point `configs/*.yaml` at it.

## Before you submit

1. **Format** — `black --check .` (line length 100).
2. **Lint** — `ruff check .`
3. **Types** — `mypy`
4. **Tests** — `pytest`
5. Run the full set locally; CI runs exactly these four gates.

```bash
make lint        # ruff + black --check
make check       # mypy
make test        # pytest
```

## Conventions

- **Commits:** one logical change per commit with a folder-scoped subject:
  `feat(agents): ...`, `fix(envs): ...`, `docs(trainers): ...`,
  `chore(configs): ...`. Subject only — keep descriptions in the body.
- **Type hints** on all new public functions; `mypy` must stay clean.
- **No secrets or large binaries** in the repo. Generated artifacts
  (`*.pth`, `embeddings.pt`, `wandb/`, `logs/`) are gitignored.
- New functionality that touches reward logic, encoders, or the rollout
  buffer must come with a test or an updated ablation explanation.

## Structure

- `see2seek/` — installable package (agents, buffers, envs, models,
  trainers, evaluation, utils).
- `scripts/` — CLI entry points (train / eval / visualize / plot).
- `configs/` — YAML overrides; `tests/` — unit tests (no sim required).
- `docs/` — architecture diagram, figures, and prose docs.

## Reporting issues

Use the issue tracker. Include: environment (`pip list` relevant rows), the
exact command, a minimal reproduction, and the full traceback. For dataset or
simulator (ai2thor) problems, say which RoboTHOR version and how you obtained
the data.
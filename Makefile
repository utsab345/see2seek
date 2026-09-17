# See2Seek developer convenience targets.
# Gates mirror CI (./.github/workflows/ci.yml).

PYTHON ?= python

.PHONY: install lint format check test pre-commit clean

install:
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pre_commit install

format:
	$(PYTHON) -m black .
	$(PYTHON) -m ruff check . --fix

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

check:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist .eggs *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
# uv puts the virtualenv in .venv.nosync so iCloud doesn't sync it.
export UV_PROJECT_ENVIRONMENT := .venv.nosync

.PHONY: setup check lint types test train generate

setup:
	uv sync
	git config core.hooksPath .githooks

check: lint types test

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run pyright

test:
	uv run pytest -q

train:
	uv run local-llm train

generate:
	uv run local-llm generate

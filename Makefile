.PHONY: sync lint fmt test test-integration up down

sync:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

test-integration:
	docker compose up -d --wait redis postgres
	uv run pytest -m integration

up:
	docker compose up -d --build

down:
	docker compose down

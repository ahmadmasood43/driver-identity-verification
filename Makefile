.PHONY: sync lint fmt test test-integration up down models midv

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

# insightface buffalo_l (non-commercial research licence): SCRFD-10G detector, ArcFace R50 (WebFace600K).
BUFFALO_URL := https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
BUFFALO_SHA := 80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f

models:
	@tmp=$$(mktemp -d) && \
	curl -sSfL -o $$tmp/buffalo_l.zip $(BUFFALO_URL) && \
	echo "$(BUFFALO_SHA)  $$tmp/buffalo_l.zip" | sha256sum -c --quiet && \
	unzip -q -o -j $$tmp/buffalo_l.zip det_10g.onnx w600k_r50.onnx -d models && \
	rm -rf $$tmp
	uv run python -c "from pathlib import Path; from idv.data.manifest import verify; \
	bad = verify(Path('models'), Path('models/manifest.csv')); assert not bad, bad; print('models ok')"

# MIDV-2020 (CC BY-SA 2.5). Archives are fetched by hand from ftp://smartengines.com/midv-2020/
# into data/raw/midv2020/archives; this unpacks each into its own folder and checks it.
MIDV := data/raw/midv2020
MIDV_PARTS := templates photo scan_upright scan_rotated clips

midv:
	@for p in $(MIDV_PARTS); do \
	  test -d $(MIDV)/$$p || { mkdir -p $(MIDV)/$$p && tar -xf $(MIDV)/archives/$$p.tar -C $(MIDV)/$$p; }; \
	  uv run python -m idv.data.manifest verify $(MIDV)/$$p data/manifests/midv2020_$$p.csv && echo "$$p ok"; \
	done

# driver-identity-verification

Identity checks for driver onboarding: the applicant submits a selfie and a
government ID, and the system approves, rejects, or sends the case to a human
reviewer. Components: document localization and OCR, selfie-to-ID face
matching, image quality, presentation-attack detection, and score-based review
routing.

Work in progress. The serving path runs end to end with a fake pipeline; the
models come next. No results yet. The evaluation is pre-registered in
[eval/protocol.md](eval/protocol.md).

## Layout

```
src/idv/eval/       metrics (FAR/FRR, ISO 30107-3 PAD errors), subject-level bootstrap
src/idv/data/       data manifests, seeded subject-disjoint splits, demographic bins
src/idv/face/       alignment, SCRFD detector, ArcFace embedder, verification pairs
src/idv/tracking.py MLflow runs tagged with git SHA and manifest hashes
src/idv/pipeline/   model pipeline interface; fake implementation for now
src/idv/service/    FastAPI app, arq worker, Redis blob store, Postgres records
deploy/             Dockerfile, Prometheus, Grafana provisioning
eval/protocol.md    pre-registered evaluation protocol
```

## Running

Requires uv and Docker. `uv sync` also installs the GPU stack (torch from the cu126
index, onnxruntime-gpu 1.26), pinned there because CUDA 13 builds no longer support
the Pascal card this is developed on. CI and the API image skip it.

```
make sync              # create .venv
make test              # unit tests
make models            # fetch face detector + embedder weights, check checksums
uv run pytest -m gpu   # real-model tests, needs CUDA and models/
make test-integration  # starts Redis + Postgres, runs API/worker tests
make up                # full stack: api :8000, prometheus :9090, grafana :3000, mlflow :5000
```

Submit a verification:

```
curl -F selfie=@selfie.jpg -F id_front=@id.jpg localhost:8000/v1/verifications
curl localhost:8000/v1/verifications/<id>
```

Images are kept only in Redis (persistence off) and deleted once a decision is
written. Postgres stores decisions, scores and reason codes, not images.

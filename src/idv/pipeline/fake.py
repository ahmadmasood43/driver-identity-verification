import hashlib

from idv.pipeline.types import Decision, PipelineResult


class FakePipeline:
    """Deterministic stand-in so the serving path runs without models or a GPU."""

    def run(self, selfie: bytes, id_front: bytes) -> PipelineResult:
        h = hashlib.sha256(selfie + id_front).digest()
        return PipelineResult(
            decision=Decision.REVIEW,
            scores={"match": h[0] / 255, "liveness": h[1] / 255, "quality": h[2] / 255},
            reason_codes=["fake_pipeline"],
            model_versions={"pipeline": "fake-0"},
        )

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime

from arq.connections import RedisSettings
from prometheus_client import start_http_server

from idv.pipeline import load_pipeline
from idv.pipeline.types import PipelineResult
from idv.service import blobs
from idv.service.config import Settings
from idv.service.db import Verification, make_engine, make_sessions
from idv.service.metrics import DECIDED, LATENCY, PIPELINE_SECONDS

log = logging.getLogger(__name__)


async def startup(ctx: dict) -> None:
    settings = Settings()
    ctx["engine"] = make_engine(settings.database_url)
    ctx["sessions"] = make_sessions(ctx["engine"])
    ctx["pipeline"] = load_pipeline(settings.pipeline)
    if settings.worker_metrics_port:
        start_http_server(settings.worker_metrics_port)


async def shutdown(ctx: dict) -> None:
    await ctx["engine"].dispose()


async def _record(
    ctx: dict, vid: uuid.UUID, status: str, result: PipelineResult | None, reasons: list[str]
) -> None:
    async with ctx["sessions"]() as s:
        v = await s.get(Verification, vid)
        if v is None:
            log.error("verification %s missing from db", vid)
            return
        v.status = status
        v.reason_codes = reasons
        v.decided_at = datetime.now(UTC)
        if result is not None:
            v.decision = result.decision.value
            v.scores = result.scores
            v.model_versions = result.model_versions
        await s.commit()
    DECIDED.labels(status, result.decision.value if result else "none").inc()


async def verify(ctx: dict, verification_id: str, submitted_at: float) -> None:
    vid = uuid.UUID(verification_id)
    redis = ctx["redis"]
    try:
        images = await blobs.get(redis, vid)
        if images is None:
            await _record(ctx, vid, "failed", None, ["blob_expired"])
            return
        t0 = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                ctx["pipeline"].run, images["selfie"], images["id_front"]
            )
        except Exception:
            log.exception("pipeline failed for %s", vid)
            await _record(ctx, vid, "failed", None, ["pipeline_error"])
            return
        PIPELINE_SECONDS.observe(time.perf_counter() - t0)
        await _record(ctx, vid, "done", result, result.reason_codes)
        LATENCY.observe(time.time() - submitted_at)
    finally:
        # Images go as soon as there is a decision, whatever the outcome.
        await blobs.delete(redis, vid)


class WorkerSettings:
    functions = (verify,)
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(Settings().redis_url)
    # One pipeline at a time on a single GPU; revisit with batching in M6.
    max_jobs = Settings().worker_concurrency
    max_tries = 1

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, HTTPException, Request, UploadFile
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from idv.service import blobs
from idv.service.config import Settings
from idv.service.db import Verification, init_db, make_engine, make_sessions
from idv.service.metrics import REJECTED_UPLOADS, SUBMITTED

_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    app.state.redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    app.state.engine = make_engine(settings.database_url)
    await init_db(app.state.engine)
    app.state.sessions = make_sessions(app.state.engine)
    yield
    await app.state.redis.aclose()
    await app.state.engine.dispose()


app = FastAPI(title="idv", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class Submitted(BaseModel):
    id: uuid.UUID
    status: str


class Result(BaseModel):
    id: uuid.UUID
    status: str
    decision: str | None
    reason_codes: list[str]
    scores: dict[str, float]
    model_versions: dict[str, str]
    submitted_at: datetime
    decided_at: datetime | None


async def _read_image(f: UploadFile, limit: int) -> bytes:
    data = await f.read(limit + 1)
    if len(data) > limit:
        REJECTED_UPLOADS.labels("too_large").inc()
        raise HTTPException(413, "image too large")
    if not data.startswith(_MAGIC):
        REJECTED_UPLOADS.labels("bad_type").inc()
        raise HTTPException(415, "expected JPEG or PNG")
    return data


@app.post("/v1/verifications", status_code=202)
async def submit(request: Request, selfie: UploadFile, id_front: UploadFile) -> Submitted:
    submitted_at = datetime.now(UTC)
    st = request.app.state
    limit = st.settings.max_upload_bytes
    images = {
        "selfie": await _read_image(selfie, limit),
        "id_front": await _read_image(id_front, limit),
    }

    vid = uuid.uuid4()
    await blobs.put(st.redis, vid, images, st.settings.blob_ttl_s)
    async with st.sessions() as s:
        s.add(Verification(id=vid, status="pending", submitted_at=submitted_at))
        await s.commit()
    await st.redis.enqueue_job("verify", str(vid), submitted_at.timestamp(), _job_id=str(vid))
    SUBMITTED.inc()
    return Submitted(id=vid, status="pending")


@app.get("/v1/verifications/{vid}")
async def get_result(request: Request, vid: uuid.UUID) -> Result:
    async with request.app.state.sessions() as s:
        v = await s.get(Verification, vid)
    if v is None:
        raise HTTPException(404, "not found")
    return Result.model_validate(v, from_attributes=True)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}

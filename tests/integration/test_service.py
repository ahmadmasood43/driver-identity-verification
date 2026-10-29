"""API -> Redis -> worker -> Postgres with the fake pipeline. Needs Redis and Postgres running."""

import uuid

import httpx
import pytest
from arq.connections import RedisSettings
from arq.worker import Worker

from idv.service import blobs
from idv.service.api import app
from idv.service.config import Settings
from idv.service.worker import shutdown, startup, verify

pytestmark = pytest.mark.integration

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


@pytest.fixture(autouse=True)
def _no_worker_metrics_server(monkeypatch):
    monkeypatch.setenv("IDV_WORKER_METRICS_PORT", "0")


@pytest.fixture
async def client():
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def run_worker() -> None:
    worker = Worker(
        functions=[verify],
        on_startup=startup,
        on_shutdown=shutdown,
        redis_settings=RedisSettings.from_dsn(Settings().redis_url),
        burst=True,
        poll_delay=0.05,
        max_tries=1,
    )
    try:
        await worker.main()
    finally:
        await worker.close()


def _files(selfie=JPEG, id_front=PNG):
    return {
        "selfie": ("selfie.jpg", selfie, "image/jpeg"),
        "id_front": ("id.png", id_front, "image/png"),
    }


async def test_submit_decide_and_delete_images(client):
    r = await client.post("/v1/verifications", files=_files())
    assert r.status_code == 202
    vid = uuid.UUID(r.json()["id"])
    assert await blobs.exists(app.state.redis, vid) == 2

    pending = (await client.get(f"/v1/verifications/{vid}")).json()
    assert pending["status"] == "pending"

    await run_worker()

    body = (await client.get(f"/v1/verifications/{vid}")).json()
    assert body["status"] == "done"
    assert body["decision"] == "review"
    assert set(body["scores"]) == {"match", "liveness", "quality"}
    assert body["decided_at"] is not None
    assert await blobs.exists(app.state.redis, vid) == 0


async def test_expired_blobs_fail_cleanly(client):
    r = await client.post("/v1/verifications", files=_files())
    vid = uuid.UUID(r.json()["id"])
    await blobs.delete(app.state.redis, vid)

    await run_worker()

    body = (await client.get(f"/v1/verifications/{vid}")).json()
    assert body["status"] == "failed"
    assert body["reason_codes"] == ["blob_expired"]


async def test_rejects_non_image(client):
    r = await client.post("/v1/verifications", files=_files(selfie=b"GIF89a" + b"\x00" * 10))
    assert r.status_code == 415


async def test_rejects_oversized(client, monkeypatch):
    monkeypatch.setattr(app.state.settings, "max_upload_bytes", 32)
    r = await client.post("/v1/verifications", files=_files())
    assert r.status_code == 413


async def test_unknown_id_is_404(client):
    r = await client.get(f"/v1/verifications/{uuid.uuid4()}")
    assert r.status_code == 404

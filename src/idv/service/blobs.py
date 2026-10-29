"""Short-lived image storage in Redis. Redis runs with persistence off, so blobs never hit disk."""

import uuid

from redis.asyncio import Redis

KINDS = ("selfie", "id_front")


def _key(vid: uuid.UUID, kind: str) -> str:
    return f"blob:{vid}:{kind}"


async def put(redis: Redis, vid: uuid.UUID, blobs: dict[str, bytes], ttl_s: int) -> None:
    async with redis.pipeline(transaction=True) as p:
        for kind in KINDS:
            p.set(_key(vid, kind), blobs[kind], ex=ttl_s)
        await p.execute()


async def get(redis: Redis, vid: uuid.UUID) -> dict[str, bytes] | None:
    values = await redis.mget([_key(vid, k) for k in KINDS])
    if any(v is None for v in values):
        return None
    return dict(zip(KINDS, values, strict=True))


async def delete(redis: Redis, vid: uuid.UUID) -> None:
    await redis.delete(*(_key(vid, k) for k in KINDS))


async def exists(redis: Redis, vid: uuid.UUID) -> int:
    return await redis.exists(*(_key(vid, k) for k in KINDS))

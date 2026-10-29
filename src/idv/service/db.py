"""Decision records. No images, names, DOBs or embeddings are stored here."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Verification(Base):
    __tablename__ = "verifications"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(16))  # pending | done | failed
    decision: Mapped[str | None] = mapped_column(String(16))
    reason_codes: Mapped[list[str]] = mapped_column(JSONB, default=list)
    scores: Mapped[dict[str, float]] = mapped_column(JSONB, default=dict)
    model_versions: Mapped[dict[str, str]] = mapped_column(JSONB, default=dict)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True)


def make_sessions(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    # Fine until the schema needs to change; switch to Alembic then.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

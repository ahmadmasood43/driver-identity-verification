from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IDV_")

    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+psycopg://idv:idv@localhost:5432/idv"
    pipeline: str = "fake"
    # Uploaded images live only in Redis, and only this long if the worker never gets to them.
    blob_ttl_s: int = 300
    max_upload_bytes: int = 10 * 1024 * 1024
    # 0 disables the worker's metrics server (tests).
    worker_metrics_port: int = 9101
    worker_concurrency: int = 1

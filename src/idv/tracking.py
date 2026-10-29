"""Every training or eval run goes through `tracked_run` so results trace back to code,
config and data."""

import json
import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from idv.data.manifest import manifest_hash

DEFAULT_TRACKING_URI = "http://localhost:5000"


class DirtyTreeError(RuntimeError):
    pass


def git_state(repo: Path | str = ".") -> tuple[str, bool]:
    """(HEAD sha, whether tracked or untracked-but-not-ignored files differ from HEAD)."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout

    return git("rev-parse", "HEAD").strip(), bool(git("status", "--porcelain").strip())


def resolve_sha(repo: Path | str = ".", allow_dirty: bool = False) -> str:
    sha, dirty = git_state(repo)
    if not dirty:
        return sha
    if not allow_dirty:
        raise DirtyTreeError("uncommitted changes; commit first or pass --allow-dirty")
    return f"{sha}-dirty"


def flatten(cfg: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


@contextmanager
def tracked_run(
    experiment: str,
    cfg: Mapping[str, Any],
    manifests: list[Path] = (),
    *,
    allow_dirty: bool = False,
    run_name: str | None = None,
) -> Iterator[Any]:
    # Check the tree before touching MLflow so a dirty run leaves nothing behind.
    sha = resolve_sha(allow_dirty=allow_dirty)

    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"git_sha": sha, "dirty": str(sha.endswith("-dirty")).lower()})
        for m in manifests:
            mlflow.set_tag(f"manifest.{Path(m).stem}", manifest_hash(Path(m)))
        mlflow.log_params({k: str(v)[:500] for k, v in flatten(cfg).items()})
        mlflow.log_text(json.dumps(cfg, indent=2, sort_keys=True, default=str), "config.json")
        yield run

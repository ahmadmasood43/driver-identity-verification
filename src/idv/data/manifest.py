"""Data manifests: relative path, sha256 and size of every file a run reads.

Manifests are committed; the data itself is not. A run logs the hash of each manifest
it used, so any result can be tied back to the exact files.
"""

import csv
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

FIELDS = ("path", "sha256", "bytes")


@dataclass(frozen=True)
class Entry:
    path: str
    sha256: str
    bytes: int


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build(root: Path, files: Iterable[Path]) -> list[Entry]:
    entries = []
    for f in files:
        f = Path(f)
        full = f if f.is_absolute() else root / f
        entries.append(
            Entry(full.relative_to(root).as_posix(), file_sha256(full), full.stat().st_size)
        )
    return sorted(entries, key=lambda e: e.path)


def write(entries: Iterable[Entry], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FIELDS)
        for e in sorted(entries, key=lambda e: e.path):
            w.writerow((e.path, e.sha256, e.bytes))


def read(path: Path) -> list[Entry]:
    with open(path, newline="") as f:
        return [Entry(r["path"], r["sha256"], int(r["bytes"])) for r in csv.DictReader(f)]


def verify(root: Path, path: Path) -> list[str]:
    """Paths that are missing or whose contents changed. Empty means the data matches."""
    bad = []
    for e in read(path):
        full = root / e.path
        if not full.is_file() or full.stat().st_size != e.bytes or file_sha256(full) != e.sha256:
            bad.append(e.path)
    return bad


def manifest_hash(path: Path) -> str:
    return file_sha256(path)

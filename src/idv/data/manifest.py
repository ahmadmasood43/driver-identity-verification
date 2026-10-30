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
    """Entries for `files` (absolute or relative to the cwd), which must sit under `root`."""
    root = Path(root).resolve()
    entries = []
    for f in files:
        full = Path(f).resolve()
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


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m idv.data.manifest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="manifest every file under ROOT")
    b.add_argument("root", type=Path)
    b.add_argument("out", type=Path)
    v = sub.add_parser("verify", help="check ROOT against a manifest")
    v.add_argument("root", type=Path)
    v.add_argument("manifest", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "build":
        files = sorted(p for p in args.root.rglob("*") if p.is_file())
        write(build(args.root, files), args.out)
        print(f"{len(files)} files -> {args.out}")
    else:
        bad = verify(args.root, args.manifest)
        for p in bad:
            print(f"MISMATCH {p}")
        raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()

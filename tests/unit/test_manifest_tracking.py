import subprocess

import pytest

from idv.data import manifest
from idv.data.demographics import age_band, fitzpatrick_group
from idv.tracking import DirtyTreeError, flatten, resolve_sha


def test_manifest_roundtrip_and_verify(tmp_path):
    root = tmp_path / "data"
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.jpg").write_bytes(b"xx")
    (root / "y.jpg").write_bytes(b"yyy")
    entries = manifest.build(root, [root / "y.jpg", root / "a" / "x.jpg"])
    assert [e.path for e in entries] == ["a/x.jpg", "y.jpg"]

    m = tmp_path / "m.csv"
    manifest.write(entries, m)
    assert manifest.read(m) == entries
    assert manifest.verify(root, m) == []

    (root / "y.jpg").write_bytes(b"yyz")
    (root / "a" / "x.jpg").unlink()
    assert manifest.verify(root, m) == ["a/x.jpg", "y.jpg"]


def test_manifest_hash_changes_with_content(tmp_path):
    m = tmp_path / "m.csv"
    manifest.write([manifest.Entry("a", "0" * 64, 1)], m)
    h1 = manifest.manifest_hash(m)
    manifest.write([manifest.Entry("a", "1" * 64, 1)], m)
    assert manifest.manifest_hash(m) != h1


@pytest.fixture
def repo(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "f.py").write_text("1\n")
    git("add", ".")
    git("commit", "-qm", "init")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_clean_tree_gives_sha(repo):
    assert len(resolve_sha()) == 40


def test_dirty_tree_refused_unless_allowed(repo):
    (repo / "f.py").write_text("2\n")
    with pytest.raises(DirtyTreeError):
        resolve_sha()
    assert resolve_sha(allow_dirty=True).endswith("-dirty")


def test_untracked_file_counts_as_dirty(repo):
    (repo / "new.py").write_text("x\n")
    with pytest.raises(DirtyTreeError):
        resolve_sha()


def test_flatten():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


@pytest.mark.parametrize(
    ("age", "band"), [(18, "18-30"), (30, "18-30"), (31, "31-45"), (46, "46+")]
)
def test_age_band(age, band):
    assert age_band(age) == band


def test_fitzpatrick_group():
    assert [fitzpatrick_group(t) for t in range(1, 7)] == [
        "I-II",
        "I-II",
        "III-IV",
        "III-IV",
        "V-VI",
        "V-VI",
    ]
    with pytest.raises(ValueError):
        fitzpatrick_group(7)

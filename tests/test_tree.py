"""Scan the repository and its commit messages for material that must not be published.

Files and messages may hold no local paths, cluster job numbers, private
commit hashes, internal repository and account names, or the names of
internal result files. The names are stored as truncated SHA-256 digests
of lower-case words, so that this file does not contain them.

Placeholders for decisions the author has not yet made (`OPEN_MARKERS`)
are allowed during development and reported as a skip; a release, that is
an unpacked archive or a checkout whose HEAD carries a version tag, fails
while any remains.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

BANNED_WORDS = frozenset({
    "c857d09db23e6822", "c70eca6b0f88f44d", "57de4cf40144bdf7", "7d3194f79e645c42",
    "60965168ce762e94", "3ea125d0bff386e6", "5d72436256ada538", "487b91042c7cf27a",
    "32e83e92d45d71f6", "e12ce8285efc67c6", "c9ad8f2cc1294afa", "d3cccc73defe2a62",
    "a0ffbbd29f129a88",
    # Names of internal result files and folders of the reference implementation.
    "75b75345683a0f9b", "643aa5feafaf4a0d", "7adeb97615801e07", "11c7bc8a93678e84",
    "d2c5f4dc6f8c79d2", "9c3aa6ce0c5440fa",
})
HOME_PATH = re.compile(r"/(Users|home|scratch|private/var|private/tmp)/|[A-Za-z]:\\Users\\")
EIGHT_DIGITS = re.compile(r"(?<![\w.])\d{8}(?![\w.])")
SHORT_HASH = re.compile(r"(?<![\w-])[0-9a-f]{7,40}(?![\w-])")
WORD = re.compile(r"[a-z0-9_]+")
# Tracked files are source, SQL, documentation and configuration; digests
# of reference tensors are JSON under tests/fixtures. Anything else, above
# all a tensor or table file (.pt, .pkl, .npz, .parquet, .csv), could carry
# patient data and is refused.
SOURCE_SUFFIXES = {".py", ".sql", ".md", ".toml"}
SOURCE_NAMES = {".gitignore", "LICENSE", "LICENSE-mimic-code"}
FIXTURES = ROOT / "tests" / "fixtures"
OPEN_MARKERS = ("AUTHOR DECISION PENDING", "[label to confirm]")
# Folders a reader's own runs create in an unpacked release: the outputs of
# the README's commands and tool caches. Operating-system metadata files are
# ignored as well.
UNTRACKED = {".git", "__pycache__", ".pytest_cache", "runs", "cache", "tables"}
OS_METADATA = {".DS_Store", "Thumbs.db", "desktop.ini"}


def word_digest(word: str) -> str:
    return hashlib.sha256(word.encode()).hexdigest()[:16]


def is_date(number: str) -> bool:
    """Seeds such as 20261006 are dates; cluster job numbers are not."""
    return number.startswith("20") and 1 <= int(number[4:6]) <= 12 and 1 <= int(number[6:]) <= 31


def problems(text: str, banned: frozenset[str] = BANNED_WORDS) -> list[str]:
    found = [f"home path {m.group(0)!r}" for m in HOME_PATH.finditer(text)]
    found += [f"job number {m.group(0)}" for m in EIGHT_DIGITS.finditer(text) if not is_date(m.group(0))]
    found += [f"commit hash {m.group(0)}" for m in SHORT_HASH.finditer(text)
              if re.search(r"[0-9]", m.group(0)) and re.search(r"[a-f]", m.group(0))]
    found += [f"banned word {w!r}" for w in sorted(set(WORD.findall(text.lower()))) if word_digest(w) in banned]
    return found


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    except (OSError, subprocess.CalledProcessError):
        return None


def _shipped(path: Path) -> bool:
    """Whether a file of an unpacked release came with it, judged by its place and name."""
    parts = path.relative_to(ROOT).parts
    if UNTRACKED & set(parts) or path.parent.name.endswith(".egg-info"):
        return False
    if path.name in OS_METADATA or path.name.startswith("._"):
        return False
    # A virtual environment inside the release is marked by its pyvenv.cfg.
    return not any((ROOT.joinpath(*parts[:i]) / "pyvenv.cfg").exists() for i in range(1, len(parts)))


def repository_files() -> list[Path]:
    """Tracked files in a git checkout; otherwise the files an unpacked release shipped.

    Without git, the walk skips the outputs of the README's commands
    (`runs/`, `cache/`, `tables/`), tool caches, virtual environments and
    operating-system metadata; anything else a reader adds is scanned, so
    the release check is meant for a fresh unpack.
    """
    listing = git("ls-files", "-z")
    if listing is not None:
        return [ROOT / name for name in listing.split("\0") if name]
    return [p for p in ROOT.rglob("*") if p.is_file() and _shipped(p)]


def is_allowed_file(path: Path) -> bool:
    if FIXTURES in path.parents:
        return path.suffix == ".json"
    return path.suffix in SOURCE_SUFFIXES or path.name in SOURCE_NAMES


def test_file_kind_detector():
    assert is_allowed_file(ROOT / "pulse" / "data.py") and is_allowed_file(FIXTURES / "any" / "x.json")
    assert not any(is_allowed_file(ROOT / name) for name in ("cache.pt", "cohort.pkl", "x.npz", "x.parquet",
                                                             "x.csv", "results.json"))
    assert not is_allowed_file(FIXTURES / "any" / "case.pt")


def test_release_walk_skips_reader_outputs():
    assert _shipped(ROOT / "pulse" / "data.py")
    assert not any(_shipped(ROOT / name) for name in ("tables/path_count.json", "runs/x/case_0000.pt",
                                                      ".DS_Store", "pulse/._data.py"))


def test_no_data_files_are_tracked():
    assert not [str(p.relative_to(ROOT)) for p in repository_files() if not is_allowed_file(p)]


REFERENCE = "reference implementation"
HEX_RUN = re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{7,40}(?![0-9A-Za-z])")


def test_fixture_metadata_names_no_commit():
    """Fixture metadata describe the reference in words.

    A commit of its private repository cannot be resolved by a reader.
    """
    for path in sorted(FIXTURES.glob("*/*.json")):
        meta = json.loads(path.read_text())["meta"]
        assert meta["reference"] == REFERENCE, path
        assert not [v for v in map(str, meta.values()) if HEX_RUN.search(v)], path
    assert HEX_RUN.search("revision " + "ABCDEF1") and HEX_RUN.search("deadbeef") and not HEX_RUN.search("2.13.0")


def test_detector():
    banned = BANNED_WORDS | {word_digest("zebra")}
    assert problems("see /" + "Users/someone/x and job 38" + "030223")
    assert problems("reference " + "ab12cd3")
    assert problems("written by Zebra", banned) and not problems("written by Zebra")
    assert not problems("seed 20261006, 7,147 patients, digest " + "ab" * 32)


def test_files_are_clean():
    report = {}
    for path in repository_files():
        if path == Path(__file__).resolve() or not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if found := problems(text):
            report[str(path.relative_to(ROOT))] = found
    assert not report, report


def test_commit_messages_are_clean():
    log = git("log", "--format=%H%n%B%x00")
    if log is None:
        pytest.skip("not a git checkout")
    report = {}
    for entry in filter(str.strip, log.split("\0")):
        commit, _, message = entry.strip().partition("\n")
        if found := problems(message):
            report[commit] = found
    assert not report, report


def is_release() -> bool:
    """An unpacked archive, or a checkout whose HEAD carries a tag starting with v."""
    tags = git("tag", "--points-at", "HEAD")
    return tags is None or any(tag.startswith("v") for tag in tags.split())


def open_markers() -> dict[str, list[str]]:
    report = {}
    for path in repository_files():
        if path == Path(__file__).resolve() or not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if found := [marker for marker in OPEN_MARKERS if marker in text]:
            report[str(path.relative_to(ROOT))] = found
    return report


def test_marker_detection(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("weights: " + OPEN_MARKERS[0] + "; arm " + OPEN_MARKERS[1])
    monkeypatch.setattr(sys.modules[__name__], "repository_files", lambda: [tmp_path / "README.md"])
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    assert open_markers() == {"README.md": list(OPEN_MARKERS)}


def test_no_open_decisions_in_a_release():
    report = open_markers()
    if report and not is_release():
        pytest.skip(f"open placeholders, allowed until a release: {report}")
    assert not report, report

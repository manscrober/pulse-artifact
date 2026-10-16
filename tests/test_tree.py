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
    # Names of internal result files and folders of the research code.
    "75b75345683a0f9b", "643aa5feafaf4a0d", "7adeb97615801e07", "11c7bc8a93678e84",
    "d2c5f4dc6f8c79d2", "9c3aa6ce0c5440fa",
})
HOME_PATH = re.compile(r"/(Users|home|scratch|private/var|private/tmp)/|[A-Za-z]:\\Users\\")
EIGHT_DIGITS = re.compile(r"(?<![\w.])\d{8}(?![\w.])")
SHORT_HASH = re.compile(r"(?<![\w-])[0-9a-f]{7,40}(?![\w-])")
WORD = re.compile(r"[a-z0-9_]+")
OPEN_MARKERS = ("AUTHOR DECISION PENDING", "[label to confirm]")
UNTRACKED = {".git", "__pycache__", ".pytest_cache", "runs", "cache"}


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


def repository_files() -> list[Path]:
    """Tracked files in a git checkout; otherwise every file of an unpacked release."""
    listing = git("ls-files", "-z")
    if listing is not None:
        return [ROOT / name for name in listing.split("\0") if name]
    return [p for p in ROOT.rglob("*") if p.is_file()
            and not UNTRACKED & set(p.relative_to(ROOT).parts) and not p.parent.name.endswith(".egg-info")]


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

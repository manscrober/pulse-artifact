"""Every command in sequence on a tiny synthetic cohort: build, train, roll out and score.

The database is replaced by synthetic stays and the split by 8, 4 and 3
subjects; everything else runs as published, with one training epoch and
two paths per patient.
"""
import json
import types

import pytest

from pulse import cohort, data
from pulse.__main__ import main
from synthetic import source_stays

SPLIT = {"train": 8, "val": 4, "test": 3}


# Variables constant over a tiny training set give undefined correlations,
# which the covariance grouping sets to zero.
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_build_train_roll_out_and_score(monkeypatch, tmp_path):
    stays = source_stays(seed=2, count=sum(SPLIT.values()) + 1)
    monkeypatch.setattr(data, "SPLIT", SPLIT)
    monkeypatch.setattr(cohort, "_connect", lambda dsn: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cohort, "cohort_subjects", lambda conn: (1, sorted(s["subject_id"] for _, s in stays)))
    monkeypatch.setattr(cohort, "load_stays", lambda conn, cohort_id, subjects: [
        stay for stay in stays if stay[1]["subject_id"] in subjects])
    cache, checkpoints, rollouts = tmp_path / "cache.pt", tmp_path / "checkpoints", tmp_path / "rollouts"

    main(["build-cache", "--dsn", "unused", "--out", str(cache)])
    main(["train", "--cache", str(cache), "--out", str(checkpoints), "--phases", "policy", "sample",
          "distribution", "--epochs", "1", "--batch", "4", "--micro", "2"])
    assert sorted(p.name for p in checkpoints.iterdir()) == ["distribution.pt", "policy.pt", "protocol.json", "sample.pt"]
    rollout = ["rollout", "--cache", str(cache), "--checkpoints", str(checkpoints), "--arms", "sample",
               "distribution", "--paths", "2", "--per-call", "2", "--out", str(rollouts)]
    main(rollout)
    written = {path: path.read_bytes() for path in rollouts.rglob("*") if path.is_file()}
    main(rollout)  # the same command into the same folder keeps every file as it was
    assert {path: path.read_bytes() for path in rollouts.rglob("*") if path.is_file()} == written
    for arm in ("sample", "distribution"):
        for origin in (2, 6):
            directory = rollouts / arm / f"k{origin}"
            assert sorted(p.name for p in directory.glob("case_*.pt")) == [f"case_{i:04d}.pt" for i in range(3)]
            assert json.loads((directory / "rollout.json").read_text())["per_call"] == 2
    table = tmp_path / "table.json"
    main(["score", "--cache", str(cache), "--protocol", str(checkpoints / "protocol.json"), "--rollouts",
          str(rollouts), "--arms", "sample", "distribution", "distribution:1", "--resamples", "20",
          "--out", str(table)])
    result = json.loads(table.read_text())
    assert result["reference"] == "sample"
    assert {row["arm"] for row in result["rows"]} == {"sample", "distribution", "distribution:1"}
    assert {row["origin"] for row in result["rows"]} == {2, 6}

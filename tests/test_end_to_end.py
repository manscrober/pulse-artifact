"""Every command in sequence on a tiny synthetic cohort: build, train, roll out and score.

The database is replaced by synthetic stays and the split by 8, 4 and 3
subjects; everything else runs as published, with one training epoch and
two paths per patient. The trained models are then rolled out and scored on
a synthetic eICU cache of 12 stays, whose test split holds 3.
"""
import json
import types

import pandas as pd
import pytest
import torch

from pulse import cohort, data, eicu
from pulse import variables as V
from pulse.__main__ import main
from synthetic import cache as synthetic_cache
from synthetic import source_stays

SPLIT = {"train": 8, "val": 4, "test": 3}


# Variables constant over a tiny training set give undefined correlations,
# which the covariance grouping sets to zero.
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_build_train_roll_out_and_score(monkeypatch, tmp_path):
    stays = source_stays(seed=2, count=sum(SPLIT.values()) + 1)
    monkeypatch.setattr(data, "SPLIT", SPLIT)
    monkeypatch.setattr(cohort, "_connect", lambda database: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cohort, "cohort_stays", lambda conn: pd.DataFrame(
        {"subject_id": [s["subject_id"] for _, s in stays], "stay_id": [s["stay_id"] for _, s in stays]}))
    monkeypatch.setattr(cohort, "load_stays", lambda conn, stay_ids, subjects: [
        stay for stay in stays if stay[1]["subject_id"] in subjects])
    cache, checkpoints, rollouts = tmp_path / "cache.pt", tmp_path / "checkpoints", tmp_path / "rollouts"

    main(["build-cache", "--database", "unused", "--out", str(cache)])
    main(["train", "--cache", str(cache), "--out", str(checkpoints), "--phases", "policy", "sample",
          "distribution", "--head", "rescnf", "--epochs", "1", "--batch", "4", "--micro", "2"])
    assert sorted(p.name for p in checkpoints.iterdir()) == ["distribution.pt", "policy.pt", "protocol.json", "sample.pt"]
    rollout = ["rollout", "--cache", str(cache), "--checkpoints", str(checkpoints), "--arms", "sample",
               "distribution", "--head", "rescnf", "--paths", "2", "--per-call", "2", "--out", str(rollouts)]
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

    transfer, transfer_rollouts = tmp_path / "eicu.pt", tmp_path / "eicu_rollouts"
    eicu.save_cache(eicu_like(), transfer)
    main(["rollout", "--cache", str(cache), "--checkpoints", str(checkpoints), "--arms", "sample", "--paths", "2",
          "--per-call", "2", "--head", "rescnf", "--eicu", str(transfer), "--out", str(transfer_rollouts)])
    record = json.loads((transfer_rollouts / "sample" / "k6" / "rollout.json").read_text())
    assert record["provenance"]["eicu_cache"] == eicu.load_cache(transfer)["sha256"]
    assert len(list((transfer_rollouts / "sample" / "k2").glob("case_*.pt"))) == 3
    table = tmp_path / "eicu_table.json"
    main(["score", "--eicu", str(transfer), "--protocol", str(checkpoints / "protocol.json"), "--rollouts",
          str(transfer_rollouts), "--arms", "sample", "carry_forward", "--resamples", "20", "--out", str(table)])
    rows = json.loads(table.read_text())["rows"]
    assert {(row["arm"], row["patients"]) for row in rows} == {("sample", 3), ("carry_forward", 3)}


def eicu_like() -> dict:
    """Twelve synthetic stays with eICU's recorded variables, no deaths or releases and unknown statics."""
    stays = synthetic_cache(seed=4, subjects=12, stays=12)
    recorded = [V.CONTINUOUS.index(n) for n in ("Heart Rate", "Systolic Blood Pressure", "Temperature", "Weight")]
    keep = torch.zeros(len(V.CONTINUOUS), dtype=torch.bool)
    keep[recorded] = True
    stays["continuous_mask"] = stays["continuous_mask"] * keep
    stays["continuous"] = torch.where(stays["continuous_mask"] > 0, stays["continuous"], torch.nan)
    stays["binary"][..., [V.DEATH, V.RELEASE]] = 0.0
    stays["static"][:, 1:3] = torch.nan
    stays["time_hours"] = torch.arange(24).float().expand(12, -1).clone()
    return stays


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_default_bspline_training_and_rollout(monkeypatch, tmp_path):
    stays = source_stays(seed=2, count=sum(SPLIT.values()) + 1)
    monkeypatch.setattr(data, "SPLIT", SPLIT)
    monkeypatch.setattr(cohort, "_connect", lambda database: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cohort, "cohort_stays", lambda conn: pd.DataFrame(
        {"subject_id": [s["subject_id"] for _, s in stays], "stay_id": [s["stay_id"] for _, s in stays]}))
    monkeypatch.setattr(cohort, "load_stays", lambda conn, stay_ids, subjects: [
        stay for stay in stays if stay[1]["subject_id"] in subjects])
    cache, checkpoints, rollouts = tmp_path / "cache.pt", tmp_path / "checkpoints", tmp_path / "rollouts"
    main(["build-cache", "--database", "unused", "--out", str(cache)])
    main(["train", "--cache", str(cache), "--out", str(checkpoints),
          "--epochs", "1", "--batch", "4", "--micro", "2"])
    payload = torch.load(checkpoints / "sample.pt", weights_only=True)
    assert payload["architecture"]["head"] == "bspline"
    main(["rollout", "--cache", str(cache), "--checkpoints", str(checkpoints), "--arms", "sample",
          "--paths", "2", "--per-call", "2", "--out", str(rollouts)])
    table = tmp_path / "table.json"
    main(["score", "--cache", str(cache), "--protocol", str(checkpoints / "protocol.json"),
          "--rollouts", str(rollouts), "--arms", "sample", "carry_forward", "--resamples", "20",
          "--out", str(table)])
    assert {row["arm"] for row in json.loads(table.read_text())["rows"]} == {"sample", "carry_forward"}

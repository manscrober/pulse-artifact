import importlib
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pulse import cohort, data, hourly
from synthetic import source_stays


def test_measurements_follow_items_and_repeat_labs_per_stay(monkeypatch):
    stays = pd.DataFrame({"stay_id": [1, 2, 3], "hadm_id": [10, 10, 11],
                          "intime": pd.to_datetime(["2150-01-01", "2150-01-03", "2150-02-01"])})
    chart = pd.DataFrame({"stay_id": [3, 1, 1], "itemid": [223761, 223762, 220045], "charttime": pd.NaT,
                          "valuenum": [99.0, 37.0, 80.0]})
    labs = pd.DataFrame({"hadm_id": [10, 11], "itemid": [51265, 51265], "charttime": pd.NaT, "valuenum": [150.0, 90.0]})
    monkeypatch.setattr(cohort, "_read", lambda conn, query, params=None: chart if "chartevents" in query else labs)
    rows = cohort._measurements(None, stays)
    assert rows["variable"].tolist() == ["Heart Rate", "Temperature", "Temperature", "Platelet", "Platelet", "Platelet"]
    assert rows["itemid"].tolist()[1:3] == [223761, 223762]
    assert rows.loc[rows["variable"] == "Platelet", "stay_id"].tolist() == [1, 2, 3]


def test_cache_holds_binned_stays_in_subject_order(monkeypatch, tmp_path):
    stays = source_stays(count=6)
    for i, (_, stay) in enumerate(stays):
        stay["subject_id"] = 100 + (5 - i) // 2
    monkeypatch.setattr(cohort, "_connect", lambda dsn: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cohort, "cohort_subjects", lambda conn: (9, [100, 101, 102]))
    monkeypatch.setattr(cohort, "load_stays", lambda conn, cohort_id, subjects: [
        s for s in stays if s[1]["subject_id"] in subjects])
    cache = data.load_cache(cohort.build_cache("unused", tmp_path / "cache.pt", subjects_per_query=2))
    order = sorted(range(6), key=lambda i: (stays[i][1]["subject_id"], stays[i][1]["intime"], i))
    assert cache["stay_ids"].tolist() == [stays[i][1]["stay_id"] for i in order]
    assert cache["subject_ids"].tolist() == [stays[i][1]["subject_id"] for i in order]
    for row, i in enumerate(order):
        binned = hourly.bin_stay(*stays[i])
        assert torch.equal(cache["continuous"][row], torch.from_numpy(binned["continuous"]))
        assert torch.equal(cache["binary"][row], torch.from_numpy(binned["binary"]))
        assert torch.equal(cache["static"][row], torch.from_numpy(hourly.static_variables(*stays[i])))
    assert torch.equal(cache["time_hours"][0], torch.arange(24, dtype=torch.float32))
    assert cache["continuous"].dtype == torch.float32 and cache["subject_ids"].dtype == torch.int64
    assert np.array_equal(cache["continuous_mask"].shape, (6, 24, 37))


def test_command_line_builds_cache(monkeypatch, tmp_path, capsys):
    from pulse.__main__ import main

    calls = []
    monkeypatch.setattr(cohort, "build_cache", lambda dsn, out: calls.append((dsn, out)) or out)
    assert main(["build-cache", "--dsn", "dbname=x", "--out", str(tmp_path / "c.pt")]) == 0
    assert calls == [("dbname=x", tmp_path / "c.pt")]


def test_package_imports_without_the_database_driver(monkeypatch):
    # A fresh process, so that no earlier import of the driver or the command line counts.
    hide = "import sys; sys.modules['psycopg2'] = None; import pulse.__main__, pulse.cohort"
    subprocess.run([sys.executable, "-c", hide], check=True, cwd=Path(__file__).resolve().parent.parent)
    monkeypatch.setitem(sys.modules, "psycopg2", None)
    importlib.reload(cohort)
    with pytest.raises(ImportError):
        cohort._connect("dbname=x")

import contextlib
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
from pulse import variables as V
from synthetic import source_stays


def test_measurements_follow_items_and_repeat_labs_per_stay(monkeypatch):
    stays = pd.DataFrame({"subject_id": [5, 5, 6], "stay_id": [1, 2, 3], "hadm_id": [10, 10, 11],
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
    monkeypatch.setattr(cohort, "_connect", lambda database: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cohort, "cohort_stays", lambda conn: pd.DataFrame(
        {"subject_id": [s["subject_id"] for _, s in stays], "stay_id": [s["stay_id"] for _, s in stays]}))

    def load_stays(conn, stay_ids, subjects):
        chosen = [s for s in stays if s[1]["subject_id"] in subjects]
        assert sorted(stay_ids) == sorted(s["stay_id"] for _, s in chosen)
        return chosen

    monkeypatch.setattr(cohort, "load_stays", load_stays)
    cache = data.load_cache(cohort.build_cache(Path("unused"), tmp_path / "cache.pt", subjects_per_query=2))
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
    monkeypatch.setattr(cohort, "build_cache", lambda database, out: calls.append((database, out)) or out)
    assert main(["build-cache", "--database", "x.duckdb", "--out", str(tmp_path / "c.pt")]) == 0
    assert calls == [(Path("x.duckdb"), tmp_path / "c.pt")]


def test_package_imports_without_the_database_driver(monkeypatch):
    # A fresh process, so that no earlier import of the driver or the command line counts.
    hide = "import sys; sys.modules['duckdb'] = None; import pulse.__main__, pulse.cohort"
    subprocess.run([sys.executable, "-c", hide], check=True, cwd=Path(__file__).resolve().parent.parent)
    monkeypatch.setitem(sys.modules, "duckdb", None)
    importlib.reload(cohort)
    with pytest.raises(ImportError):
        cohort._connect(Path("x.duckdb"))


TABLES = {
    "mimiciv_derived.sepsis3": "subject_id INTEGER, stay_id INTEGER, sepsis3 BOOLEAN",
    "mimiciv_icu.icustays": "subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER, intime TIMESTAMP, outtime TIMESTAMP",
    "mimiciv_hosp.admissions": "subject_id INTEGER, hadm_id INTEGER, admission_type VARCHAR, edregtime TIMESTAMP",
    "mimiciv_hosp.patients": "subject_id INTEGER, gender VARCHAR, anchor_age SMALLINT, dod DATE",
    "mimiciv_icu.inputevents": "subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER, starttime TIMESTAMP, "
                               "endtime TIMESTAMP, itemid INTEGER, amount DOUBLE, amountuom VARCHAR, rate DOUBLE, "
                               "rateuom VARCHAR, ordercategoryname VARCHAR",
    "mimiciv_icu.chartevents": "subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER, charttime TIMESTAMP, "
                               "itemid INTEGER, valuenum DOUBLE",
    "mimiciv_hosp.labevents": "subject_id INTEGER, hadm_id INTEGER, charttime TIMESTAMP, itemid INTEGER, valuenum DOUBLE",
    "mimiciv_icu.procedureevents": "subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER, starttime TIMESTAMP, "
                                   "endtime TIMESTAMP, itemid INTEGER",
    "mimiciv_icu.outputevents": "subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER, charttime TIMESTAMP, "
                                "itemid INTEGER, value DOUBLE, valueuom VARCHAR",
    "mimiciv_icu.d_items": "itemid INTEGER, label VARCHAR",
}


def test_cohort_query_and_cache_on_a_duckdb_database(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    database = tmp_path / "mimiciv.duckdb"
    con = duckdb.connect(str(database))
    for schema in ("mimiciv_hosp", "mimiciv_icu", "mimiciv_derived"):
        con.execute(f"CREATE SCHEMA {schema}")
    for table, columns in TABLES.items():
        con.execute(f"CREATE TABLE {table} ({columns})")
    intime = pd.Timestamp("2150-01-01 08:00")
    # Stays 100 and 500 qualify; 200 lacks Sepsis-3, 300 an emergency department stay, 400 pre-ICU fluids,
    # and 501, a second stay of subject 5, Sepsis-3.
    stays = [(1, 10, 100, True, True, "16-Pre Admission/Non-ICU"), (2, 20, 200, False, True, "16-Pre Admission/Non-ICU"),
             (3, 30, 300, True, False, "16-Pre Admission/Non-ICU"), (4, 40, 400, True, True, "01-Drips"),
             (5, 50, 500, True, True, "16-Pre Admission/Non-ICU"), (5, 51, 501, False, True, "16-Pre Admission/Non-ICU")]
    for subject, hadm, stay, sepsis, ed, category in stays:
        con.execute("INSERT INTO mimiciv_derived.sepsis3 VALUES (?, ?, ?)", [subject, stay, sepsis])
        con.execute("INSERT INTO mimiciv_icu.icustays VALUES (?, ?, ?, ?, ?)",
                    [subject, hadm, stay, intime, intime + pd.Timedelta(days=3)])
        con.execute("INSERT INTO mimiciv_hosp.admissions VALUES (?, ?, 'EW EMER.', ?)",
                    [subject, hadm, intime - pd.Timedelta(hours=5) if ed else None])
        con.execute("INSERT INTO mimiciv_icu.inputevents VALUES (?, ?, ?, ?, ?, 225158, 500, 'ml', NULL, NULL, ?)",
                    [subject, hadm, stay, intime - pd.Timedelta(hours=3), intime - pd.Timedelta(hours=2), category])
        con.execute("INSERT INTO mimiciv_icu.chartevents VALUES (?, ?, ?, ?, 220045, ?)",
                    [subject, hadm, stay, intime + pd.Timedelta(minutes=30), 60.0 + stay / 10])
    con.execute("INSERT INTO mimiciv_icu.d_items VALUES (225158, 'NaCl 0.9%')")
    con.execute("INSERT INTO mimiciv_hosp.labevents VALUES (1, 10, ?, 51265, 150.0)", [intime + pd.Timedelta(hours=2)])
    for subject, age in ((1, 61), (2, 62), (3, 63), (4, 64), (5, 65)):
        con.execute("INSERT INTO mimiciv_hosp.patients VALUES (?, 'M', ?, NULL)", [subject, age])
    con.close()

    with contextlib.closing(cohort._connect(database)) as conn:
        assert cohort.cohort_stays(conn)["stay_id"].tolist() == [100, 500]
    cache = data.load_cache(cohort.build_cache(database, tmp_path / "cache.pt", subjects_per_query=1))
    assert cache["stay_ids"].tolist() == [100, 500] and cache["subject_ids"].tolist() == [1, 5]
    heart, platelet = V.CONTINUOUS.index("Heart Rate"), V.CONTINUOUS.index("Platelet")
    assert cache["continuous"][:, 0, heart].tolist() == [70.0, 110.0]
    assert cache["continuous_mask"][0, :, platelet].nonzero().flatten().tolist() == [2]
    assert cache["continuous"][0, 2, platelet] == 150.0 and cache["continuous_mask"][1, :, platelet].sum() == 0
    assert cache["static"][:, V.STATIC.index("Age")].tolist() == [61.0, 65.0]

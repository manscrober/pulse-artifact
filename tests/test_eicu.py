"""The eICU transfer cohort: selection, hourly vital signs, events, statics, split and test part.

The eICU files are written here with six made-up stays: four with sepsis and
24 covered hours, one with sepsis and an uncovered hour, and one without
sepsis. They contain no patient data.
"""
import csv
import gzip

import numpy as np
import pytest
import torch

from pulse import data, eicu
from pulse import variables as V
from synthetic import part as synthetic_part
from synthetic import protocol

HOURLY = range(24)


def write(directory, table, header, rows):
    with gzip.open(directory / f"{table}.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def eicu_files(directory):
    write(directory, "patient", ["patientunitstayid", "gender", "age", "apacheadmissiondx", "admissionweight",
                                 "dischargeweight", "unitdischargeoffset", "unitdischargestatus"], [
        ["105", "Male", "> 89", "Sepsis, pulmonary", "80.5", "79", "2500", "Alive"],
        ["1002", "Female", "45", "", "", "61.2", "1400", "Expired"],
        ["33", "Male", "70", "Overdose", "NULL", "75", "3000", "Alive"],
        ["17", "Female", "50", "", "70", "", "1500", "Alive"],
        ["200", "Male", "60", "Chest pain", "90", "", "1500", "Alive"],
        ["4", "", "", "Sepsis, renal/UTI", "", "", "1430", "Alive"]])
    write(directory, "diagnosis", ["diagnosisid", "patientunitstayid", "diagnosisstring", "icd9code"], [
        [1, "1002", "infectious diseases|systemic/other infections|septic shock", ""],
        [2, "1002", "cardiovascular|ventricular disorders|congestive heart failure", "428.0, I50.9"],
        [3, "33", "infectious diseases|unspecified", "038.9, A41.9"],
        [4, "33", "pulmonary|pulmonary edema", ""],
        [5, "33", "renal|disorder of kidney|ESRD (end stage renal disease)", ""],
        [6, "17", "infectious diseases", "995.92, R65.20"],
        [7, "200", "cardiovascular|chest pain", ""]])
    periodic = []
    for h in HOURLY:
        periodic += [[h * 60 + 5, "105", 37.0, 97, 80.1, 18, 120, 60, 80],
                     [h * 60 + 35, "105", 37.2, 98, 80.2, 20, 124, 64, 84],
                     [h * 60 + 10, "1002", "", "", 90, "", "", "", ""],
                     [h * 60 + 10, "33", "", "", 100, "", "", "", ""],
                     [h * 60 + 10, "200", "", "", 75, "", "", "", ""]]
        if h < 23:
            periodic.append([h * 60 + 10, "17", "", "", 70, "", "", "", ""])
    periodic += [[190, "1002", 98.6, "", "", "", "", "", ""],            # Fahrenheit, hour 3
                 [-5, "105", 40, 50, 200, 40, 200, 100, 150],              # before admission
                 [1440, "105", 40, 50, 200, 40, 200, 100, 150],            # after the first day
                 [30, "4", "", "", "", "", "", "", ""]]                    # an empty reading
    write(directory, "vitalPeriodic", ["patientunitstayid", "observationoffset", "temperature", "sao2", "heartrate",
                                       "respiration", "systemicsystolic", "systemicdiastolic", "systemicmean"],
          [[row[1], row[0], *row[2:]] for row in periodic])
    aperiodic = [[h * 60 + 20, "1002", 110, 70, 83] for h in HOURLY] + [[h * 60 + 20, "4", 130, 80, 97] for h in HOURLY]
    aperiodic.append([20, "105", 999, 999, 999])  # an invasive reading in the same hour wins
    write(directory, "vitalAperiodic", ["patientunitstayid", "observationoffset", "noninvasivesystolic",
                                        "noninvasivediastolic", "noninvasivemean"],
          [[row[1], row[0], *row[2:]] for row in aperiodic])
    write(directory, "medication", ["patientunitstayid", "drugorderoffset", "drugstartoffset", "drugname",
                                    "drugstopoffset"], [
        ["33", "100", "120", "norepinephrine 4 mg/250 mL", "300"],    # hours 2 to 5
        ["33", "60", "-30", "Furosemide", ""],                         # start before admission: order hour 1
        ["33", "", "3000", "Vasopressin", "3600"],                     # both outside: hour 0
        ["105", "10", "10", "acetaminophen", "100"]])
    write(directory, "treatment", ["patientunitstayid", "treatmentoffset", "treatmentstring"], [
        ["33", "400", "pulmonary|ventilation and oxygenation|mechanical ventilation"],
        ["33", "2000", "renal|dialysis|hemodialysis"],
        ["33", "700", "cardiovascular|shock|vasopressors|norepinephrine > 0.1 micrograms/kg/min"]])


def test_build_selects_covered_sepsis_stays_and_bins_them(tmp_path):
    pytest.importorskip("duckdb")
    eicu_files(tmp_path)
    counts = eicu.build_cache(tmp_path, tmp_path / "eicu.pt")
    assert counts == {"stays": 6, "sepsis_stays": 5, "covered_stays": 4, "test_stays": 1,
                      "cache": str(tmp_path / "eicu.pt")}
    cache = eicu.load_cache(tmp_path / "eicu.pt")
    assert cache["stay_ids"].tolist() == [1002, 105, 33, 4]  # identifiers sorted as text
    x, mask = cache["continuous"], cache["continuous_mask"]
    assert torch.equal(mask.bool(), torch.isfinite(x))
    column = {name: V.CONTINUOUS.index(name) for name in V.CONTINUOUS}
    recorded = {"Heart Rate", "Systolic Blood Pressure", "Diastolic Blood Pressure", "Mean Blood Pressure",
                "Temperature", "Respiratory Rate", "spO2", "Weight"}
    assert not mask[..., [column[n] for n in V.CONTINUOUS if n not in recorded]].any()

    f32 = np.float32
    assert torch.equal(x[0, :, column["Heart Rate"]], torch.full((24,), 90.0))
    assert x[1, 0, column["Heart Rate"]].item() == (f32(80.1) + f32(80.2)) / f32(2)
    assert torch.equal(x[1, :, column["Systolic Blood Pressure"]], torch.full((24,), 122.0))
    assert torch.equal(x[0, :, column["Systolic Blood Pressure"]], torch.full((24,), 110.0))
    assert torch.equal(x[3, :, column["Mean Blood Pressure"]], torch.full((24,), 97.0))
    assert x[0, 3, column["Temperature"]].item() == f32((98.6 - 32.0) * (5.0 / 9.0))
    assert mask[0, :, column["Temperature"]].tolist() == [h == 3 for h in HOURLY]
    assert not mask[3, :, column["Heart Rate"]].any()
    assert torch.allclose(x[:, 0, column["Weight"]], torch.tensor([61.2, 80.5, float("nan"), float("nan")]),
                          equal_nan=True)

    b, channel = cache["binary"], {name: V.BINARY.index(name) for name in V.BINARY}
    assert b[0, :, V.DEATH].tolist() == [h >= 23 for h in HOURLY] and not b[0, :, V.RELEASE].any()
    assert b[3, :, V.RELEASE].tolist() == [h >= 23 for h in HOURLY]
    assert not b[1].any()
    assert b[2, :, channel["Vasopressor Indicator"]].tolist() == [h in (0, 2, 3, 4, 5) or h >= 11 for h in HOURLY]
    assert b[2, :, channel["Diuretics Indicator"]].tolist() == [h == 1 for h in HOURLY]
    assert b[2, :, channel["Mechanical Ventilation Indicator"]].tolist() == [h >= 6 for h in HOURLY]
    assert not b[2, :, channel["Dialysis Indicator"]].any()
    assert b[2, :, channel["Pulmonary Edema Indicator"]].all() and not b[[0, 1, 3], :, 1].any()

    nan = float("nan")
    assert torch.allclose(cache["static"], torch.tensor([[45, nan, nan, 0, 0, 1], [89, nan, nan, 1, 0, 0],
                                                         [70, nan, nan, 1, 1, 0], [nan, nan, nan, 0, 0, 0]]),
                          equal_nan=True)
    assert torch.equal(cache["time_hours"], torch.arange(24).float().expand(4, -1))


def test_split_of_the_published_cohort_and_its_first_test_stays():
    stays = {"stay_ids": torch.arange(21659)}
    order = np.random.default_rng(42).permutation(21659)
    rows = eicu.test_rows(stays)
    assert eicu._sizes(21659) == (15161, 2166) and len(rows) == 1000
    assert np.array_equal(rows, order[15161 + 2166:][:1000])
    assert len(eicu.test_rows(stays, limit=10**6)) == 4332


def _mapped_cache(n: int = 30):
    g = torch.Generator().manual_seed(3)
    x = torch.full((n, 24, len(V.CONTINUOUS)), float("nan"))
    x[..., :6] = 50 + 20 * torch.rand(n, 24, 6, generator=g)
    x[torch.rand(n, 24, len(V.CONTINUOUS), generator=g) < 0.3] = float("nan")
    x[..., V.CONTINUOUS.index("Weight")] = (70 + 20 * torch.rand(n, 1, generator=g)).expand(n, 24)
    static = torch.tensor([[60.0, float("nan"), float("nan"), 1.0, 1.0, 1.0]]).repeat(n, 1)
    return {"continuous": x, "continuous_mask": torch.isfinite(x).float(),
            "binary": (torch.rand(n, 24, len(V.BINARY), generator=g) < 0.1).float(),
            "time_hours": torch.arange(24).float().expand(n, -1), "static": static, "stay_ids": torch.arange(n),
            "sha256": "e" * 64}


def test_training_part_marks_what_eicu_records_and_cotraining_appends_it():
    cache = _mapped_cache()
    p = protocol()
    rows = eicu.training_rows(cache)
    assert np.array_equal(rows, np.random.default_rng(42).permutation(30)[:eicu._sizes(30)[0]])
    part = eicu.training_part(cache, p)
    w = V.CONTINUOUS.index("Weight")
    others = [v for v in range(len(V.CONTINUOUS)) if v != w]
    recorded = torch.isfinite(cache["continuous"][rows])
    assert torch.equal(part["continuous_mask"][..., others], recorded[..., others].float())
    # The weight copied into every hour counts once per stay, in its first hour; the inputs carry it forward.
    assert recorded[..., w].all() and part["continuous_mask"][..., w].sum(1).eq(1).all()
    assert part["continuous_mask"][:, 0, w].eq(1).all()
    assert part["continuous_target"][:, 1:, w].eq(p.mean[w]).all()
    assert torch.equal(part["continuous"][..., w], cache["continuous"][rows, :, w])
    assert eicu.test_part(cache, p)["continuous_mask"][..., w].eq(1).all()
    vitals = sorted(V.CONTINUOUS.index(name) for name in eicu.PERIODIC)
    assert part["observable"].sum(0).nonzero().flatten().tolist() == vitals
    assert part["observable"][:, vitals].eq(1).all()
    assert part["outcomes_known"].eq(0).nonzero()[:, 1].unique().tolist() == [V.BINARY.index("Cumulative Edema")]
    mimic = data.Prepared(train=synthetic_part(seed=7, n=5), val=synthetic_part(seed=8, n=2),
                          test=synthetic_part(seed=9, n=1), protocol=p)
    joined = eicu.cotrain(mimic, cache)
    assert joined.sources == ("e" * 64,) and joined.val is mimic.val and joined.test is mimic.test
    assert torch.equal(joined.train["continuous"][:5], mimic.train["continuous"])
    assert torch.equal(joined.train["continuous"][5:], part["continuous"])
    assert joined.train["observable"][:5].eq(1).all() and joined.train["outcomes_known"][:5].eq(1).all()
    assert torch.equal(joined.train["observable"][5:], part["observable"])


def test_test_part_fills_and_scales_with_the_mimic_statistics():
    g = torch.Generator().manual_seed(3)
    n = 30
    x = torch.full((n, 24, len(V.CONTINUOUS)), float("nan"))
    x[..., :6] = 50 + 20 * torch.rand(n, 24, 6, generator=g)
    x[torch.rand(n, 24, len(V.CONTINUOUS), generator=g) < 0.3] = float("nan")
    static = torch.tensor([[60.0, float("nan"), float("nan"), 1.0, 1.0, 1.0]]).repeat(n, 1)
    static[::2, 0] = float("nan")
    cache = {"continuous": x, "continuous_mask": torch.isfinite(x).float(),
             "binary": (torch.rand(n, 24, len(V.BINARY), generator=g) < 0.1).float(),
             "time_hours": torch.arange(24).float().expand(n, -1), "static": static, "stay_ids": torch.arange(n)}
    p = protocol()
    p = data.Protocol(**{**p.__dict__, "static_mean": [63.0, 200.0, 0.0, 0.0, 0.0, 0.0],
                         "static_sd": [17.0, 900.0, 0.01, 1.0, 1.0, 1.0]})
    part = eicu.test_part(cache, p, limit=4)
    rows = eicu.test_rows(cache, limit=4)
    mean = torch.tensor(p.mean)
    keep = torch.isfinite(x[rows])
    filled = torch.where(keep, x[rows], mean)
    for t in range(1, 24):
        filled[:, t] = torch.where(keep[:, t], filled[:, t], filled[:, t - 1])
    assert torch.equal(part["continuous"], filled)
    assert torch.equal(part["continuous_target"], torch.where(keep, x[rows], mean))
    assert torch.equal(part["continuous_mask"], keep.float())
    expected = torch.zeros(4, 6)
    expected[:, 0] = torch.where(torch.isnan(static[rows, 0]), 0.0, (static[rows, 0] - 63.0) / 17.0)
    expected[:, 3] = 1.0
    assert torch.equal(part["static"], expected)
    assert torch.equal(part["binary"], cache["binary"][rows])

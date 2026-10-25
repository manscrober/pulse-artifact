"""eICU transfer: the MIMIC-trained models rolled out on eICU sepsis stays without refitting.

`build_cache` reads the CSV files of the eICU Collaborative Research
Database v2.0 and keeps the sepsis stays whose first 24 hours each hold a
vital sign. `test_part` maps the first `TRANSFER_STAYS` of their test stays
onto the MIMIC variables, filled and scaled with the training statistics of
the MIMIC protocol as the MIMIC test patients are; `python -m pulse rollout
--eicu` rolls them out and `python -m pulse score --eicu` scores them.
`cotrain` adds the eICU training stays, mapped the same way, to the MIMIC
training part (`python -m pulse train --cotrain-eicu`).

A stay has sepsis when its APACHE admission diagnosis or one of its
diagnosis strings contains a sepsis term, or a diagnosis code, stripped of
everything but letters and digits, starts with a sepsis prefix. Hour h holds
the offsets of 60h to 60h + 59 minutes from unit admission. A vital sign
takes the mean of its readings in the periodic record within the hour; blood
pressure takes the non-invasive readings of the aperiodic record in hours
without an invasive one, and temperatures above 70 are read as Fahrenheit.
The admission weight, else the discharge weight, is recorded in every hour.
eICU records none of the other MIMIC variables, so they are missing in every
hour and never scored.

Binary events: death or release from the hour of unit discharge; ventilation,
dialysis and vasopressors recorded as treatments from their hour on;
vasopressors and diuretics given as medication from their start to their
stop hour, where a start outside the first 24 hours is replaced by the order
hour, else hour 0, and a stop outside them by the start; pulmonary oedema
from the diagnoses, in every hour. Static variables: age in years, an age
above 89 counting as 89; sex; end-stage renal failure and congestive heart
failure from the diagnoses. An unknown age, the pre-ICU fluid amount and the
Elixhauser score, which eICU does not record, are stored as NaN.

The stays are sorted by their identifier as text and split 70/10/20 by a
permutation with seed 42, as in the published transfer. The DuckDB driver
(the `cohort` extra) reads the files with one thread, so rows arrive in file
order, and hourly sums accumulate in float32 in that order, as in the
published cohort.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pulse import data
from pulse import variables as V

TRANSFER_STAYS = 1000
SPLIT = (0.7, 0.1)  # training and validation fractions; the rest is test
SPLIT_SEED = 42
HOURS = 24
TENSORS = ("continuous", "continuous_mask", "binary", "time_hours", "static", "stay_ids")

SEPSIS_TERMS = ("sepsis", "septic", "septic shock", "severe sepsis", "urosepsis", "bacteremia", "pneumosepsis",
                "abd sepsis", "abdominal sepsis")
SEPSIS_CODES = ("038", "78552", "99591", "99592", "R6520", "R6521", "A40", "A41")
HEART_FAILURE = (("congestive heart failure", "chf", "heart failure"), ("428", "I50"))
RENAL_FAILURE = (("end stage renal", "esrd", "esrf"), ("5856", "N186"))
OEDEMA_TERMS = ("pulmonary edema", "pulmonary oedema")
VASOPRESSORS = ("norepinephrine", "levophed", "epinephrine", "phenylephrine", "vasopressin", "dopamine",
                "dobutamine", "milrinone", "vasopressor")
DIURETICS = ("furosemide", "lasix", "bumetanide", "torsemide", "metolazone", "diuretic")
VENTILATION = ("ventilation", "ventilator", "intubation", "mechanical vent", "ett", "trach")
DIALYSIS = ("dialysis", "cvvh", "crrt", "hemodialysis", "haemodialysis")
# Vital signs and their columns in vitalPeriodic; blood pressure also in vitalAperiodic.
PERIODIC = {"Heart Rate": "heartrate", "Systolic Blood Pressure": "systemicsystolic",
            "Diastolic Blood Pressure": "systemicdiastolic", "Mean Blood Pressure": "systemicmean",
            "Temperature": "temperature", "Respiratory Rate": "respiration", "spO2": "sao2"}
APERIODIC = {"Systolic Blood Pressure": "noninvasivesystolic", "Diastolic Blood Pressure": "noninvasivediastolic",
             "Mean Blood Pressure": "noninvasivemean"}
# Binary outcomes the cohort never records.
UNRECORDED_OUTCOMES = ("Cumulative Edema",)
# Variables the cohort writes into every hour from one value per stay; co-training counts the first hour only.
ONCE_PER_STAY = ("Weight",)


def _connect():
    import duckdb

    conn = duckdb.connect()
    conn.execute("SET threads = 1")
    return conn


def _csv(conn, directory: Path, table: str, columns: str, where: str = "") -> pd.DataFrame:
    """Columns of one eICU table, read as text; `where` may name the registered stays as `wanted`."""
    path = Path(directory) / f"{table}.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing")
    literal = "'" + str(path).replace("'", "''") + "'"
    return conn.execute(f"SELECT {columns} FROM read_csv({literal}, all_varchar = true, header = true) {where}").df()


def _number(column: str) -> str:
    return f"TRY_CAST({column} AS DOUBLE) AS {column}"


def _hours(minutes: pd.Series) -> np.ndarray:
    """Hour of each offset in minutes from unit admission; -1 when missing or outside the first 24 hours."""
    minutes = minutes.to_numpy(float)
    with np.errstate(invalid="ignore"):
        inside = (minutes >= 0) & (minutes < 60 * HOURS)
    return np.where(inside, np.floor_divide(np.nan_to_num(minutes), 60), -1).astype(np.int64)


def _contains(texts: pd.Series, terms: tuple[str, ...]) -> np.ndarray:
    pattern = re.compile("|".join(map(re.escape, terms)))
    return np.array([isinstance(t, str) and pattern.search(t.lower()) is not None for t in texts], dtype=bool)


def _coded(codes: pd.Series, prefixes: tuple[str, ...]) -> np.ndarray:
    return np.array([re.sub(r"[^A-Za-z0-9]", "", (c or "").upper()).startswith(prefixes) for c in codes], dtype=bool)


def _age(text: str | None) -> float:
    text = (text or "").strip()
    if text.startswith(">"):
        return float("".join(ch for ch in text if ch.isdigit()) or 90)
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _hourly(conn, directory: Path, table: str, columns: dict[str, str], stays: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Sums and counts [stays, 24, columns] of the readings of each column per stay and hour."""
    frame = _csv(conn, directory, table,
                 f"patientunitstayid AS stay, {_number('observationoffset')}, "
                 + ", ".join(_number(column) for column in columns.values()),
                 "WHERE patientunitstayid IN (SELECT stay FROM wanted) "
                 f"AND TRY_CAST(observationoffset AS DOUBLE) >= 0 AND TRY_CAST(observationoffset AS DOUBLE) < {60 * HOURS}")
    hour, row = _hours(frame["observationoffset"]), stays.get_indexer(frame["stay"])
    sums = np.zeros((len(stays), HOURS, len(columns)), np.float32)
    counts = np.zeros((len(stays), HOURS, len(columns)), np.int64)
    for j, (name, column) in enumerate(columns.items()):
        x = frame[column].to_numpy(float)
        if name == "Temperature":
            x = np.where(x > 70.0, (x - 32.0) * (5.0 / 9.0), x)
        ok = ~np.isnan(x)
        # np.add.at adds in the order of its indices, here file order, in float32.
        np.add.at(sums, (row[ok], hour[ok], j), x[ok].astype(np.float32))
        np.add.at(counts, (row[ok], hour[ok], j), 1)
    return sums, counts


def _patients(conn, directory: Path) -> pd.DataFrame:
    """Every unit stay with the columns read and its sepsis and diagnosis flags, indexed by identifier."""
    patients = _csv(conn, directory, "patient", "patientunitstayid AS stay, age, gender, admissionweight, "
                    f"dischargeweight, {_number('unitdischargeoffset')}, unitdischargestatus, apacheadmissiondx")
    diagnoses = _csv(conn, directory, "diagnosis", "patientunitstayid AS stay, diagnosisstring, icd9code")
    text, code = diagnoses["diagnosisstring"], diagnoses["icd9code"]
    flags = pd.DataFrame({
        "stay": diagnoses["stay"],
        "sepsis": _contains(text, SEPSIS_TERMS) | _coded(code, SEPSIS_CODES),
        "heart_failure": _contains(text, HEART_FAILURE[0]) | _coded(code, HEART_FAILURE[1]),
        "renal_failure": _contains(text, RENAL_FAILURE[0]) | _coded(code, RENAL_FAILURE[1]),
        "oedema": _contains(text, OEDEMA_TERMS)}).groupby("stay").any().reindex(patients["stay"], fill_value=False)
    flags["sepsis"] |= _contains(patients["apacheadmissiondx"], SEPSIS_TERMS)
    return patients.set_index("stay").join(flags)


def _static(patients: pd.DataFrame) -> np.ndarray:
    """Static variables [stays, 6] in physical units, NaN where unknown."""
    static = np.full((len(patients), len(V.STATIC)), np.nan, np.float32)
    static[:, V.STATIC.index("Age")] = [_age(a) for a in patients["age"]]
    static[:, V.STATIC.index("Gender")] = [str(g or "").strip().lower().startswith("m") for g in patients["gender"]]
    static[:, V.STATIC.index("End Stage Renal Failure")] = patients["renal_failure"].to_numpy(bool)
    static[:, V.STATIC.index("Congestive Heart Failure")] = patients["heart_failure"].to_numpy(bool)
    return static


def _events(conn, directory: Path, patients: pd.DataFrame) -> np.ndarray:
    """Binary events [stays, 24, 8] of the stays of `patients`, in its order."""
    stays = patients.index
    binary = np.zeros((len(stays), HOURS, len(V.BINARY)), np.float32)
    channel = {name: V.BINARY.index(name) for name in V.BINARY}
    binary[patients["oedema"].to_numpy(bool), :, channel["Pulmonary Edema Indicator"]] = 1.0
    discharge = _hours(patients["unitdischargeoffset"])
    expired = _contains(patients["unitdischargestatus"], ("expired",))
    for i in np.flatnonzero(discharge >= 0):
        binary[i, discharge[i]:, V.DEATH if expired[i] else V.RELEASE] = 1.0

    conn.register("wanted", pd.DataFrame({"stay": stays}))
    drugs = _csv(conn, directory, "medication", "patientunitstayid AS stay, drugname, "
                 + ", ".join(map(_number, ("drugstartoffset", "drugstopoffset", "drugorderoffset"))),
                 "WHERE patientunitstayid IN (SELECT stay FROM wanted)")
    start, stop, order = (_hours(drugs[c]) for c in ("drugstartoffset", "drugstopoffset", "drugorderoffset"))
    start = np.where(start >= 0, start, np.maximum(order, 0))
    stop = np.where(stop >= 0, np.maximum(stop, start), start)
    rows = stays.get_indexer(drugs["stay"])
    for name, terms in (("Vasopressor Indicator", VASOPRESSORS), ("Diuretics Indicator", DIURETICS)):
        for i in np.flatnonzero(_contains(drugs["drugname"], terms)):
            binary[rows[i], start[i]:stop[i] + 1, channel[name]] = 1.0

    treatments = _csv(conn, directory, "treatment", f"patientunitstayid AS stay, treatmentstring, "
                      f"{_number('treatmentoffset')}", "WHERE patientunitstayid IN (SELECT stay FROM wanted)")
    hour, rows = _hours(treatments["treatmentoffset"]), stays.get_indexer(treatments["stay"])
    for name, terms in (("Mechanical Ventilation Indicator", VENTILATION), ("Dialysis Indicator", DIALYSIS),
                        ("Vasopressor Indicator", ("vasopressor",))):
        for i in np.flatnonzero(_contains(treatments["treatmentstring"], terms) & (hour >= 0)):
            binary[rows[i], hour[i]:, channel[name]] = 1.0
    return binary


def build_cache(eicu_dir: Path, out: Path) -> dict:
    """Select the transfer cohort from the eICU CSV files in `eicu_dir`, write it to `out` and return counts."""
    conn = _connect()
    try:
        patients = _patients(conn, eicu_dir)
        candidates = pd.Index(sorted(patients.index[patients["sepsis"]]))
        conn.register("wanted", pd.DataFrame({"stay": candidates}))
        periodic, periodic_counts = _hourly(conn, eicu_dir, "vitalPeriodic", PERIODIC, candidates)
        aperiodic, aperiodic_counts = _hourly(conn, eicu_dir, "vitalAperiodic", APERIODIC, candidates)
        kept = np.flatnonzero(((periodic_counts > 0).any(-1) | (aperiodic_counts > 0).any(-1)).all(1))
        selected = patients.loc[candidates[kept]]

        continuous = np.full((len(kept), HOURS, len(V.CONTINUOUS)), np.nan, np.float32)
        counts, fallback_counts = periodic_counts[kept], aperiodic_counts[kept]
        with np.errstate(invalid="ignore", divide="ignore"):
            means = periodic[kept] / counts.astype(np.float32)
            fallback = aperiodic[kept] / fallback_counts.astype(np.float32)
        for j, name in enumerate(PERIODIC):
            value = np.where(counts[..., j] > 0, means[..., j], np.nan)
            if name in APERIODIC:
                a = list(APERIODIC).index(name)
                value = np.where((counts[..., j] == 0) & (fallback_counts[..., a] > 0), fallback[..., a], value)
            continuous[..., V.CONTINUOUS.index(name)] = value
        weight = [a if a not in (None, "") else d for a, d in zip(selected["admissionweight"], selected["dischargeweight"])]
        continuous[..., V.CONTINUOUS.index("Weight")] = pd.to_numeric(
            pd.Series(weight, dtype=object), errors="coerce").to_numpy(np.float32)[:, None]
        binary = _events(conn, eicu_dir, selected)
    finally:
        conn.close()
    cache = {"continuous": torch.from_numpy(continuous),
             "continuous_mask": torch.from_numpy(np.isfinite(continuous)).float(),
             "binary": torch.from_numpy(binary), "time_hours": torch.arange(HOURS).float().expand(len(kept), -1).clone(),
             "static": torch.from_numpy(_static(selected)), "stay_ids": torch.tensor([int(s) for s in selected.index])}
    save_cache(cache, out)
    return {"stays": len(patients), "sepsis_stays": len(candidates), "covered_stays": len(kept),
            "test_stays": len(kept) - sum(_sizes(len(kept))), "cache": str(out)}


def save_cache(cache: dict, path: Path) -> Path:
    return data.save_cache(cache, path, keys=TENSORS)


def load_cache(path: Path) -> dict:
    return data.load_cache(path, keys=TENSORS)


def _sizes(n: int) -> tuple[int, int]:
    return int(round(SPLIT[0] * n)), int(round(SPLIT[1] * n))


def test_rows(cache: dict, limit: int = TRANSFER_STAYS) -> np.ndarray:
    """Cache rows of the first `limit` test stays, in the order of the published transfer."""
    n = len(cache["stay_ids"])
    return np.random.default_rng(SPLIT_SEED).permutation(n)[sum(_sizes(n)):][:limit]


def training_rows(cache: dict) -> np.ndarray:
    """Cache rows of the training stays of the split."""
    n = len(cache["stay_ids"])
    return np.random.default_rng(SPLIT_SEED).permutation(n)[:_sizes(n)[0]]


def test_part(cache: dict, protocol: data.Protocol, limit: int = TRANSFER_STAYS) -> data.Part:
    """The first `limit` test stays as a part filled and scaled with the protocol's MIMIC training statistics.

    Unknown static variables take the training mean. The MIMIC cohort cache
    records no end-stage renal failure and no congestive heart failure, so
    the models never saw either; both are 0, as in the published transfer.
    """
    return _mapped_part(cache, protocol, test_rows(cache, limit))


def training_part(cache: dict, protocol: data.Protocol) -> data.Part:
    """The training stays mapped as `test_part` maps the test stays, with what eICU records.

    `observable` is one for the vital signs of `PERIODIC` and `outcomes_known`
    is one except for `UNRECORDED_OUTCOMES`, so the training loss leaves the
    observation and timing terms of every other variable, and the outcome
    terms of those outcomes, out (`pulse.train.pulse_loss`). The weight is
    a value in every hour but no hourly record: the cohort copies one weight
    of the stay into each hour. Each variable of `ONCE_PER_STAY` therefore
    counts as recorded in its first recorded hour only, so the likelihood
    sees one weight per stay; the inputs still carry it forward. MIMIC
    records the weight in 2% of training rows.
    """
    part = _mapped_part(cache, protocol, training_rows(cache))
    mask, target = part["continuous_mask"].clone(), part["continuous_target"].clone()
    for name in ONCE_PER_STAY:
        v = V.CONTINUOUS.index(name)
        first = (mask[:, :, v] > 0) & (mask[:, :, v].cumsum(1) == 1)
        mask[:, :, v] = first.float()
        target[:, :, v] = torch.where(first, target[:, :, v], protocol.mean[v])
    part = {**part, "continuous_mask": mask, "continuous_target": target}
    n = len(part["continuous"])
    observable = torch.zeros(n, len(V.CONTINUOUS))
    observable[:, [V.CONTINUOUS.index(name) for name in PERIODIC]] = 1.0
    known = torch.ones(n, len(V.BINARY))
    known[:, [V.BINARY.index(name) for name in UNRECORDED_OUTCOMES]] = 0.0
    return {**part, "observable": observable, "outcomes_known": known}


def cotrain(prepared: data.Prepared, cache: dict) -> data.Prepared:
    """`prepared` with the eICU training stays of `cache` after its MIMIC training patients.

    Validation and test parts stay MIMIC's, so the best epoch is the one
    that forecasts the MIMIC validation patients best, and the statistics
    stay the protocol's. The eICU cache digest joins `prepared.sources`.
    """
    train = data.concatenate([prepared.train, training_part(cache, prepared.protocol)])
    return dataclasses.replace(prepared, train=train, sources=prepared.sources + (cache["sha256"],))


def _mapped_part(cache: dict, protocol: data.Protocol, rows: np.ndarray) -> data.Part:
    """The stays of `rows` as a part filled and scaled with the protocol's MIMIC training statistics."""
    x = cache["continuous"].float()
    mask = cache["continuous_mask"].bool() & torch.isfinite(x)
    static_mean, static_sd = torch.tensor(protocol.static_mean), torch.tensor(protocol.static_sd)
    static = cache["static"].float()
    static = torch.where(torch.isnan(static), static_mean, static)
    static[:, [V.STATIC.index("End Stage Renal Failure"), V.STATIC.index("Congestive Heart Failure")]] = 0.0
    return data._part(cache, rows, x, mask, torch.tensor(protocol.mean), (static - static_mean) / static_sd)

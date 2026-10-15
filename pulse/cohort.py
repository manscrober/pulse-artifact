"""Read the sepsis cohort from a MIMIC-IV PostgreSQL database and write the cache.

The cohort is created by the scripts in `sql/` and stored under the name
`COHORT`. Subjects are read 25 at a time, in subject order, with one query
per source table; each stay is binned by `pulse.hourly`, and the cache holds
the stays sorted by subject, ICU admission time and stay.

The queries are those that built the published cache, including columns
this module does not read. The order of rows with equal sort keys is
unspecified in SQL, so `pulse.hourly` sorts every table by a total key
before binning, and the cache depends only on the records. The published
cache was binned in the order the database returned. The sort reorders
records within the same hour, whether their times are equal or not, so a
rebuild can differ from the published cache in the last bit of hourly sums
and means.

Only this module needs the PostgreSQL driver, installed with the `cohort`
extra; it is imported when a connection is opened.
"""
from __future__ import annotations

import contextlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pulse import data, hourly
from pulse import variables as V

COHORT = "GT-Sepsis-Paper"
SUBJECTS_PER_QUERY = 25

# Measured variables with their MIMIC-IV items, in query order.
ITEMS = (
    ("Heart Rate", "chartevents", (220045,)),
    ("Diastolic Blood Pressure", "chartevents", (220051, 220180)),
    ("Systolic Blood Pressure", "chartevents", (220050, 220179)),
    ("Mean Blood Pressure", "chartevents", (220052, 220181)),
    ("Temperature", "chartevents", (223762, 223761)),
    ("Respiratory Rate", "chartevents", (220210,)),
    ("Platelet", "labevents", (51265,)),
    ("Hemoglobin", "labevents", (51222,)),
    ("Calcium", "labevents", (50893,)),
    ("BUN", "labevents", (51006,)),
    ("Creatinine", "labevents", (50912,)),
    ("Bicarbonate", "labevents", (50882,)),
    ("Lactate", "labevents", (50813,)),
    ("pO2", "labevents", (50821,)),
    ("sO2", "labevents", (50817,)),
    ("spO2", "chartevents", (220277,)),
    ("pCO2", "labevents", (50818,)),
    ("Total CO2", "labevents", (50804,)),
    ("pH", "labevents", (50820,)),
    ("Base excess", "labevents", (50802,)),
    ("Weight", "chartevents", (224639, 226512)),
)

STAYS = """
SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime,
    EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 as los_hours,
    a.admission_type, a.admission_location, a.discharge_location, a.insurance, a.language,
    a.marital_status, a.race, a.admittime, a.dischtime, a.edregtime, a.edouttime
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.admissions a USING (subject_id, hadm_id)
JOIN public.cohort c
  ON c.subject_id = i.subject_id AND c.hadm_id = i.hadm_id AND c.stay_id = i.stay_id AND c.cohort_id = %s
WHERE i.subject_id = ANY(%s)
ORDER BY i.subject_id, i.intime
"""
PATIENTS = """
SELECT subject_id, gender, anchor_age, anchor_year, dod
FROM mimiciv_hosp.patients
WHERE subject_id = ANY(%s)
ORDER BY subject_id
"""
CHARTEVENTS = """
SELECT subject_id, hadm_id, stay_id, charttime, storetime, itemid, value, valuenum, valueuom
FROM mimiciv_icu.chartevents
WHERE stay_id IN ({stays}) AND itemid IN ({items}) AND valuenum IS NOT NULL
"""
LABEVENTS = """
SELECT subject_id, hadm_id, charttime, storetime, itemid, value, valuenum, valueuom
FROM mimiciv_hosp.labevents
WHERE hadm_id IN ({admissions}) AND itemid IN ({items}) AND valuenum IS NOT NULL
"""
INPUTEVENTS = """
SELECT ie.subject_id, ie.hadm_id, ie.stay_id, ie.starttime, ie.endtime, ie.storetime, ie.itemid,
    ie.amount, ie.amountuom, ie.rate, ie.rateuom, di.label, di.category,
    EXTRACT(EPOCH FROM (ie.starttime - i.intime))/3600 as hours_since_icu_admit
FROM mimiciv_icu.inputevents ie
JOIN mimiciv_icu.icustays i USING (subject_id, hadm_id, stay_id)
JOIN mimiciv_icu.d_items di ON ie.itemid = di.itemid
WHERE ie.stay_id = ANY(%s)
ORDER BY ie.subject_id, ie.stay_id, ie.starttime
"""
PROCEDUREEVENTS = """
SELECT pe.subject_id, pe.hadm_id, pe.stay_id, pe.starttime, pe.endtime, pe.storetime, pe.itemid,
    pe.value, pe.valueuom, di.label, di.category,
    EXTRACT(EPOCH FROM (pe.starttime - i.intime))/3600 as hours_since_icu_admit
FROM mimiciv_icu.procedureevents pe
JOIN mimiciv_icu.icustays i USING (subject_id, hadm_id, stay_id)
JOIN mimiciv_icu.d_items di ON pe.itemid = di.itemid
WHERE pe.stay_id = ANY(%s)
ORDER BY pe.subject_id, pe.stay_id, pe.starttime
"""
OUTPUTEVENTS = """
SELECT oe.subject_id, oe.hadm_id, oe.stay_id, oe.charttime, oe.storetime, oe.itemid, oe.value,
    oe.valueuom, di.label, di.category,
    EXTRACT(EPOCH FROM (oe.charttime - i.intime))/3600 as hours_since_icu_admit
FROM mimiciv_icu.outputevents oe
JOIN mimiciv_icu.icustays i USING (subject_id, hadm_id, stay_id)
JOIN mimiciv_icu.d_items di ON oe.itemid = di.itemid
WHERE oe.stay_id = ANY(%s)
ORDER BY oe.subject_id, oe.stay_id, oe.charttime
"""


def _connect(dsn: str):
    """An open PostgreSQL connection; tests replace this function."""
    import psycopg2

    return psycopg2.connect(dsn)


def _read(conn, query: str, params=None) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*SQLAlchemy connectable.*")
        return pd.read_sql_query(query, conn, params=params)


def cohort_subjects(conn) -> tuple[int, list[int]]:
    """The cohort's identifier and its subjects in ascending order."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT cohort_id FROM public.d_cohorts WHERE cohort_name = %s", (COHORT,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"cohort {COHORT!r} not found; run the scripts in sql/ first")
        cursor.execute("SELECT DISTINCT subject_id FROM public.cohort WHERE cohort_id = %s ORDER BY subject_id",
                       (row[0],))
        return row[0], [r[0] for r in cursor.fetchall()]


def _measurements(conn, stays: pd.DataFrame) -> pd.DataFrame:
    """Chart and laboratory values of the stays, one row per value and stay, labelled by variable.

    Laboratory values belong to the admission, so a value is repeated for
    every stay of its admission. Within a variable, rows run by item and
    then in query order; `pulse.hourly` re-sorts them by time before
    binning.
    """
    # These two queries paste the identifier lists into the text, as the
    # queries that built the published cache did; the query plan, and with
    # it the order of rows, can differ between `IN (...)` and `= ANY(%s)`.
    # The identifiers are integers read from the database.
    frames = []
    for table in ("chartevents", "labevents"):
        items = {item: name for name, source, ids in ITEMS if source == table for item in ids}
        if table == "chartevents":
            query = CHARTEVENTS.format(stays=",".join(map(str, stays["stay_id"].tolist())),
                                       items=",".join(map(str, items)))
            rows = _read(conn, query)
            rows = rows.merge(stays[["stay_id", "intime"]], on="stay_id", how="inner")
        else:
            query = LABEVENTS.format(admissions=",".join(map(str, stays["hadm_id"].unique().tolist())),
                                     items=",".join(map(str, items)))
            rows = _read(conn, query)
            rows = rows.merge(stays[["hadm_id", "stay_id", "intime"]], on="hadm_id", how="inner")
        for item, group in rows.groupby("itemid"):
            frames.append(group.assign(variable=items[item]))
    if not frames:
        return pd.DataFrame(columns=["variable", "itemid", "charttime", "valuenum", "stay_id"])
    order = {name: i for i, (name, _, _) in enumerate(ITEMS)}
    frames.sort(key=lambda frame: order[frame["variable"].iloc[0]])
    return pd.concat(frames, ignore_index=True)


def load_stays(conn, cohort_id: int, subjects: list[int]) -> list[tuple[dict, dict]]:
    """Source records and stay details of the cohort stays of `subjects`, in stay order."""
    stays = _read(conn, STAYS, (cohort_id, subjects))
    if stays.empty:
        return []
    stay_ids = stays["stay_id"].tolist()
    patients = _read(conn, PATIENTS, (subjects,)).set_index("subject_id", drop=False)
    tables = {"measurements": _measurements(conn, stays),
              "inputevents": _read(conn, INPUTEVENTS, (stay_ids,)),
              "procedureevents": _read(conn, PROCEDUREEVENTS, (stay_ids,)),
              "outputevents": _read(conn, OUTPUTEVENTS, (stay_ids,))}
    grouped = {key: dict(tuple(frame.groupby("stay_id"))) for key, frame in tables.items()}
    result = []
    for row in stays.sort_values("stay_id").itertuples(index=False):
        patient = patients.loc[row.subject_id]
        frames = {key: grouped[key].get(row.stay_id, tables[key].iloc[:0]) for key in tables}
        stay = {"subject_id": row.subject_id, "hadm_id": row.hadm_id, "stay_id": row.stay_id,
                "intime": row.intime, "outtime": row.outtime,
                "dod": patient["dod"] if pd.notna(patient["dod"]) else None,
                "anchor_age": patient["anchor_age"], "gender": patient["gender"]}
        result.append((frames, stay))
    return result


def build_cache(dsn: str, out: Path, subjects_per_query: int = SUBJECTS_PER_QUERY) -> Path:
    """Bin every cohort stay and write the cache to `out`."""
    rows = []
    with contextlib.closing(_connect(dsn)) as conn:
        cohort_id, subjects = cohort_subjects(conn)
        for start in range(0, len(subjects), subjects_per_query):
            for frames, stay in load_stays(conn, cohort_id, subjects[start:start + subjects_per_query]):
                rows.append((stay, hourly.bin_stay(frames, stay), hourly.static_variables(frames, stay)))
    rows.sort(key=lambda r: (r[0]["subject_id"], r[0]["intime"], r[0]["stay_id"]))

    def stack(key: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([binned[key] for _, binned, _ in rows]))

    cache = {"continuous": stack("continuous"), "continuous_mask": stack("continuous_mask"),
             "binary": stack("binary"),
             "time_hours": torch.arange(hourly.HOURS, dtype=torch.float32).expand(len(rows), -1).clone(),
             "static": torch.from_numpy(np.stack([static for _, _, static in rows]))}
    for key in ("subject_ids", "stay_ids", "hadm_ids"):
        cache[key] = torch.tensor([int(stay[key[:-1]]) for stay, _, _ in rows], dtype=torch.long)
    return data.save_cache(cache, out)

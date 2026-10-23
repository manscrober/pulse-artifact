"""Read the sepsis cohort from a MIMIC-IV DuckDB database and write the cache.

The database holds MIMIC-IV and the derived concepts of the MIMIC Code
Repository under their usual schema names (`mimiciv_hosp`, `mimiciv_icu`,
`mimiciv_derived`). `sql/cohort.sql` selects the cohort stays, and every
query only reads. Subjects are read 25 at a time, in subject order, with one
query per source table; each stay is binned by `pulse.hourly`, and the cache
holds the stays sorted by subject, ICU admission time and stay.

The queries select only the columns the binning reads, with identifier
lists bound as parameters. Each query also bounds the subject identifier to
the range of the subjects read, which every requested row lies in, so DuckDB
skips the blocks of a table sorted by subject that hold other subjects. The
order of rows with equal sort keys is unspecified in SQL, so `pulse.hourly`
sorts every table by a total key before binning, and the cache depends only
on the records. The published cache was binned in the order the database
returned. The sort reorders records within the same hour, whether their
times are equal or not, so a rebuild can differ from the published cache in
the last bit of hourly sums and means.

Only this module needs the DuckDB driver, installed with the `cohort` extra;
it is imported when a connection is opened.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pulse import data, hourly
from pulse import variables as V

COHORT_SQL = Path(__file__).resolve().parent.parent / "sql" / "cohort.sql"
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

# Each query selects the columns the binning reads and the identifiers that
# group the rows by stay. Joins with icustays, admissions and d_items keep
# the rows of the published build; identifier lists are bound as lists, and
# $first and $last are the smallest and largest subject read.
STAYS = """
SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.admissions a USING (subject_id, hadm_id)
WHERE i.subject_id BETWEEN $first AND $last AND i.stay_id = ANY($stays)
ORDER BY i.subject_id, i.intime
"""
PATIENTS = """
SELECT subject_id, gender, anchor_age, dod
FROM mimiciv_hosp.patients
WHERE subject_id BETWEEN $first AND $last AND subject_id = ANY($subjects)
ORDER BY subject_id
"""
CHARTEVENTS = """
SELECT stay_id, charttime, itemid, valuenum
FROM mimiciv_icu.chartevents
WHERE subject_id BETWEEN $first AND $last AND stay_id = ANY($stays) AND itemid = ANY($items)
  AND valuenum IS NOT NULL
"""
LABEVENTS = """
SELECT hadm_id, charttime, itemid, valuenum
FROM mimiciv_hosp.labevents
WHERE subject_id BETWEEN $first AND $last AND hadm_id = ANY($admissions) AND itemid = ANY($items)
  AND valuenum IS NOT NULL
"""
INPUTEVENTS = """
SELECT ie.stay_id, ie.starttime, ie.endtime, ie.itemid, ie.amount, ie.amountuom, ie.rate, ie.rateuom, di.label
FROM mimiciv_icu.inputevents ie
JOIN mimiciv_icu.icustays i USING (subject_id, hadm_id, stay_id)
JOIN mimiciv_icu.d_items di ON ie.itemid = di.itemid
WHERE ie.subject_id BETWEEN $first AND $last AND ie.stay_id = ANY($stays)
ORDER BY ie.subject_id, ie.stay_id, ie.starttime
"""
PROCEDUREEVENTS = """
SELECT pe.stay_id, pe.starttime, pe.endtime, pe.itemid
FROM mimiciv_icu.procedureevents pe
JOIN mimiciv_icu.icustays i USING (subject_id, hadm_id, stay_id)
JOIN mimiciv_icu.d_items di ON pe.itemid = di.itemid
WHERE pe.subject_id BETWEEN $first AND $last AND pe.stay_id = ANY($stays)
ORDER BY pe.subject_id, pe.stay_id, pe.starttime
"""
OUTPUTEVENTS = """
SELECT oe.stay_id, oe.charttime, oe.itemid, oe.value, oe.valueuom
FROM mimiciv_icu.outputevents oe
JOIN mimiciv_icu.icustays i USING (subject_id, hadm_id, stay_id)
JOIN mimiciv_icu.d_items di ON oe.itemid = di.itemid
WHERE oe.subject_id BETWEEN $first AND $last AND oe.stay_id = ANY($stays)
ORDER BY oe.subject_id, oe.stay_id, oe.charttime
"""


def _connect(database: Path):
    """A read-only DuckDB connection; tests replace this function."""
    import duckdb

    return duckdb.connect(str(database), read_only=True)


def _read(conn, query: str, params: dict | None = None) -> pd.DataFrame:
    return conn.execute(query, params or {}).df()


def cohort_stays(conn) -> pd.DataFrame:
    """Subject, admission and stay of every cohort stay, selected by `sql/cohort.sql`."""
    return _read(conn, COHORT_SQL.read_text())


def _measurements(conn, stays: pd.DataFrame) -> pd.DataFrame:
    """Chart and laboratory values of the stays, one row per value and stay, labelled by variable.

    Laboratory values belong to the admission, so a value is repeated for
    every stay of its admission. Within a variable, rows run by item and
    then in query order; `pulse.hourly` re-sorts them by time before
    binning.
    """
    frames = []
    bounds = {"first": int(stays["subject_id"].min()), "last": int(stays["subject_id"].max())}
    for table in ("chartevents", "labevents"):
        items = {item: name for name, source, ids in ITEMS if source == table for item in ids}
        if table == "chartevents":
            rows = _read(conn, CHARTEVENTS, {**bounds, "stays": stays["stay_id"].tolist(), "items": list(items)})
            rows = rows.merge(stays[["stay_id", "intime"]], on="stay_id", how="inner")
        else:
            rows = _read(conn, LABEVENTS, {**bounds, "admissions": stays["hadm_id"].unique().tolist(),
                                           "items": list(items)})
            rows = rows.merge(stays[["hadm_id", "stay_id", "intime"]], on="hadm_id", how="inner")
        for item, group in rows.groupby("itemid"):
            frames.append(group.assign(variable=items[item]))
    if not frames:
        return pd.DataFrame(columns=["variable", "itemid", "charttime", "valuenum", "stay_id"])
    order = {name: i for i, (name, _, _) in enumerate(ITEMS)}
    frames.sort(key=lambda frame: order[frame["variable"].iloc[0]])
    return pd.concat(frames, ignore_index=True)


def load_stays(conn, stay_ids: list[int], subjects: list[int]) -> list[tuple[dict, dict]]:
    """Source records and stay details of the cohort stays `stay_ids` of `subjects`, in stay order."""
    bounds = {"first": min(subjects), "last": max(subjects)}
    stays = _read(conn, STAYS, {**bounds, "stays": stay_ids})
    if stays.empty:
        return []
    stay_ids = stays["stay_id"].tolist()
    patients = _read(conn, PATIENTS, {**bounds, "subjects": subjects}).set_index("subject_id", drop=False)
    tables = {"measurements": _measurements(conn, stays),
              "inputevents": _read(conn, INPUTEVENTS, {**bounds, "stays": stay_ids}),
              "procedureevents": _read(conn, PROCEDUREEVENTS, {**bounds, "stays": stay_ids}),
              "outputevents": _read(conn, OUTPUTEVENTS, {**bounds, "stays": stay_ids})}
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


def build_cache(database: Path, out: Path, subjects_per_query: int = SUBJECTS_PER_QUERY) -> Path:
    """Bin every cohort stay of the DuckDB `database` and write the cache to `out`."""
    rows = []
    with contextlib.closing(_connect(database)) as conn:
        cohort = cohort_stays(conn)
        subjects = sorted(int(subject) for subject in cohort["subject_id"].unique())
        for start in range(0, len(subjects), subjects_per_query):
            chunk = subjects[start:start + subjects_per_query]
            stay_ids = [int(stay) for stay in cohort.loc[cohort["subject_id"].isin(chunk), "stay_id"]]
            for frames, stay in load_stays(conn, stay_ids, chunk):
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

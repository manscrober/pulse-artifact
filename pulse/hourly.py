"""Hourly binning of one ICU stay and its static variables.

Each record enters the hour of its clinical time, counted from ICU
admission, over the first 24 hours: measurements at chart time, infusions
and procedures in every hour they run, maintenance fluid hour by hour from
its rate, boluses (under five minutes) at their end time in their own
channel, and output volumes at chart time. Values outside a variable's range
are dropped. Derived variables follow from the binned values, and the hours
from death or release on are filled and marked unrecorded.

Diagnoses carry no time in MIMIC-IV, so nothing that depends on them is
known at admission: the pulmonary edema indicator, end-stage renal failure,
congestive heart failure and the Elixhauser score stay zero.

`bin_stay` takes the stay's records as data frames:

    measurements     variable, itemid, charttime, valuenum
    inputevents      starttime, endtime, amount, amountuom, rate, rateuom, label
    outputevents     charttime, value, valueuom
    procedureevents  starttime, endtime, itemid

and `stay` with `intime`, `outtime` and `dod` (date of death or None).

MIMIC-IV records the date of death without a time, so death is placed at
midnight starting that date. A patient who dies on the calendar day of ICU
admission gets a negative death hour and is never marked dead within the 24
hours; one who dies on a later day is marked dead from midnight, up to a day
before the actual death. `dod` also covers deaths after hospital discharge,
which cannot fall inside the first 24 ICU hours, so the variable named
"In-Hospital Mortality Indicator" marks death from any recorded cause
within the window.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from pulse import variables as V

HOURS = 24
VASOPRESSORS = ("Norepinephrine", "Epinephrine", "Phenylephrine", "Vasopressin", "Dopamine", "Dobutamine",
                "Milrinone")
DIURETICS = ("Furosemide", "Lasix", "Bumetanide", "Torsemide", "Metolazone")
VENTILATION_ITEMS = (225792, 225794)
DIALYSIS_ITEMS = (225802, 225803, 225805, 225809)
FAHRENHEIT_ITEM = 223761
VOLUME_UNITS = ("ml", "mL", "l", "L")
BOLUS_MINUTES = 5
EDEMA_BALANCE = 500.0

# Two changes of mean blood pressure are computed only because the minimum
# changes are taken over them; they are not model variables.
CHANGE_FROM_BASELINE = "Change in Mean Blood Pressure from Baseline"
CHANGE_FROM_PREVIOUS = "Change in Mean Blood Pressure from Previous"
HIDDEN_RANGES = {CHANGE_FROM_BASELINE: (-80, 80), CHANGE_FROM_PREVIOUS: (-80, 80)}

# Derived variables in the order they are computed: (name, source, rule).
DERIVED = (
    ("Minimum Mean Blood Pressure", "Mean Blood Pressure", "running_min"),
    (CHANGE_FROM_BASELINE, "Mean Blood Pressure", "from_baseline"),
    ("Minimum Change in Mean Blood Pressure from Baseline", CHANGE_FROM_BASELINE, "running_min"),
    ("Minimum Mean Blood Pressure from Baseline", "Mean Blood Pressure", "min_from_baseline"),
    (CHANGE_FROM_PREVIOUS, "Mean Blood Pressure", "from_previous"),
    ("Minimum Change in Mean Blood Pressure from Previous", CHANGE_FROM_PREVIOUS, "running_min"),
    ("SOFA Score", None, "sofa"),
    ("Change in SOFA Score from Baseline", "SOFA Score", "from_baseline"),
    ("Change in SOFA Score from Previous", "SOFA Score", "from_previous"),
    ("O2 Requirement Level", None, "oxygen"),
    ("Change in O2 from Baseline", "O2 Requirement Level", "from_baseline"),
    ("Change in O2 from Previous", "O2 Requirement Level", "from_previous"),
    ("Change in Weight", "Weight", "from_previous"),
)
DERIVED_NAMES = {name for name, _, _ in DERIVED}

# Sort keys of each source table, applied before binning. SQL leaves the
# order of rows with equal sort keys unspecified, and hourly float64 sums
# and means depend on the order of their terms, so every table is put in
# one total order over the columns the binning reads. Rows equal in all of
# these columns are interchangeable.
ROW_ORDER = {
    "measurements": ("variable", "charttime", "itemid", "valuenum"),
    "inputevents": ("starttime", "endtime", "itemid", "label", "amount", "amountuom", "rate", "rateuom"),
    "outputevents": ("charttime", "itemid", "value", "valueuom"),
    "procedureevents": ("starttime", "endtime", "itemid"),
}
FLUIDS = ("Fluid Volume", "Urine Output", "Bolus Volume", "Fluid Balance in ICU", "Cumulative Fluids")
MEASURED = tuple(n for n in V.CONTINUOUS if n not in DERIVED_NAMES and n not in FLUIDS)
# Variables whose fill after death is the stay's own minimum; all others take zero.
NEGATIVE_AFTER_DEATH = ("Base excess", "Fluid Balance in ICU", "Minimum Mean Blood Pressure from Baseline",
                        CHANGE_FROM_BASELINE)


def _hours(frame: pd.DataFrame, column: str, intime) -> pd.DataFrame:
    """Rows with a known `column` time, with their hours since ICU admission in `hours`."""
    if frame.empty:
        return frame.copy()
    clinical = pd.to_datetime(frame[column], errors="coerce")
    out = frame.loc[clinical.notna()].copy()
    out["time"] = clinical[clinical.notna()]
    out["hours"] = (out["time"] - pd.to_datetime(intime, errors="coerce")).dt.total_seconds() / 3600.
    return out


def _in_hours(frame: pd.DataFrame) -> pd.DataFrame:
    hour = frame["hours"].apply(np.floor).astype(int)
    return frame.assign(hour=hour)[(hour >= 0) & (hour < HOURS)]


def _volumes(frame: pd.DataFrame, value: str, unit: str) -> pd.DataFrame:
    """Rows with a volume in millilitres or litres, converted to millilitres."""
    frame = frame[frame[unit].isin(VOLUME_UNITS)].copy()
    litres = frame[unit].isin(["l", "L"])
    if litres.any():
        frame.loc[litres, value] *= 1000
    return frame


def _active_hours(frame: pd.DataFrame, intime) -> list[int]:
    """Hours overlapped by each record's start-to-end interval; instantaneous records count at their start."""
    intime = pd.to_datetime(intime)
    active: set[int] = set()
    for row in frame.itertuples(index=False):
        rate = getattr(row, "rate", np.nan)
        if pd.notna(rate) and rate <= 0:
            continue
        start, end = pd.to_datetime(getattr(row, "starttime", pd.NaT)), pd.to_datetime(getattr(row, "endtime", pd.NaT))
        if pd.isna(start):
            continue
        if pd.isna(end) or end < start:
            end = start
        start_h = (start - intime).total_seconds() / 3600.0
        end_h = (end - intime).total_seconds() / 3600.0
        if end_h < 0 or start_h >= HOURS:
            continue
        last = max(start_h, end_h - 1e-9)
        active.update(range(max(0, int(np.floor(start_h))), min(HOURS - 1, int(np.floor(last))) + 1))
    return sorted(active)


def _maintenance_fluid(inputs: pd.DataFrame, intime, value: np.ndarray, mask: np.ndarray) -> None:
    """Add fluid volume per hour: rate times the time run in that hour, or the completed amount at its end."""
    origin = pd.to_datetime(intime)
    for row in inputs.itertuples(index=False):
        duration = pd.to_datetime(row.endtime) - pd.to_datetime(row.starttime)
        if pd.notna(duration) and duration.total_seconds() < 60 * BOLUS_MINUTES:
            continue
        per_hour = {"mL/hour": 1., "mL/min": 60.}.get(row.rateuom)
        if per_hour is not None and pd.notna(row.rate) and pd.notna(duration):
            start_h = (pd.to_datetime(row.starttime) - origin).total_seconds() / 3600
            end_h = (pd.to_datetime(row.endtime) - origin).total_seconds() / 3600
            rate = float(row.rate) * per_hour
            hour = int(np.floor(start_h))
            while hour * 1.0 < end_h:
                lo, hi = max(start_h, hour * 1.0), min(end_h, (hour + 1) * 1.0)
                volume = rate * (hi - lo)
                if hi > lo and 0 <= hour < HOURS and volume > 0:
                    value[hour] += volume
                    mask[hour] = 1
                hour += 1
            continue
        recorded = _hours(pd.DataFrame([row._asdict()]), "endtime", intime)
        if not recorded.empty:
            hour = int(np.floor(float(recorded["hours"].iloc[0])))
            if 0 <= hour < HOURS:
                value[hour] += float(row.amount)
                mask[hour] = 1


def _sofa(value: dict, mask: dict, t: int) -> int:
    """Partial SOFA score from SpO2, platelets, mean blood pressure and creatinine.

    The respiratory component uses SpO2 in place of PaO2/FiO2, and the
    liver and central nervous system components are left out, so this score
    differs from the full SOFA of `sql/sofa.sql`, which selects the cohort.
    """
    score = 0
    if mask["spO2"][t] > 0:
        score += next((s for limit, s in ((90, 4), (92, 3), (94, 2), (96, 1)) if value["spO2"][t] < limit), 0)
    if mask["Platelet"][t] > 0:
        score += next((s for limit, s in ((20, 4), (50, 3), (100, 2), (150, 1)) if value["Platelet"][t] < limit), 0)
    if mask["Mean Blood Pressure"][t] > 0 and value["Mean Blood Pressure"][t] < 70:
        score += 1
    if mask["Creatinine"][t] > 0:
        score += next((s for limit, s in ((5.0, 4), (3.5, 3), (2.0, 2), (1.2, 1)) if value["Creatinine"][t] >= limit), 0)
    return score


def _oxygen_level(value: dict, mask: dict, t: int) -> int:
    """Oxygen requirement from SpO2: 2 below 88%, 1 below 92%, else 0.

    The reference implementation also assigned level 3 to ventilated hours,
    but computed this variable before binning ventilation, so level 3 never
    occurs in the published cohort; the rule is kept as it ran.
    """
    if mask["spO2"][t] > 0:
        spo2 = value["spO2"][t]
        return 2 if spo2 < 88 else 1 if spo2 < 92 else 0
    return 0


def _derive(value: dict, mask: dict) -> None:
    for name, source, rule in DERIVED:
        out, out_mask = value[name], mask[name]
        if rule in ("sofa", "oxygen"):
            level = _sofa if rule == "sofa" else _oxygen_level
            for t in range(HOURS):
                out[t] = level(value, mask, t)
                out_mask[t] = 1
            continue
        x, m = value[source], mask[source]
        if rule == "from_previous":
            last = None
            for t in range(HOURS):
                if m[t] > 0:
                    if last is not None:
                        out[t], out_mask[t] = x[t] - last, 1
                    last = x[t]
                elif last is not None:
                    out[t], out_mask[t] = 0, 1
        elif rule == "from_baseline" and m[0] > 0:
            for t in range(HOURS):
                if m[t] > 0:
                    out[t], out_mask[t] = x[t] - x[0], 1
        elif rule == "running_min":
            current = float("inf")
            for t in range(HOURS):
                if m[t] > 0:
                    current = x[t] if current == float("inf") else min(current, x[t])
                if current != float("inf"):
                    out[t], out_mask[t] = current, 1
        elif rule == "min_from_baseline" and m[0] > 0:
            current = x[0]
            for t in range(HOURS):
                if m[t] > 0:
                    current = min(current, x[t])
                    out[t], out_mask[t] = current - x[0], 1


def _drop_out_of_range(value: dict, mask: dict, names) -> None:
    for name in names:
        low, high = HIDDEN_RANGES.get(name) or V.RANGES[name]
        invalid = (value[name] < low) | (value[name] > high)
        value[name][invalid] = 0
        mask[name][invalid] = 0


def _ordered(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The source tables sorted by `ROW_ORDER`, over the key columns each table has."""
    out = dict(frames)
    for name, keys in ROW_ORDER.items():
        frame = frames.get(name)
        present = [k for k in keys if frame is not None and k in frame.columns]
        if frame is not None and not frame.empty and present:
            out[name] = frame.sort_values(present, kind="mergesort", na_position="last").reset_index(drop=True)
    return out


def bin_stay(frames: Mapping[str, pd.DataFrame], stay: Mapping) -> dict[str, np.ndarray]:
    """Hourly `continuous` and `continuous_mask` [24, V] and `binary` [24, K] of one stay, as float32.

    The result does not depend on the order of rows within each table (`ROW_ORDER`).
    """
    frames = _ordered(frames)
    intime = stay["intime"]
    names = list(V.CONTINUOUS) + list(HIDDEN_RANGES)
    value = {name: np.zeros(HOURS) for name in names}
    mask = {name: np.zeros(HOURS) for name in names}
    binary = {name: np.zeros(HOURS) for name in V.BINARY}

    measurements = frames["measurements"]
    for name in MEASURED:
        records = _hours(measurements[measurements["variable"] == name], "charttime", intime)
        if records.empty:
            continue
        if name == "Temperature":
            fahrenheit = records["itemid"] == FAHRENHEIT_ITEM
            if fahrenheit.any():
                records.loc[fahrenheit, "valuenum"] = (records.loc[fahrenheit, "valuenum"] - 32) * 5 / 9
        binned = _in_hours(records).groupby("hour")["valuenum"].mean()
        value[name][binned.index.values] = binned.values
        mask[name][binned.index.values] = 1

    inputs = frames["inputevents"]
    if not inputs.empty:
        _maintenance_fluid(_volumes(inputs[inputs["amount"].notna()], "amount", "amountuom"), intime,
                           value["Fluid Volume"], mask["Fluid Volume"])
        boluses = inputs[inputs["amount"].notna() & inputs["endtime"].notna() & inputs["starttime"].notna()]
        boluses = _volumes(boluses, "amount", "amountuom")
        if not boluses.empty:
            minutes = (boluses["endtime"] - boluses["starttime"]).dt.total_seconds() / 60
            boluses = boluses[minutes < BOLUS_MINUTES]
            if not boluses.empty:
                binned = _in_hours(_hours(boluses, "endtime", intime))
                if not binned.empty:
                    binned = binned.groupby("hour")["amount"].sum()
                    value["Bolus Volume"][binned.index.values] += binned.values
                    mask["Bolus Volume"][binned.index.values] = 1

    outputs = frames["outputevents"]
    if not outputs.empty:
        urine = _volumes(outputs[outputs["value"].notna()], "value", "valueuom")
        if not urine.empty:
            binned = _in_hours(_hours(urine, "charttime", intime))
            if not binned.empty:
                binned = binned.groupby("hour")["value"].sum()
                value["Urine Output"][binned.index.values] = binned.values
                mask["Urine Output"][binned.index.values] = 1

    _drop_out_of_range(value, mask, [n for n in V.CONTINUOUS if n not in DERIVED_NAMES])

    balance = np.zeros(HOURS)
    balance += value["Fluid Volume"]
    balance += value["Bolus Volume"]
    balance -= value["Urine Output"]
    balance_mask = np.maximum(np.maximum(mask["Fluid Volume"], mask["Bolus Volume"]), mask["Urine Output"])
    value["Fluid Balance in ICU"], mask["Fluid Balance in ICU"] = balance, balance_mask
    total = 0
    for t in range(HOURS):
        if balance_mask[t] > 0:
            total += balance[t]
            value["Cumulative Fluids"][t], mask["Cumulative Fluids"][t] = total, 1

    _derive(value, mask)
    _drop_out_of_range(value, mask, [name for name, _, _ in DERIVED])

    if not inputs.empty:
        started = _hours(inputs, "starttime", intime)
        for name, drugs in (("Vasopressor Indicator", VASOPRESSORS), ("Diuretics Indicator", DIURETICS)):
            given = started[started["label"].str.contains("|".join(drugs), case=False, na=False)]
            if not given.empty:
                binary[name][_active_hours(given, intime)] = 1
    procedures = frames["procedureevents"]
    if not procedures.empty:
        procedures = _hours(procedures, "starttime", intime)
        for name, items in (("Mechanical Ventilation Indicator", VENTILATION_ITEMS),
                            ("Dialysis Indicator", DIALYSIS_ITEMS)):
            binary[name][_active_hours(procedures[procedures["itemid"].isin(items)], intime)] = 1

    if stay.get("dod") is not None and pd.notna(stay["dod"]):
        # A date without a time: death is placed at midnight of that date.
        death_hours = (pd.to_datetime(stay["dod"]) - pd.to_datetime(intime)).total_seconds() / 3600
        if 0 <= death_hours < HOURS:
            binary["In-Hospital Mortality Indicator"][int(death_hours):] = 1
    if stay.get("outtime") is not None:
        release_hours = (pd.to_datetime(stay["outtime"]) - pd.to_datetime(intime)).total_seconds() / 3600
        if release_hours <= HOURS and int(release_hours) < HOURS:
            binary["Release Indicator"][int(release_hours):] = 1
    # The name is kept from the published cohort, but the rule is hourly: the
    # indicator is set in every hour whose own fluid balance exceeds 500 mL,
    # whatever the running total.
    edema = (mask["Fluid Balance in ICU"] > 0) & (value["Fluid Balance in ICU"] > EDEMA_BALANCE)
    binary["Cumulative Edema"][edema] = 1

    _fill_cache_after_events(value, mask, binary)
    return {"continuous": np.stack([value[n] for n in V.CONTINUOUS], 1).astype(np.float32),
            "continuous_mask": np.stack([mask[n] for n in V.CONTINUOUS], 1).astype(np.float32),
            "binary": np.stack([binary[n] for n in V.BINARY], 1).astype(np.float32)}


def _fill_cache_after_events(value: dict, mask: dict, binary: dict) -> None:
    """From the first death hour, or without death from the first release hour, fill and unmark values.

    This is the rule that built the cache, from the stay's own statistics:
    after death a variable becomes zero, or the stay's minimum for the four
    variables in `NEGATIVE_AFTER_DEATH`; after release it becomes the
    stay's mean. Statistics use the recorded hours of the stay. Rollouts and
    the scorer use a different rule, `pulse.variables.fill_after_events`,
    with range bounds and training means.
    """
    died = np.flatnonzero(binary["In-Hospital Mortality Indicator"] > 0.5)
    released = np.flatnonzero(binary["Release Indicator"] > 0.5)
    if len(died):
        start, after_death = int(died[0]), True
    elif len(released):
        start, after_death = int(released[0]), False
    else:
        return
    for name in value:
        recorded = mask[name] > 0
        mean = value[name][recorded].mean() if recorded.sum() > 0 else 0.0
        minimum = value[name][recorded].min() if recorded.sum() > 0 else 0.0
        if after_death:
            value[name][start:] = minimum if name in NEGATIVE_AFTER_DEATH else 0.0
        else:
            value[name][start:] = mean
        mask[name][start:] = 0


def static_variables(frames: Mapping[str, pd.DataFrame], stay: Mapping) -> np.ndarray:
    """Age, fluid given before ICU admission (mL), Elixhauser score, sex, renal failure, heart failure, as float32 [6]."""
    frames = _ordered(frames)
    inputs = _hours(frames["inputevents"], "endtime", stay["intime"])
    before = 0.0
    if not inputs.empty:
        fluids = inputs[inputs["amount"].notna() & (inputs["hours"] < 0)].copy()
        if not fluids.empty:
            before = float(max(_volumes(fluids, "amount", "amountuom")["amount"].sum(), 0.0))
    static = np.zeros(len(V.STATIC))
    static[V.STATIC.index("Age")] = stay["anchor_age"]
    static[V.STATIC.index("Pre-ICU Fluid Amount")] = before
    static[V.STATIC.index("Gender")] = 1.0 if stay["gender"] == "M" else 0.0
    return static.astype(np.float32)

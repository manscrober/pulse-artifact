import numpy as np
import pandas as pd
import torch

from pulse import hourly
from pulse import variables as V
from synthetic import source_stays


def empty_frames() -> dict:
    return {"measurements": pd.DataFrame(columns=["variable", "itemid", "charttime", "valuenum"]),
            "inputevents": pd.DataFrame(), "outputevents": pd.DataFrame(), "procedureevents": pd.DataFrame()}


def test_measurements_and_release():
    intime = pd.Timestamp("2150-01-01 08:00")
    frames = empty_frames()
    frames["measurements"] = pd.DataFrame({
        "variable": ["Heart Rate", "Heart Rate", "Heart Rate", "Temperature", "Mean Blood Pressure"],
        "itemid": [1, 1, 1, hourly.FAHRENHEIT_ITEM, 1],
        "charttime": intime + pd.to_timedelta([0.2, 0.7, 3.5, 1.1, 0.5], unit="h"),
        "valuenum": [80.0, 90.0, 300.0, 98.6, 65.0]})
    stay = {"intime": intime, "outtime": intime + pd.Timedelta(hours=5.5), "dod": None}
    out = hourly.bin_stay(frames, stay)
    hr, temp = V.CONTINUOUS.index("Heart Rate"), V.CONTINUOUS.index("Temperature")
    assert out["continuous"][0, hr] == 85.0 and out["continuous_mask"][3, hr] == 0  # 300 is out of range
    assert abs(out["continuous"][1, temp] - 37.0) < 1e-5
    assert out["binary"][:, V.RELEASE].tolist() == [0.0] * 5 + [1.0] * 19
    assert not out["continuous_mask"][5:].any()
    assert out["continuous"][0, V.CONTINUOUS.index("SOFA Score")] == 1  # mean pressure below 70


def test_against_reference(expected):
    for s, (frames, stay) in enumerate(source_stays()):
        out = hourly.bin_stay(frames, stay)
        for key in ("continuous", "continuous_mask", "binary"):
            expected(f"stay{s}_{key}", torch.from_numpy(out[key]))
        expected(f"stay{s}_static", torch.from_numpy(hourly.static_variables(frames, stay)))
        assert out["continuous"].dtype == np.float32


def test_row_order_within_tables_does_not_matter():
    """Shuffling every table, ties included, gives the same sorted tables and the same tensors.

    Times are rounded down to the hour to force ties. Float32 outputs rarely
    show a different summation order, so the sorted tables are compared too.
    """
    rng = np.random.default_rng(3)
    for frames, stay in source_stays(count=6):
        tied = {}
        for name, frame in frames.items():
            frame = frame.copy()
            for column in ("charttime", "starttime", "endtime"):
                if column in frame:
                    frame[column] = frame[column].dt.floor("h")
            tied[name] = frame
        reference = hourly.bin_stay(tied, stay), hourly.static_variables(tied, stay)
        for _ in range(3):
            shuffled = {name: frame.iloc[rng.permutation(len(frame))] for name, frame in tied.items()}
            ordered, expected_tables = hourly._ordered(shuffled), hourly._ordered(tied)
            for name in tied:
                pd.testing.assert_frame_equal(ordered[name], expected_tables[name])
            out = hourly.bin_stay(shuffled, stay)
            for key in ("continuous", "continuous_mask", "binary"):
                assert np.array_equal(out[key], reference[0][key], equal_nan=True)
            assert np.array_equal(hourly.static_variables(shuffled, stay), reference[1], equal_nan=True)

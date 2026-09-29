"""
Load BEV field-data parquet files into the PyDPEET column schema.

Field data from the fleet logger arrives with different column names than
the lab tester output. This module is the one-stop converter that turns a
raw field-data parquet (signals from CAN bus: pack voltage, cell voltage
mean, pack current, SOC, temperatures, …) into a DataFrame that has the
same column layout as a Neware tester export, so the existing extraction
pipeline (`extract_pauses`, `extract_fuds`, `pauses_to_ocv_simple`,
`extract_ocv_iocv`, …) can be reused unchanged.

Target schema
-------------
``Meta_Data``        : str, constant ``"FieldData"``
``Step_Count``       : int, ``0`` (field data has no step grouping)
``Voltage[V]``       : cell-average voltage (``voltage_cell_avg``)
``Current[A]``       : pack current (``current``, same for every series cell)
``Temperature[°C]``  : average cell temperature (``temperature_cell_avg``)
``Test_Time[s]``     : seconds since the first kept sample
``Date_Time``        : original UTC timestamp from the parquet index
``EIS_*``            : NaN (no EIS in field data)
``Capacity[Ah]``     : cumulative ``∫ I dt / 3600`` (signed; sign follows
                      the source convention)
``SOC``              : ``state_of_charge / 100`` after forward-fill, in [0, 1]

Source / target voltage choice
------------------------------
The half-cell fitting pipeline operates at the **cell** level, so the
cell-mean voltage (``voltage_cell_avg``, ≈ 3.85 V) is mapped to
``Voltage[V]`` by default. The pack voltage (``voltage``, ≈ 700 V) is
preserved in the auxiliary column ``PackVoltage[V]`` for sanity checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# Canonical column order matching Neware tester exports
_TARGET_COLUMNS = [
    "Meta_Data",
    "Step_Count",
    "Voltage[V]",
    "Current[A]",
    "Temperature[°C]",
    "Test_Time[s]",
    "Date_Time",
    "EIS_f[Hz]",
    "EIS_Z_Real[Ohm]",
    "EIS_Z_Imag[Ohm]",
    "EIS_DC[A]",
    "Capacity[Ah]",
    "SOC",
]


def load_field_data(
    parquet_path: str | Path,
    *,
    voltage_source: Literal["cell_avg", "pack"] = "cell_avg",
    forward_fill_soc: bool = True,
    soc_scale: float = 0.01,
    integrate_capacity: bool = True,
    drop_nan_current: bool = True,
    drop_nan_voltage: bool = True,
    start_time: pd.Timestamp | str | None = None,
    end_time: pd.Timestamp | str | None = None,
    keep_pack_voltage: bool = True,
    keep_extra_columns: bool = False,
) -> pd.DataFrame:
    """
    Read a BEV field-data parquet and return it in the PyDPEET column schema.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the source parquet (e.g.
        ``input_data/Felddaten/1320097179092396.parquet``).
    voltage_source : {"cell_avg", "pack"}, default "cell_avg"
        Which voltage to map to ``Voltage[V]``. For half-cell fitting use
        ``"cell_avg"``; for pack-level diagnostics use ``"pack"``.
    forward_fill_soc : bool, default True
        Field data only reports SOC when the BMS pushes an update
        (≈ 5 % of samples). Forward-fill makes SOC available on every
        retained row.
    soc_scale : float, default 0.01
        Multiplier applied to the source SOC. Source is in percent (0–100);
        the default 0.01 yields the [0, 1] convention used by the half-cell
        fit. Set to 1.0 to keep percent.
    integrate_capacity : bool, default True
        Derive ``Capacity[Ah]`` by cumulative trapezoidal integration of
        ``current * dt / 3600``. With the source sign convention
        (positive = discharge), positive ``Capacity[Ah]`` means net charge
        out. Set to False to leave the column NaN.
    drop_nan_current, drop_nan_voltage : bool, default True
        Drop rows with NaN in the respective source column before mapping.
        Rest detection relies on a numeric current, so NaN-current rows
        would break ``extract_pauses``.
    start_time, end_time : str or Timestamp, optional
        Restrict to ``[start_time, end_time]`` (both inclusive). Useful to
        cut out a single trip out of a 15-month log. Accepts ISO strings
        like ``"2024-06-01"``.
    keep_pack_voltage : bool, default True
        Preserve pack voltage in the auxiliary column ``PackVoltage[V]``
        (only when ``voltage_source="cell_avg"``).
    keep_extra_columns : bool, default False
        If True, the remaining source columns (``voltage_cell_min``,
        ``voltage_cell_max``, ``temperature_cell_min``, …) are kept too.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least the columns listed in the module docstring,
        sorted by ``Test_Time[s]`` (strictly increasing).

    Raises
    ------
    FileNotFoundError
        If ``parquet_path`` does not exist.
    KeyError
        If a required source column is missing.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    raw = pd.read_parquet(parquet_path)

    src_voltage = "voltage_cell_avg" if voltage_source == "cell_avg" else "voltage"
    required = {src_voltage, "current"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(
            f"Source parquet is missing required column(s) {sorted(missing)}; "
            f"available: {list(raw.columns)}"
        )

    df = raw.copy()

    if start_time is not None:
        df = df.loc[df.index >= pd.Timestamp(start_time, tz="UTC")]
    if end_time is not None:
        df = df.loc[df.index <= pd.Timestamp(end_time, tz="UTC")]

    if drop_nan_current:
        df = df.loc[df["current"].notna()]
    if drop_nan_voltage:
        df = df.loc[df[src_voltage].notna()]

    if df.empty:
        return pd.DataFrame(columns=_TARGET_COLUMNS)

    df = df.sort_index()
    # Strict-increasing time stamps required downstream — coalesce ties.
    df = df.loc[~df.index.duplicated(keep="first")]

    out = pd.DataFrame(index=df.index.copy())
    out["Meta_Data"]   = "FieldData"
    out["Step_Count"]  = 0
    out["Voltage[V]"]  = df[src_voltage].astype(float).to_numpy()
    out["Current[A]"]  = df["current"].astype(float).to_numpy()

    if "temperature_cell_avg" in df.columns:
        out["Temperature[°C]"] = df["temperature_cell_avg"].astype(float).to_numpy()
    else:
        out["Temperature[°C]"] = np.nan

    t_seconds = (df.index - df.index[0]).total_seconds().to_numpy()
    out["Test_Time[s]"] = t_seconds
    out["Date_Time"]    = df.index

    for c in ("EIS_f[Hz]", "EIS_Z_Real[Ohm]", "EIS_Z_Imag[Ohm]", "EIS_DC[A]"):
        out[c] = np.nan

    if integrate_capacity:
        dt = np.diff(t_seconds, prepend=t_seconds[0])
        out["Capacity[Ah]"] = np.cumsum(out["Current[A]"].to_numpy() * dt) / 3600.0
    else:
        out["Capacity[Ah]"] = np.nan

    if "state_of_charge" in df.columns:
        soc = df["state_of_charge"].astype(float)
        if forward_fill_soc:
            soc = soc.ffill()
        out["SOC"] = soc.to_numpy() * float(soc_scale)
    else:
        out["SOC"] = np.nan

    if keep_pack_voltage and voltage_source == "cell_avg" and "voltage" in df.columns:
        out["PackVoltage[V]"] = df["voltage"].astype(float).to_numpy()

    if keep_extra_columns:
        already = set(out.columns) | {src_voltage, "current", "temperature_cell_avg",
                                      "state_of_charge", "voltage"}
        for c in df.columns:
            if c not in already:
                out[c] = df[c].to_numpy()

    out = out.reset_index(drop=True)

    ordered = [c for c in _TARGET_COLUMNS if c in out.columns]
    extras = [c for c in out.columns if c not in ordered]
    return out[ordered + extras]


def _mean_tilde_floats(s) -> float:
    if not isinstance(s, str):
        return float("nan")
    try:
        parts = [p for p in s.split("~") if p]
        if not parts:
            return float("nan")
        return float(np.mean([float(p) for p in parts]))
    except (ValueError, TypeError):
        return float("nan")


def load_field_data_csv(
    csv_path: str | Path,
    *,
    voltage_source: Literal["cell_avg", "pack"] = "cell_avg",
    cell_voltage_method: Literal["cells_mean", "minmax_avg"] = "cells_mean",
    forward_fill_soc: bool = True,
    soc_scale: float = 0.01,
    integrate_capacity: bool = True,
    drop_nan_current: bool = True,
    drop_nan_voltage: bool = True,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
    keep_pack_voltage: bool = True,
    keep_extra_columns: bool = False,
    invert_current_sign: bool = False,
    time_origin: pd.Timestamp | str = "1970-01-01 00:00:00",
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Read a BEV field-data CSV (``vinXX`` schema) and return it in the
    PyDPEET column schema. Same output contract as :func:`load_field_data`.

    Source columns expected
    -----------------------
    ``terminaltime``         : float seconds (per-sample timestamp, monotonic)
    ``soc``                  : state of charge in percent (0–100)
    ``totalvoltage``         : pack voltage
    ``totalcurrent``         : pack current (source sign convention preserved)
    ``minvoltagebattery``    : min cell voltage
    ``maxvoltagebattery``    : max cell voltage
    ``batteryvoltage``       : tilde-separated list of individual cell voltages
    ``mintemperaturevalue``  : min cell temperature
    ``maxtemperaturevalue``  : max cell temperature

    Parameters
    ----------
    csv_path : str or Path
        Path to the source CSV (e.g.
        ``input_data/Felddaten/vin38.csv``).
    voltage_source : {"cell_avg", "pack"}, default "cell_avg"
        Which voltage to map to ``Voltage[V]``.
    cell_voltage_method : {"cells_mean", "minmax_avg"}, default "cells_mean"
        How the cell-average voltage is computed when
        ``voltage_source="cell_avg"``. ``"cells_mean"`` parses
        ``batteryvoltage`` and averages all cells (matches the parquet
        ``voltage_cell_avg``); ``"minmax_avg"`` uses
        ``(minvoltagebattery + maxvoltagebattery) / 2`` (much faster,
        slight bias).
    forward_fill_soc : bool, default True
        Forward-fill ``soc`` over rows where the BMS did not push an update.
    soc_scale : float, default 0.01
        Multiplier applied to the source SOC (percent → fraction by default).
    integrate_capacity : bool, default True
        Derive ``Capacity[Ah]`` by cumulative integration of
        ``current * dt / 3600``. Source sign convention is preserved.
    drop_nan_current, drop_nan_voltage : bool, default True
        Drop rows with NaN in the respective source column before mapping.
    start_time_s, end_time_s : float, optional
        Restrict to ``terminaltime`` in ``[start_time_s, end_time_s]``
        (both inclusive). The CSV has no absolute timestamps, so the cut
        is in seconds of ``terminaltime``.
    keep_pack_voltage : bool, default True
        Preserve pack voltage in the auxiliary column ``PackVoltage[V]``
        (only when ``voltage_source="cell_avg"``).
    keep_extra_columns : bool, default False
        If True, the remaining source columns (``speed``, ``totalodometer``,
        ``chargestatus``, …) are kept too.
    invert_current_sign : bool, default False
        Multiply ``totalcurrent`` by -1 before mapping. Use this if your
        downstream pipeline expects the opposite charge/discharge sign.
    time_origin : Timestamp or str, default "1970-01-01 00:00:00"
        Synthetic origin used to build ``Date_Time``. The CSV provides only
        relative seconds, so the absolute date is a placeholder unless the
        caller knows the trip start and passes it in.
    nrows : int, optional
        Read only the first ``nrows`` rows of the CSV. Useful for testing
        on a slice of the 3 GB file.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same columns as :func:`load_field_data`,
        sorted by ``Test_Time[s]`` (strictly increasing).

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    KeyError
        If a required source column is missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    needed = {
        "terminaltime", "soc", "totalvoltage", "totalcurrent",
        "minvoltagebattery", "maxvoltagebattery",
        "mintemperaturevalue", "maxtemperaturevalue",
    }
    parse_cells = voltage_source == "cell_avg" and cell_voltage_method == "cells_mean"
    if parse_cells:
        needed.add("batteryvoltage")

    raw = pd.read_csv(
        csv_path,
        usecols=lambda c: c in needed,
        nrows=nrows,
        low_memory=False,
    )

    required = {"terminaltime", "totalcurrent"}
    if voltage_source == "pack":
        required.add("totalvoltage")
    elif cell_voltage_method == "cells_mean":
        required.add("batteryvoltage")
    else:
        required |= {"minvoltagebattery", "maxvoltagebattery"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(
            f"Source CSV is missing required column(s) {sorted(missing)}; "
            f"available: {list(raw.columns)}"
        )

    df = raw.copy()

    if start_time_s is not None:
        df = df.loc[df["terminaltime"] >= float(start_time_s)]
    if end_time_s is not None:
        df = df.loc[df["terminaltime"] <= float(end_time_s)]

    if drop_nan_current:
        df = df.loc[df["totalcurrent"].notna()]

    if voltage_source == "pack":
        src_v = df["totalvoltage"].astype(float)
    elif cell_voltage_method == "cells_mean":
        src_v = df["batteryvoltage"].map(_mean_tilde_floats)
    else:
        src_v = (df["minvoltagebattery"].astype(float)
                 + df["maxvoltagebattery"].astype(float)) / 2.0

    if drop_nan_voltage:
        mask = src_v.notna()
        df = df.loc[mask]
        src_v = src_v.loc[mask]

    if df.empty:
        return pd.DataFrame(columns=_TARGET_COLUMNS)

    df = df.sort_values("terminaltime", kind="mergesort")
    src_v = src_v.loc[df.index]
    keep = ~df["terminaltime"].duplicated(keep="first")
    df = df.loc[keep].reset_index(drop=True)
    src_v = src_v.loc[keep.index[keep]].reset_index(drop=True)

    t = df["terminaltime"].astype(float).to_numpy()
    t_seconds = t - t[0]
    timestamps = pd.Timestamp(time_origin) + pd.to_timedelta(t, unit="s")

    current = df["totalcurrent"].astype(float).to_numpy()
    if invert_current_sign:
        current = -current

    out = pd.DataFrame(index=range(len(df)))
    out["Meta_Data"]   = "FieldData"
    out["Step_Count"]  = 0
    out["Voltage[V]"]  = src_v.astype(float).to_numpy()
    out["Current[A]"]  = current

    if {"mintemperaturevalue", "maxtemperaturevalue"} <= set(df.columns):
        out["Temperature[°C]"] = (
            df["mintemperaturevalue"].astype(float).to_numpy()
            + df["maxtemperaturevalue"].astype(float).to_numpy()
        ) / 2.0
    else:
        out["Temperature[°C]"] = np.nan

    out["Test_Time[s]"] = t_seconds
    out["Date_Time"]    = timestamps.to_numpy()

    for c in ("EIS_f[Hz]", "EIS_Z_Real[Ohm]", "EIS_Z_Imag[Ohm]", "EIS_DC[A]"):
        out[c] = np.nan

    if integrate_capacity:
        dt = np.diff(t_seconds, prepend=t_seconds[0])
        out["Capacity[Ah]"] = np.cumsum(out["Current[A]"].to_numpy() * dt) / 3600.0
    else:
        out["Capacity[Ah]"] = np.nan

    if "soc" in df.columns:
        soc = df["soc"].astype(float)
        if forward_fill_soc:
            soc = soc.ffill()
        out["SOC"] = soc.to_numpy() * float(soc_scale)
    else:
        out["SOC"] = np.nan

    if keep_pack_voltage and voltage_source == "cell_avg" and "totalvoltage" in df.columns:
        out["PackVoltage[V]"] = df["totalvoltage"].astype(float).to_numpy()

    if keep_extra_columns:
        already = set(out.columns) | {
            "terminaltime", "totalcurrent", "totalvoltage",
            "minvoltagebattery", "maxvoltagebattery",
            "mintemperaturevalue", "maxtemperaturevalue",
            "batteryvoltage", "soc",
        }
        for c in df.columns:
            if c not in already:
                out[c] = df[c].to_numpy()

    ordered = [c for c in _TARGET_COLUMNS if c in out.columns]
    extras = [c for c in out.columns if c not in ordered]
    return out[ordered + extras]

import pandas as pd

_NEEDED_COLUMNS = {
    "terminaltime",
    "soc",
    "totalcurrent",
    "batteryvoltage",
    "mintemperaturevalue",
    "maxtemperaturevalue",
}

_TIME_ORIGIN = "1970-01-01 00:00:00"


def to_dataframe(input_path: str) -> tuple[pd.DataFrame, str]:
    """
    Parses a BEV field-data CSV log (vin schema) into a pandas DataFrame.

    Parameters:
    input_path (str): Path to the input file.

    Returns:
    (pandas.DataFrame, str): A tuple containing the DataFrame with data and metadata as a string.
    """
    raw = pd.read_csv(
        input_path,
        usecols=lambda c: c in _NEEDED_COLUMNS,
        low_memory=False,
    )

    missing = _NEEDED_COLUMNS - set(raw.columns)
    if missing:
        raise KeyError(
            f"Source CSV is missing required column(s) {sorted(missing)}; available: {list(raw.columns)}"
        )

    df = raw.loc[raw["totalcurrent"].notna()].copy()
    df["voltage_cell_avg"] = df["batteryvoltage"].map(_mean_tilde_floats)
    df = df.loc[df["voltage_cell_avg"].notna()]

    df = df.sort_values("terminaltime", kind="mergesort")
    df = df.loc[~df["terminaltime"].duplicated(keep="first")].reset_index(drop=True)

    df["current"] = df["totalcurrent"].astype(float)
    df["temperature_cell_avg"] = (
        df["mintemperaturevalue"].astype(float) + df["maxtemperaturevalue"].astype(float)
    ) / 2.0
    df["Date_Time"] = pd.Timestamp(_TIME_ORIGIN) + pd.to_timedelta(df["terminaltime"].astype(float), unit="s")
    df["Step_Count"] = 0
    df = df.rename(columns={"soc": "SOC"})

    out = df[[
        "terminaltime", "SOC", "current", "voltage_cell_avg",
        "temperature_cell_avg", "Date_Time", "Step_Count",
    ]].reset_index(drop=True)

    meta_data = (
        f"Field data CSV (vin schema): {input_path}\n"
        f"{len(out)} samples after cleaning (dropped NaN current/voltage, deduplicated timestamps)\n"
        f"source columns: {sorted(raw.columns)}"
    )

    return out, meta_data


def _mean_tilde_floats(value: object) -> float:
    """
    Averages a tilde-separated list of per-cell voltages, e.g. "3.80~3.81~3.79".
    """
    if not isinstance(value, str):
        return float("nan")
    try:
        parts = [p for p in value.split("~") if p]
        if not parts:
            return float("nan")
        return sum(float(p) for p in parts) / len(parts)
    except (ValueError, TypeError):
        return float("nan")

"""
Pause extraction from a test DataFrame.

A *pause* is a contiguous run of samples where the battery is neither
being charged nor discharged — i.e. ``|Current[A]| < threshold``. This
module provides one function that returns every pause in a test
DataFrame whose duration is at or above a user-specified minimum.

Typical use case: isolate the rest phases inside a FUDS, iOCV or HPPC
block so they can be inspected individually, plotted or fed one-by-one
into the relaxation extrapolation
(:func:`pydpeet.process.analyze.extract.relaxation.extrapolate_relaxation_ocv`).
For the combined ‘‘find pauses + fit each'' workflow,
:func:`pydpeet.process.analyze.extract.relaxation.extract_relaxation_anchors`
is the higher-level shortcut.

Stand-alone: no dependency on the PyDPEET step-analyzer. Only the
columns ``Test_Time[s]`` and ``Current[A]`` are required; all other
columns (``Voltage[V]``, ``Temperature[°C]``, ``SOC``, ``Capacity[Ah]``,
``EIS_*``, …) pass through unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_pauses(
    df: pd.DataFrame,
    *,
    min_pause_duration_s: float,
    rest_current_threshold_a: float = 0.01,
    time_col: str = "Test_Time[s]",
    current_col: str = "Current[A]",
) -> list[pd.DataFrame]:
    """
    Return every pause in ``df`` whose duration is at least
    ``min_pause_duration_s`` seconds.

    A pause is a contiguous run of samples with
    ``|current| < rest_current_threshold_a``. The returned DataFrames
    are time-ordered slices of (a sorted copy of) ``df``; all original
    columns are preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Test data. Must contain ``time_col`` and ``current_col``. Any
        additional columns pass through unchanged.
    min_pause_duration_s : float
        Minimum duration (in seconds) for a pause to be included.
        Shorter pauses are dropped.
    rest_current_threshold_a : float, default 0.01
        Absolute-current threshold below which a sample counts as
        rest. Pick a value above the instrument noise floor but well
        below the smallest meaningful load current — 10 mA is a
        sensible default for cell-level testers.
    time_col, current_col : str
        Column names in ``df``.

    Returns
    -------
    list[pd.DataFrame]
        One DataFrame per qualifying pause, in chronological order.
        Empty list if no pause meets the threshold or the input is too
        short.

    Raises
    ------
    ValueError
        If a required column is missing, ``min_pause_duration_s`` is
        negative or ``time_col`` is not strictly increasing.

    Examples
    --------
    >>> pauses = extract_pauses(df_fuds, min_pause_duration_s=10.0)
    >>> len(pauses)
    27
    >>> first = pauses[0]
    >>> first[["Test_Time[s]", "Voltage[V]", "SOC"]].head()
    """
    for c in (time_col, current_col):
        if c not in df.columns:
            raise ValueError(f"DataFrame missing required column {c!r}.")
    if min_pause_duration_s < 0:
        raise ValueError("min_pause_duration_s must be non-negative.")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    if len(df_sorted) < 2:
        return []

    t = df_sorted[time_col].to_numpy(dtype=float)
    i = df_sorted[current_col].to_numpy(dtype=float)

    if not np.all(np.diff(t) > 0):
        raise ValueError(f"{time_col!r} must be strictly increasing.")

    is_rest = np.abs(i) < rest_current_threshold_a
    state_diff = np.diff(is_rest.astype(np.int8))
    transitions = np.flatnonzero(state_diff != 0) + 1
    seg_starts = np.concatenate([[0], transitions])
    seg_ends = np.concatenate([transitions, [len(is_rest)]])  # exclusive

    pauses: list[pd.DataFrame] = []
    for s, e in zip(seg_starts, seg_ends):
        if not is_rest[s]:
            continue
        duration = float(t[e - 1] - t[s])
        if duration < min_pause_duration_s:
            continue
        pauses.append(df_sorted.iloc[s:e].copy())

    return pauses

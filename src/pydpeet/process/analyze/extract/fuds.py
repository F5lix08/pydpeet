"""
FUDS sub-test extraction from a test DataFrame.

The FUDS (Federal Urban Driving Schedule) phase of a checkup is characterised
by many short load segments with short rest intervals — in contrast to:

  * iOCV phases, where every load pulse is followed by a long (several
    minutes) rest so the voltage can relax;
  * Capacity tests, which consist of one or two long CC / CV segments;
  * HPPC, which has short rests but a very regular pulse pattern.

This module locates the FUDS block in a flat test DataFrame (raw or
primitives — the only required columns are ``Test_Time[s]`` and
``Current[A]``) using a current-threshold-based heuristic:

  1. A sample counts as "rest" if ``|Current[A]| < rest_current_threshold_a``,
     otherwise "load". Contiguous samples of the same type form segments.
  2. Rest segments longer than ``long_rest_threshold_s`` define boundaries
     between top-level test phases (Capacity / iOCV / FUDS / etc.).
  3. Inside each resulting block, FUDS is identified by:
       * at least ``min_segments_in_block`` segments, AND
       * a median internal rest duration below ``fuds_max_median_rest_s``.

The longest qualifying block is returned by default.

Stand-alone: this module does not depend on the PyDPEET step-analyzer or on
any segment-classification columns (``Type``, ``ID``, ``Step_Count``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def extract_fuds(
    df: pd.DataFrame,
    *,
    long_rest_threshold_s: float = 600.0,
    fuds_max_median_rest_s: float = 60.0,
    min_segments_in_block: int = 20,
    rest_current_threshold_a: float = 0.01,
    select: Literal["longest", "all", "first"] = "longest",
    time_col: str = "Test_Time[s]",
    current_col: str = "Current[A]",
) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Extract only the FUDS portion of a test DataFrame.

    Works on any flat DataFrame with the columns ``Test_Time[s]`` and
    ``Current[A]`` — no segment-classification columns (``Type``, ``ID``,
    ``Step_Count``) are required. Rest vs. load is decided sample-by-sample
    from the current magnitude.

    Detection (two-step heuristic):

      1. Rest samples (``|I| < rest_current_threshold_a``) and load samples
         form contiguous segments. Rest segments longer than
         ``long_rest_threshold_s`` act as boundaries between top-level test
         phases.
      2. Inside each resulting block, FUDS is identified if the block
         contains at least ``min_segments_in_block`` segments AND its median
         internal rest duration is below ``fuds_max_median_rest_s``.

    iOCV blocks fail step 2 because their internal rests are long
    (typically several minutes). Capacity blocks fail because they only
    contain a handful of long CC segments.

    Parameters
    ----------
    df : pd.DataFrame
        Test data. Must contain at least ``time_col`` and ``current_col``.
        Any additional columns (Voltage, SOC, Capacity, Temperature, …)
        pass through unchanged.
    long_rest_threshold_s : float, default 600.0
        Rests longer than this delimit phases. 10 min is typical for the
        gaps between Capacity / iOCV / FUDS in a checkup file.
    fuds_max_median_rest_s : float, default 60.0
        Maximum allowed median rest duration inside a FUDS block.
    min_segments_in_block : int, default 20
        Minimum segment count to consider a block FUDS-like.
    rest_current_threshold_a : float, default 0.01
        Absolute-current threshold below which a sample counts as rest.
        Pick a value above the instrument noise floor but well below the
        smallest meaningful load current (10 mA is a reasonable default
        for cell-level testers).
    select : {"longest", "all", "first"}, default "longest"
        Which candidate block to return:

        * ``"longest"`` — the block with the largest total duration.
        * ``"first"``   — the earliest block in test time.
        * ``"all"``     — return all candidates as a list of DataFrames.
    time_col, current_col : str
        Column names in ``df``.

    Returns
    -------
    pd.DataFrame or list[pd.DataFrame]
        For ``select`` ∈ {"longest", "first"}: a single DataFrame holding
        the rows of ``df`` belonging to the selected block, sorted by
        ``time_col``. For ``select == "all"``: a list of such DataFrames
        (possibly empty). Empty DataFrame (or empty list) if no block
        qualifies as FUDS.

    Raises
    ------
    ValueError
        If a required column is missing.
    """
    for c in (time_col, current_col):
        if c not in df.columns:
            raise ValueError(f"DataFrame missing required column {c!r}.")
    if select not in ("longest", "all", "first"):
        raise ValueError(f"select must be 'longest', 'all' or 'first', got {select!r}.")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    if len(df_sorted) < 2:
        empty = df_sorted.iloc[0:0].copy()
        return [] if select == "all" else empty

    time = df_sorted[time_col].to_numpy(dtype=float)
    current = df_sorted[current_col].to_numpy(dtype=float)

    is_rest = np.abs(current) < rest_current_threshold_a

    # Segment boundaries from rest/load transitions.
    state_changes = np.flatnonzero(np.diff(is_rest.astype(np.int8)) != 0)
    seg_starts = np.concatenate([[0], state_changes + 1])
    seg_ends = np.concatenate([state_changes + 1, [len(is_rest)]])  # exclusive
    seg_is_rest = is_rest[seg_starts]
    seg_durations = np.array(
        [time[e - 1] - time[s] for s, e in zip(seg_starts, seg_ends)],
        dtype=float,
    )

    is_long_rest = seg_is_rest & (seg_durations > long_rest_threshold_s)

    # Split segment indices into blocks separated by long-rest segments.
    blocks: list[tuple[int, int]] = []
    cur_start = 0
    for seg_idx, is_lr in enumerate(is_long_rest):
        if is_lr:
            if seg_idx > cur_start:
                blocks.append((cur_start, seg_idx))
            cur_start = seg_idx + 1
    if cur_start < len(seg_is_rest):
        blocks.append((cur_start, len(seg_is_rest)))

    fuds_blocks: list[dict] = []
    for b_start, b_end in blocks:
        n_segs = b_end - b_start
        if n_segs < min_segments_in_block:
            continue
        rest_mask = seg_is_rest[b_start:b_end]
        rests_durations = seg_durations[b_start:b_end][rest_mask]
        if rests_durations.size == 0:
            continue
        median_rest = float(np.median(rests_durations))
        if median_rest >= fuds_max_median_rest_s:
            continue
        sample_start = int(seg_starts[b_start])
        sample_end = int(seg_ends[b_end - 1])
        fuds_blocks.append(
            {
                "sample_start": sample_start,
                "sample_end": sample_end,
                "n_segments": int(n_segs),
                "median_rest_s": median_rest,
                "total_duration_s": float(time[sample_end - 1] - time[sample_start]),
            }
        )

    if not fuds_blocks:
        empty = df_sorted.iloc[0:0].copy()
        return [] if select == "all" else empty

    def _slice(b: dict) -> pd.DataFrame:
        return df_sorted.iloc[b["sample_start"]:b["sample_end"]].copy()

    if select == "all":
        return [_slice(b) for b in fuds_blocks]
    if select == "first":
        return _slice(fuds_blocks[0])
    best = max(fuds_blocks, key=lambda b: b["total_duration_s"])
    return _slice(best)

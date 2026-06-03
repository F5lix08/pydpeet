"""
End-to-end half-cell fit for a single field-data battery.

This module wraps the full pipeline behind one call: a raw field-data
frame goes in, a ranked table of half-cell candidate pairs comes out.
The steps are

1. split the record into fixed-length time windows
   (:func:`split_by_time_window`),
2. per window, isolate the rest pauses (:func:`extract_pauses`) and take
   the endpoint OCV of each (:func:`pauses_to_ocv_simple`),
3. per window with enough OCV anchors, grid-search every anode/cathode
   pair (:func:`find_best_half_cell_match`),
4. drop windows whose *best* achievable fit is worse than a quality gate,
5. rank the pairs by their median RMSE across all kept windows.

Because the cell never changes its electrode chemistry as it ages, the
electrode pair is identified once across the whole time series; only the
stoichiometry drifts per window. Aggregating over windows is what makes
the chemistry pick robust against the partial-SOC identifiability problem
that a single window suffers from — a single short window can be fit
equally well by several near-degenerate reference datasets, but only one
pair explains every window consistently.
"""

from __future__ import annotations

import pandas as pd

from pydpeet.process.analyze.extract.field_data_loader import split_by_time_window
from pydpeet.process.analyze.extract.pauses import extract_pauses
from pydpeet.process.analyze.extract.ocv_simple import pauses_to_ocv_simple
from pydpeet.process.analyze.extract.haf_cell_fitting import (
    find_best_half_cell_match,
    dir_anode,
    dir_cathode,
)

_REQUIRED_COLUMNS = ("Test_Time[s]", "Current[A]", "Voltage[V]", "SOC")

_CONSENSUS_COLUMNS = [
    "Anode", "Cathode", "n_chunks",
    "median_rmse", "mean_rmse", "max_rmse", "wins",
]

# Mapping from the public sort_by name to (columns, ascending) for
# DataFrame.sort_values. RMSE criteria rank ascending; "wins" ranks
# descending with median_rmse as tiebreaker.
_SORT_OPTIONS = {
    "median": (["median_rmse"], [True]),
    "mean":   (["mean_rmse"], [True]),
    "wins":   (["wins", "median_rmse"], [False, True]),
}


def fit_real_data(
    df: pd.DataFrame,
    *,
    window_days: float = 30.0,
    min_pause_duration_s: float = 600.0,
    min_points: int = 20,
    max_rmse_mv: float | None = 50.0,
    sort_by: str = "median",
    rest_current_threshold_a: float = 0.01,
    anodes_dir: str = dir_anode,
    cathodes_dir: str = dir_cathode,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Identify the most consistent half-cell pair for one field-data cell.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded field data. Must contain ``Test_Time[s]``, ``Current[A]``,
        ``Voltage[V]`` and ``SOC``; any further columns (``Date_Time``,
        ``Temperature[°C]``, ``EIS_*``, ``Capacity[Ah]``, ``PackVoltage[V]``,
        …) are ignored and may be present or absent.
    window_days : float, default 30
        Length of each time window the record is split into.
    min_pause_duration_s : float, default 600
        Minimum rest duration for a pause to yield an OCV anchor.
    min_points : int, default 20
        Windows with fewer cleaned OCV anchors are skipped (too little
        SOC coverage for a meaningful fit).
    max_rmse_mv : float or None, default 50
        Quality gate. A window whose *best* fit (over all pairs) exceeds
        this RMSE is treated as a broken OCV cloud and dropped before the
        consensus ranking. Pass ``None`` to disable the gate.
    sort_by : {"median", "mean", "wins"}, default "median"
        Ranking criterion for the returned table. ``"median"`` / ``"mean"``
        sort by the respective RMSE ascending (lower is better);
        ``"wins"`` sorts by win count descending with ``median_rmse`` as
        tiebreaker. The consensus pair is always the first row.
    rest_current_threshold_a : float, default 0.01
        Absolute-current threshold below which a sample counts as rest,
        forwarded to :func:`extract_pauses`.
    anodes_dir, cathodes_dir : str
        Half-cell reference directories, forwarded to
        :func:`find_best_half_cell_match`.
    verbose : bool, default True
        Print per-window skip reasons, the dropped windows and the chosen
        pair.

    Returns
    -------
    pd.DataFrame
        One row per candidate pair that was solvable in every kept window,
        sorted according to ``sort_by``. Columns:

        =========== =====================================================
        Anode        Anode reference filename
        Cathode      Cathode reference filename
        n_chunks     Number of kept windows the pair was fit in
        median_rmse  Median RMSE [mV] of the pair across those windows
        mean_rmse    Mean RMSE [mV]
        max_rmse     Worst-case RMSE [mV]
        wins         Number of windows where the pair was the rank-1 fit
        =========== =====================================================

        The chosen consensus pair is the first row. An empty frame with
        these columns is returned if no window survives.

    Raises
    ------
    ValueError
        If ``df`` is missing one of the required columns.
    """
    if sort_by not in _SORT_OPTIONS:
        raise ValueError(
            f"sort_by must be one of {list(_SORT_OPTIONS)}, got {sort_by!r}."
        )
    for c in _REQUIRED_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Input frame missing required column {c!r}.")

    chunks = split_by_time_window(df, window_days=window_days)

    per_chunk_rankings: list[pd.DataFrame] = []
    for i, chunk in enumerate(chunks):
        pauses = extract_pauses(
            chunk,
            min_pause_duration_s=min_pause_duration_s,
            rest_current_threshold_a=rest_current_threshold_a,
        )
        if not pauses:
            if verbose:
                print(f"chunk {i:>2}: keine Pausen ≥ {min_pause_duration_s:.0f} s")
            continue

        anchors = pauses_to_ocv_simple(pauses)
        df_for_fit = (
            anchors[["SOC", "U"]]
            .rename(columns={"U": "Voltage[V]"})
            .dropna()
            .sort_values("SOC")
            .drop_duplicates(subset="SOC")
            .reset_index(drop=True)
        )
        if len(df_for_fit) < min_points:
            if verbose:
                print(f"chunk {i:>2}: nur {len(df_for_fit)} OCV-Punkte → übersprungen")
            continue

        ranking = find_best_half_cell_match(
            df_for_fit,
            anodes_dir=anodes_dir,
            cathodes_dir=cathodes_dir,
            full_cell_name=f"chunk_{i:02d}",
        )
        if ranking.empty:
            continue
        per_chunk_rankings.append(ranking.assign(chunk=i))
        if verbose:
            print(f"chunk {i:>2}: {len(df_for_fit):>3} Punkte → fit OK")

    if not per_chunk_rankings:
        return pd.DataFrame(columns=_CONSENSUS_COLUMNS)

    all_pair_fits = pd.concat(per_chunk_rankings, ignore_index=True)

    # Quality gate: drop windows whose best-of-all-pairs fit is too poor.
    if max_rmse_mv is not None:
        chunk_best = all_pair_fits.groupby("chunk")["RMSE[mV]"].min()
        good = chunk_best[chunk_best <= max_rmse_mv].index
        bad = chunk_best[chunk_best > max_rmse_mv]
        if verbose and len(bad):
            print(f"\nVerworfen (bester Fit > {max_rmse_mv:.0f} mV):")
            for c, r in bad.items():
                print(f"  chunk {c:>2}: bester RMSE {r:.1f} mV")
        all_pair_fits = all_pair_fits[all_pair_fits["chunk"].isin(good)].reset_index(drop=True)

    if all_pair_fits.empty:
        return pd.DataFrame(columns=_CONSENSUS_COLUMNS)

    n_kept = all_pair_fits["chunk"].nunique()

    consensus = (
        all_pair_fits
        .groupby(["Anode", "Cathode"])
        .agg(
            n_chunks=("chunk", "nunique"),
            median_rmse=("RMSE[mV]", "median"),
            mean_rmse=("RMSE[mV]", "mean"),
            max_rmse=("RMSE[mV]", "max"),
        )
        .reset_index()
    )
    # keep only pairs that were solvable in every kept window
    consensus = consensus[consensus["n_chunks"] == n_kept]

    winners = (
        all_pair_fits.loc[
            all_pair_fits.groupby("chunk")["RMSE[mV]"].idxmin(),
            ["Anode", "Cathode"],
        ]
        .value_counts()
        .rename("wins")
        .reset_index()
    )
    consensus = consensus.merge(winners, on=["Anode", "Cathode"], how="left")
    consensus["wins"] = consensus["wins"].fillna(0).astype(int)

    sort_cols, ascending = _SORT_OPTIONS[sort_by]
    consensus = consensus.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    if verbose and not consensus.empty:
        top = consensus.iloc[0]
        print(f"\nGewähltes Konsens-Paar: {top['Anode']}  +  {top['Cathode']}")

    return consensus[_CONSENSUS_COLUMNS]

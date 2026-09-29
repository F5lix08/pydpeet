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

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from pydpeet.process.analyze.extract.phases.pauses import extract_pauses
from pydpeet.process.analyze.extract.phases.ocv_simple import pauses_to_ocv_simple
from pydpeet.process.analyze.extract.half_cell_fitting import (
    find_best_half_cell_match,
    fit_half_cells,
    dir_anode,
    dir_cathode,
)
from pydpeet.process.analyze.extract.capacity.capa_curvefit import build_full_cell_ocv_curve
from pydpeet.process.analyze.extract.capacity.capa_covered import (
    covered_soc_range,
    estimate_capacity_covered_mAh,
)
from pydpeet.process.analyze.extract.degradation.lithium_amount import calculate_electrode_quantities

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


def split_by_time_window(
    df: pd.DataFrame,
    window_days: float = 30.0,
    *,
    time_column: str = "Test_Time[s]",
    reset_time: bool = False,
) -> list[pd.DataFrame]:
    """
    Split a field-data DataFrame into chunks of ``window_days`` days each,
    bucketed on ``time_column`` (default ``Test_Time[s]``).

    Each chunk covers a contiguous ``[k * window, (k + 1) * window)``
    interval, in chronological order. Empty windows are skipped, so
    chunks are not guaranteed to be exactly ``window_days`` apart when
    the source has gaps.

    Parameters
    ----------
    df : pd.DataFrame
        Frame produced by :func:`~pydpeet.process.analyze.extract.field_data.field_data_loader.load_field_data`
        or :func:`~pydpeet.process.analyze.extract.field_data.field_data_loader.load_field_data_csv`,
        or from :func:`eet.read` with ``config="field_data_csv"``.
    window_days : float, default 30
        Window length in days.
    time_column : str, default "Test_Time[s]"
        Column to bucket on.
    reset_time : bool, default False
        If True, subtract each chunk's first ``time_column`` value so
        every chunk starts at 0.

    Returns
    -------
    list[pd.DataFrame]
        One DataFrame per window, with the original column layout.
    """
    if df.empty:
        return []
    window_s = float(window_days) * 86400.0
    bucket = (df[time_column].to_numpy() // window_s).astype(int)
    chunks: list[pd.DataFrame] = []
    for _, group in df.groupby(bucket, sort=True):
        chunk = group.reset_index(drop=True)
        if reset_time:
            chunk[time_column] = chunk[time_column] - chunk[time_column].iloc[0]
        chunks.append(chunk)
    return chunks


def fit_field_data(
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
    max_workers: int | None = None,
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
    max_workers : int, optional
        Number of worker threads used to run the per-window
        :func:`find_best_half_cell_match` calls in parallel. The grid
        search is CPU-bound but spends most of its time inside SciPy
        routines that release the GIL, so threading scales well in
        practice. ``None`` (default) uses
        :class:`~concurrent.futures.ThreadPoolExecutor`'s default
        (``min(32, os.cpu_count() + 4)``). Set to ``1`` to run
        sequentially.

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

    # Phase 1 (sequential, cheap): pauses + OCV anchors per chunk, logging
    # the skip decisions in stable order along the way. What remains is a
    # list of chunks that are actually fit in phase 2.
    fit_jobs: list[tuple[int, pd.DataFrame]] = []
    for i, chunk in enumerate(chunks):
        pauses = extract_pauses(
            chunk,
            min_pause_duration_s=min_pause_duration_s,
            rest_current_threshold_a=rest_current_threshold_a,
        )
        if not pauses:
            if verbose:
                print(f"chunk {i:>2}: no pauses ≥ {min_pause_duration_s:.0f} s")
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
                print(f"chunk {i:>2}: only {len(df_for_fit)} OCV points → skipped")
            continue

        fit_jobs.append((i, df_for_fit))

    # Phase 2 (parallel): the expensive grid-search fits over all
    # electrode pairs run per chunk in a thread pool.
    def _run_fit(job: tuple[int, pd.DataFrame]) -> tuple[int, int, pd.DataFrame]:
        i, df_for_fit = job
        ranking = find_best_half_cell_match(
            df_for_fit,
            anodes_dir=anodes_dir,
            cathodes_dir=cathodes_dir,
            full_cell_name=f"chunk_{i:02d}",
        )
        return i, len(df_for_fit), ranking

    per_chunk_rankings: list[pd.DataFrame] = []
    if fit_jobs:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_fit, job): job[0] for job in fit_jobs}
            for fut in as_completed(futures):
                i, n_pts, ranking = fut.result()
                if ranking.empty:
                    if verbose:
                        print(f"chunk {i:>2}: {n_pts:>3} points → fit failed")
                    continue
                per_chunk_rankings.append(ranking.assign(chunk=i))
                if verbose:
                    print(f"chunk {i:>2}: {n_pts:>3} points → fit OK")

    if not per_chunk_rankings:
        return pd.DataFrame(columns=_CONSENSUS_COLUMNS)

    all_pair_fits = pd.concat(per_chunk_rankings, ignore_index=True)

    # Quality gate: drop windows whose best-of-all-pairs fit is too poor.
    if max_rmse_mv is not None:
        chunk_best = all_pair_fits.groupby("chunk")["RMSE[mV]"].min()
        good = chunk_best[chunk_best <= max_rmse_mv].index
        bad = chunk_best[chunk_best > max_rmse_mv]
        if verbose and len(bad):
            print(f"\nDropped (best fit > {max_rmse_mv:.0f} mV):")
            for c, r in bad.items():
                print(f"  chunk {c:>2}: best RMSE {r:.1f} mV")
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
        print(f"\nChosen consensus pair: {top['Anode']}  +  {top['Cathode']}")

    return consensus[_CONSENSUS_COLUMNS]


def evaluate_chunks(
    chunks: list[pd.DataFrame],
    anode_name: str,
    cathode_name: str,
    *,
    anodes_dir: str = dir_anode,
    cathodes_dir: str = dir_cathode,
    min_pause_duration_s: float = 600.0,
    min_points: int = 20,
    max_rmse_mv: float = 50.0,
    min_dsoc_pair: float = 0.02,
    max_cycling_ratio: float | None = 2.0,
    soc_coverage_min: float = 0.60,
    cap_mad_k: float = 3.0,
    n_ref_chunks: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Per-chunk evaluation against a fixed electrode pair: fit, joint SOC
    window, curve-shape capacity and plausibility filter.

    Runs in three passes over the chunks (typically the output of
    :func:`split_by_time_window`, electrode pair typically the consensus
    pair from :func:`fit_field_data`):

    1. **Fit + coverage** — extract pauses, build anchors, fit against the
       fixed pair (gate ``max_rmse_mv``), build the reference curve, and
       determine the covered SOC range.
    2. **Capacity in the joint window** — intersection of the covered
       ranges of all chunks with width >= ``soc_coverage_min`` (otherwise
       collapses due to individual narrow chunks); per chunk,
       :func:`estimate_capacity_covered_mAh` in this window as well as the
       electrode quantities via :func:`calculate_electrode_quantities`.
    3. **Plausibility** — filter out capacity outliers via the MAD
       criterion (``|Cap - Median| > cap_mad_k * MAD``).

    Parameters
    ----------
    chunks : list of pd.DataFrame
        Time-window chunks with ``Test_Time[s]``, ``Current[A]``,
        ``Voltage[V]`` and ``SOC``.
    anode_name, cathode_name : str
        File names of the reference half-cells (e.g. from the consensus
        table of :func:`fit_field_data`).
    anodes_dir, cathodes_dir : str
        Directories of the half-cell references.
    min_pause_duration_s : float, default 600
        Minimum duration of a rest pause for an OCV anchor.
    min_points : int, default 20
        Chunks with fewer anchors are skipped.
    max_rmse_mv : float, default 50
        Fit quality gate per chunk.
    min_dsoc_pair, max_cycling_ratio
        Forwarded to :func:`estimate_capacity_covered_mAh`.
    soc_coverage_min : float, default 0.60
        Minimum width of the covered SOC range for a chunk to contribute
        to building the joint window. If fewer than three chunks remain,
        the filter is disabled for this run.
    cap_mad_k : float, default 3.0
        Strictness of the capacity outlier filter (pass 3).
    n_ref_chunks : int, default 3
        Number of the first plausible chunks whose median forms the SOH
        reference capacity.
    verbose : bool, default True
        Print rejection reasons and intermediate results.

    Returns
    -------
    dict
        ``chunk_data``      — per chunk index: raw data, anchors, fit,
        reference curve (``soc_ref``/``u_ref``) and coverage
        (``lo``/``hi``) from pass 1;
        ``global_window``   — joint SOC window ``(lo, hi)``;
        ``window_ids``      — chunks that form the window;
        ``cap_per_chunk_mAh``, ``fits_per_chunk``,
        ``quantities_per_chunk`` — results from pass 2;
        ``valid_ids``       — chunks with a valid capacity;
        ``soh_ids``         — chunks after the cap-MAD filter (pass 3);
        ``cap_ref_mAh``     — SOH reference capacity;
        ``overview``        — overview table (one row per chunk).

    Raises
    ------
    RuntimeError
        If no chunk survives pass 1, no valid capacity can be determined,
        or no chunk passes the cap-MAD filter.
    """
    anode_df = pd.read_csv(f"{anodes_dir}/{anode_name}")
    cathode_df = pd.read_csv(f"{cathodes_dir}/{cathode_name}")

    # --- Pass 1: fit + covered SOC range per chunk ---
    chunk_data: dict[int, dict] = {}
    for i, chunk in enumerate(chunks):
        pauses = extract_pauses(chunk, min_pause_duration_s=min_pause_duration_s)
        if not pauses:
            continue
        anchors = pauses_to_ocv_simple(pauses)
        df_for_fit = (
            anchors[["SOC", "U"]].rename(columns={"U": "Voltage[V]"})
                  .dropna().sort_values("SOC").drop_duplicates("SOC").reset_index(drop=True)
        )
        if len(df_for_fit) < min_points:
            continue

        fit = fit_half_cells(df_for_fit, anode_df, cathode_df,
                             anode_name=anode_name, cathode_name=cathode_name,
                             full_cell_name=f"chunk_{i:02d}")
        if fit.empty or float(fit["RMSE[mV]"].iloc[0]) > max_rmse_mv:
            continue

        soc_ref_i, u_ref_i = build_full_cell_ocv_curve(fit, anode_df, cathode_df)
        lo, hi = covered_soc_range(anchors, soc_ref_i, u_ref_i)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue

        chunk_data[i] = {"chunk": chunk, "anchors": anchors, "fit": fit,
                         "soc_ref": soc_ref_i, "u_ref": u_ref_i, "lo": lo, "hi": hi}

    if not chunk_data:
        raise RuntimeError(
            "No chunk survived pass 1 — "
            "check min_pause_duration_s / min_points / max_rmse_mv."
        )

    # Coverage filter BEFORE building the window (a narrow chunk would
    # otherwise collapse the window).
    window_ids = sorted(i for i, d in chunk_data.items()
                        if (d["hi"] - d["lo"]) >= soc_coverage_min)
    dropped_cov = sorted(set(chunk_data) - set(window_ids))
    if verbose and dropped_cov:
        print(f"Coverage filter (width < {soc_coverage_min:.2f}): "
              f"chunks {dropped_cov} excluded.")
    if len(window_ids) < 3:
        window_ids = sorted(chunk_data)
        if verbose:
            print(f"WARNING: fewer than 3 chunks above soc_coverage_min "
                  f"— coverage filter disabled for this run.")

    global_window = (max(chunk_data[i]["lo"] for i in window_ids),
                     min(chunk_data[i]["hi"] for i in window_ids))
    if verbose:
        print(f"Joint SOC window over {len(window_ids)} chunks: "
              f"{global_window[0]:.3f} .. {global_window[1]:.3f}  "
              f"(width {global_window[1] - global_window[0]:.3f})")

    # --- Pass 2: capacity per chunk in the joint window ---
    cap_per_chunk_mAh: dict[int, float] = {}
    quantities_per_chunk: dict[int, pd.DataFrame] = {}
    fits_per_chunk: dict[int, pd.DataFrame] = {}

    for i in window_ids:
        d = chunk_data[i]
        cap = estimate_capacity_covered_mAh(
            d["chunk"], d["anchors"], d["soc_ref"], d["u_ref"],
            soc_window=global_window,
            min_dsoc_pair=min_dsoc_pair,
            max_cycling_ratio=max_cycling_ratio,
        )
        if not np.isfinite(cap):
            continue
        cap_per_chunk_mAh[i] = cap
        fits_per_chunk[i] = d["fit"]
        quantities_per_chunk[i] = calculate_electrode_quantities(d["fit"], full_cap_mAh=cap)

    valid_ids = sorted(cap_per_chunk_mAh.keys())
    if verbose:
        print(f"{len(valid_ids)} chunks with a valid capacity: {valid_ids}")
    if not valid_ids:
        raise RuntimeError(
            "No valid capacity — check min_dsoc_pair / max_cycling_ratio."
        )

    # --- Pass 3: plausibility filter (capacity outliers via MAD) ---
    caps = np.array([cap_per_chunk_mAh[i] for i in valid_ids])
    cap_med = float(np.median(caps))
    cap_mad = float(np.median(np.abs(caps - cap_med)))
    if cap_mad > 0:
        cap_ok = {i: abs(cap_per_chunk_mAh[i] - cap_med) <= cap_mad_k * cap_mad
                  for i in valid_ids}
    else:
        cap_ok = {i: True for i in valid_ids}

    soh_ids = [i for i in valid_ids if cap_ok[i]]
    if verbose:
        print(f"Cap outliers (> {cap_mad_k:.0f}*MAD around median {cap_med/1000:.1f} Ah): "
              f"{sorted(set(valid_ids) - set(soh_ids))}")
        print(f"-> plausible chunks: {len(soh_ids)} {soh_ids}")
    if not soh_ids:
        raise RuntimeError("No chunk passes the cap-MAD filter — check data/filters.")

    # Robust SOH reference: median of the first plausible chunks instead of chunk 0.
    cap_ref_mAh = float(np.median([cap_per_chunk_mAh[i] for i in soh_ids[:n_ref_chunks]]))
    if verbose:
        print(f"SOH reference capacity (median of the first "
              f"{min(n_ref_chunks, len(soh_ids))} plausible chunks): {cap_ref_mAh/1000:.1f} Ah")

    overview = pd.DataFrame({
        "chunk":         valid_ids,
        "RMSE_mV":       [float(fits_per_chunk[i]["RMSE[mV]"].iloc[0]) for i in valid_ids],
        "Cap_Ah":        [cap_per_chunk_mAh[i] / 1000 for i in valid_ids],
        "SOH_pct":       [cap_per_chunk_mAh[i] / cap_ref_mAh * 100 for i in valid_ids],
        "C_Anode_Ah":    [quantities_per_chunk[i]["C_Anode_mAh"].iloc[0] / 1000 for i in valid_ids],
        "C_Cathode_Ah":  [quantities_per_chunk[i]["C_Cathode_mAh"].iloc[0] / 1000 for i in valid_ids],
        "Cap_plausible": [i in set(soh_ids) for i in valid_ids],
    }).round(2)

    return {
        "chunk_data": chunk_data,
        "global_window": global_window,
        "window_ids": window_ids,
        "cap_per_chunk_mAh": cap_per_chunk_mAh,
        "fits_per_chunk": fits_per_chunk,
        "quantities_per_chunk": quantities_per_chunk,
        "valid_ids": valid_ids,
        "soh_ids": soh_ids,
        "cap_ref_mAh": cap_ref_mAh,
        "overview": overview,
    }

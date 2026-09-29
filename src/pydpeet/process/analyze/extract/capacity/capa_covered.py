"""
Capacity and LLI/LAM over the (jointly) covered SOC range.

The existing capacity methods (``capa_curvefit``) return capacity
*per 100% SOC*: they divide the locally measured charge by the SOC span
covered and thereby implicitly extrapolate to SOC 0→1. For FUDS/field data,
however, SOC 0→1 is not covered; the reference and aged states cover
*different* SOC ranges and extrapolate differently there — this creates a
bias in SOH, LLI and LAM.

This module instead measures capacity only over the **jointly covered**
SOC range of both states. Both are then measured on the same SOC window;
the differential extrapolation is eliminated. With identical full coverage
(iOCV, SOC 0→1) the window coincides with the full range and the values
match the previous ones.

Capacity is still expressed *per 100% SOC* — it is just determined from
the anchors *within* the joint window, instead of from the entire
(differently) covered range.

Scope / limitations
--------------------
The electrode split ``C_Anode = C_full / Δa`` in
:func:`calculate_electrode_quantities` still uses the stoichiometry window
``Δa`` of the half-cell fit. On partially covered data, ``Δa`` of the
linear model is only weakly identifiable. This module fixes the
**capacity extrapolation bias**, *not* the ``Δa`` identifiability limit —
LAM therefore remains less certain than SOH/LLI.

Built entirely on existing functions
(:func:`build_full_cell_ocv_curve`, :func:`soc_from_ocv`,
:func:`estimate_capacity_curvefit_mAh`, :func:`calculate_electrode_quantities`,
:func:`calculate_lli_lam`) and does not change any other module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pydpeet.process.analyze.extract.capacity.capa_curvefit import (
    build_full_cell_ocv_curve,
    soc_from_ocv,
    estimate_capacity_curvefit_mAh,
)
from pydpeet.process.analyze.extract.degradation.lithium_amount import calculate_electrode_quantities
from pydpeet.process.analyze.extract.degradation.lli_lam import calculate_lli_lam

_STATE_KEYS = ("df", "anchors", "fit_result", "anode_df", "cathode_df")


def _sorted_ref(soc_ref: np.ndarray, u_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort and deduplicate the reference curve by ascending SOC.

    Same preparation as used internally in :func:`estimate_capacity_curvefit_mAh`,
    so the SOC values derived here stay consistent.
    """
    order = np.argsort(np.asarray(soc_ref, dtype=float))
    s = np.asarray(soc_ref, dtype=float)[order]
    v = np.asarray(u_ref, dtype=float)[order]
    keep = np.concatenate([[True], np.diff(s) > 0])
    return s[keep], v[keep]


def covered_soc_range(
    anchors: pd.DataFrame,
    soc_ref: np.ndarray,
    u_ref: np.ndarray,
    *,
    voltage_col: str = "U",
) -> tuple[float, float]:
    """
    Covered SOC range ``[s_lo, s_hi]`` of a state's anchors.

    For each anchor rest voltage, SOC is determined via the (inverted)
    reference curve; the minimum and maximum are returned.

    Parameters
    ----------
    anchors : pd.DataFrame
        Rest anchors with the voltage column ``voltage_col`` (default
        ``"U"``, from :func:`pauses_to_ocv_simple`).
    soc_ref, u_ref : np.ndarray
        Reference OCV curve SOC → U (e.g. from
        :func:`build_full_cell_ocv_curve`).
    voltage_col : str, default "U"
        Name of the rest-voltage column in ``anchors``.

    Returns
    -------
    (float, float)
        ``(s_lo, s_hi)``, or ``(nan, nan)`` if fewer than two valid anchors.

    Raises
    ------
    ValueError
        If ``voltage_col`` is missing.
    """
    if voltage_col not in anchors.columns:
        raise ValueError(f"anchors is missing column {voltage_col!r}.")
    u = anchors[voltage_col].dropna().to_numpy(dtype=float)
    if len(u) < 2:
        return (float("nan"), float("nan"))
    s_ref, v_ref = _sorted_ref(soc_ref, u_ref)
    if len(s_ref) < 2:
        return (float("nan"), float("nan"))
    soc = np.asarray(soc_from_ocv(u, s_ref, v_ref), dtype=float)
    return (float(np.min(soc)), float(np.max(soc)))


def estimate_capacity_covered_mAh(
    df: pd.DataFrame,
    anchors: pd.DataFrame,
    soc_ref: np.ndarray,
    u_ref: np.ndarray,
    *,
    soc_window: tuple[float, float] | None = None,
    time_col: str = "Test_Time[s]",
    current_col: str = "Current[A]",
    min_dsoc_pair: float = 0.02,
    max_cycling_ratio: float | None = 2.0,
    min_pairs: int = 3,
    return_details: bool = False,
):
    """
    Capacity (mAh per 100% SOC), measured only over an SOC window.

    Only anchors whose SOC (read off the reference curve) falls in
    ``soc_window`` enter the curve-shape fit. This determines capacity
    locally over the desired SOC range, instead of extrapolating over the
    entire covered range. The actual capacity calculation is done by
    :func:`estimate_capacity_curvefit_mAh`.

    Parameters
    ----------
    df : pd.DataFrame
        Time series with ``time_col`` and ``current_col``.
    anchors : pd.DataFrame
        Rest anchors with ``t_start_s``, ``t_end_s`` and ``U``.
    soc_ref, u_ref : np.ndarray
        Reference OCV curve SOC → U.
    soc_window : (float, float) or None, default None
        ``(s_lo, s_hi)`` — only use anchors with SOC in this window.
        ``None`` uses the full covered range (then identical to
        :func:`estimate_capacity_curvefit_mAh`).
    time_col, current_col, min_dsoc_pair, max_cycling_ratio, min_pairs,
    return_details
        As in :func:`estimate_capacity_curvefit_mAh`.

    Returns
    -------
    float or dict
        Capacity in mAh per 100% SOC (or NaN/NaN-dict), analogous to
        :func:`estimate_capacity_curvefit_mAh`.

    Raises
    ------
    ValueError
        If a required column is missing.
    """
    for c in ("t_start_s", "t_end_s", "U"):
        if c not in anchors.columns:
            raise ValueError(f"anchors is missing column {c!r}.")

    if soc_window is not None:
        s_lo, s_hi = float(soc_window[0]), float(soc_window[1])
        a = anchors.dropna(subset=["t_start_s", "t_end_s", "U"]).copy()
        s_ref, v_ref = _sorted_ref(soc_ref, u_ref)
        soc = np.asarray(soc_from_ocv(a["U"].to_numpy(dtype=float), s_ref, v_ref), dtype=float)
        a = a[(soc >= s_lo) & (soc <= s_hi)]
    else:
        a = anchors

    return estimate_capacity_curvefit_mAh(
        df, a, soc_ref, u_ref,
        time_col=time_col, current_col=current_col,
        min_dsoc_pair=min_dsoc_pair, max_cycling_ratio=max_cycling_ratio,
        min_pairs=min_pairs, return_details=return_details,
    )


def _prepare(state: dict, name: str) -> dict:
    """Derive the reference curve and covered SOC range of a state."""
    for k in _STATE_KEYS:
        if k not in state:
            raise ValueError(f"State {name!r} is missing key {k!r} (expected: {list(_STATE_KEYS)}).")
    soc_ref, u_ref = build_full_cell_ocv_curve(
        state["fit_result"], state["anode_df"], state["cathode_df"],
    )
    lo, hi = covered_soc_range(state["anchors"], soc_ref, u_ref)
    return {"soc_ref": soc_ref, "u_ref": u_ref, "lo": lo, "hi": hi, **state}


def lli_lam_covered(
    ref: dict,
    aged: dict,
    *,
    min_dsoc_pair: float = 0.02,
    max_cycling_ratio: float | None = 2.0,
    min_pairs: int = 3,
) -> pd.DataFrame:
    """
    SOH / LLI / LAM between two states, capacity measured only over the
    **jointly covered** SOC range.

    Procedure: for each state, the reference OCV curve is built from the
    half-cell fit and the covered SOC range is determined. The joint window
    is their intersection. Within this window, capacity (per 100% SOC) is
    measured for both states via :func:`estimate_capacity_covered_mAh`;
    from that, :func:`calculate_electrode_quantities` and
    :func:`calculate_lli_lam` yield the degradation quantities.

    Parameters
    ----------
    ref, aged : dict
        One state each (reference = fresh, aged = aged) with the keys
        ``"df"``, ``"anchors"``, ``"fit_result"``, ``"anode_df"``,
        ``"cathode_df"`` — as produced by the existing pipeline.
    min_dsoc_pair, max_cycling_ratio, min_pairs
        Forwarded to :func:`estimate_capacity_covered_mAh`.

    Returns
    -------
    pd.DataFrame
        The result of :func:`calculate_lli_lam`, extended with the columns
        ``SOC_window_lo``, ``SOC_window_hi`` (joint window) as well as
        ``SOC_covered_ref_lo/hi`` and ``SOC_covered_aged_lo/hi``. Empty
        DataFrame if no usable joint window exists or a capacity cannot be
        determined.

    Raises
    ------
    ValueError
        If a state is missing a required key.
    """
    r = _prepare(ref, "ref")
    a = _prepare(aged, "aged")

    # Joint SOC window = intersection of the covered ranges
    s_lo = max(r["lo"], a["lo"])
    s_hi = min(r["hi"], a["hi"])
    if not (np.isfinite(s_lo) and np.isfinite(s_hi)) or (s_hi - s_lo) < min_dsoc_pair:
        return pd.DataFrame()

    window = (s_lo, s_hi)
    kw = dict(min_dsoc_pair=min_dsoc_pair, max_cycling_ratio=max_cycling_ratio, min_pairs=min_pairs)
    cap_ref = estimate_capacity_covered_mAh(r["df"], r["anchors"], r["soc_ref"], r["u_ref"],
                                            soc_window=window, **kw)
    cap_aged = estimate_capacity_covered_mAh(a["df"], a["anchors"], a["soc_ref"], a["u_ref"],
                                             soc_window=window, **kw)
    if not (np.isfinite(cap_ref) and np.isfinite(cap_aged)):
        return pd.DataFrame()

    q_ref = calculate_electrode_quantities(r["fit_result"], full_cap_mAh=cap_ref)
    q_aged = calculate_electrode_quantities(a["fit_result"], full_cap_mAh=cap_aged)
    res = calculate_lli_lam(q_ref, q_aged)

    return res.assign(
        SOC_window_lo=s_lo,
        SOC_window_hi=s_hi,
        SOC_covered_ref_lo=r["lo"],
        SOC_covered_ref_hi=r["hi"],
        SOC_covered_aged_lo=a["lo"],
        SOC_covered_aged_hi=a["hi"],
    )

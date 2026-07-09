"""
Kapazitätsbestimmung per Kurvenform-Fit (statt 2-Punkt-Extrapolation).

Die 2-Punkt-Methode (``capa_real_ocv.estimate_capacity_anchored_mAh``)
bestimmt die Kapazität aus dem Ladungsdurchsatz zwischen zwei OCV-Ankern
geteilt durch deren SOC-Differenz — die gesamte Aussage hängt also an zwei
Spannungswerten und ist in flachen OCV-Bereichen entsprechend verrauscht.

Dieses Modul nutzt stattdessen *alle* Anker und die Referenz-OCV-Kurve. Für
jedes Paar aufeinanderfolgender Ruhepausen liefert die Kurve

    ΔSOC_k = soc(U_{k+1}) − soc(U_k)

und aus dem Strom das *lokale* Ladungsinkrement ΔQ_k = ∫ I dt zwischen den
beiden Pausen. Die Kapazität ist die Steigung in ΔQ = C · ΔSOC, robust über
alle Paare gefittet (Least-Squares durch den Ursprung, nach Hub-Größe
gewichtet, Ausreißer-Paare per MAD verworfen).

Bewusst *lokale* Inkremente statt kumulativem Q über den ganzen Datensatz:
Letzteres driftet bei vielen Lade-/Entladezyklen durch Stromoffset weg und
lässt die Kapazität explodieren. So funktioniert die Methode für einzelne
Sweeps (Labor-iOCV) wie für Mehrzyklen-Chunks (Felddaten). Die 2-Punkt-Methode
ist der Spezialfall mit einem einzigen Anker-Paar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


# --- Referenzkennlinie aus dem Halbzellen-Fit -------------------------------

_FIT_COLUMNS = ("Sol_Anode_Min", "Sol_Anode_Max",
                "Sol_Cathode_Min", "Sol_Cathode_Max")


def _detect_y_col(df: pd.DataFrame, x_col: str) -> str:
    """Return the voltage column of an electrode reference table."""
    for candidate in ("OCV", "Voltage", "Voltage[V]", "V", "U"):
        if candidate in df.columns:
            return candidate
    cols = [c for c in df.columns if c != x_col]
    return cols[0] if cols else df.columns[-1]


def _detect_x_col(df: pd.DataFrame) -> str:
    """Return the lithiation column of an electrode reference table."""
    if "Lithiation" in df.columns:
        return "Lithiation"
    return df.columns[0]


def build_full_cell_ocv_curve(
    fit_result: pd.DataFrame,
    anode_df: pd.DataFrame,
    cathode_df: pd.DataFrame,
    *,
    n_grid: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the full-cell open-circuit voltage ``U_full(SOC)`` from a
    half-cell fit.

    Parameters
    ----------
    fit_result : pd.DataFrame
        Single-row DataFrame from :func:`fit_half_cells` /
        :func:`get_best_half_cell_fit`. The stoichiometric window
        ``Sol_Anode_Min / Max`` and ``Sol_Cathode_Min / Max`` is read
        from the first row.
    anode_df, cathode_df : pd.DataFrame
        The same reference tables used for the fit (lithiation as the
        ``x`` column, half-cell voltage as the ``y`` column).
    n_grid : int, default 500
        Number of points on the uniform SOC grid in ``[0, 1]``. 500 is
        enough for sub-mV interpolation accuracy.

    Returns
    -------
    soc_grid : np.ndarray
        SOC samples in ``[0, 1]``, monotonically increasing.
    u_grid : np.ndarray
        Corresponding full-cell OCV in V, monotonically increasing.

    Raises
    ------
    ValueError
        If ``fit_result`` is empty or missing the stoichiometry
        columns.
    """
    if fit_result.empty:
        raise ValueError("fit_result is empty — provide a half-cell fit row.")
    missing = [c for c in _FIT_COLUMNS if c not in fit_result.columns]
    if missing:
        raise ValueError(f"fit_result is missing column(s) {missing}.")

    row = fit_result.iloc[0]
    a_min = float(row["Sol_Anode_Min"])
    a_max = float(row["Sol_Anode_Max"])
    c_min = float(row["Sol_Cathode_Min"])
    c_max = float(row["Sol_Cathode_Max"])

    an_x = _detect_x_col(anode_df)
    an_y = _detect_y_col(anode_df, an_x)
    ca_x = _detect_x_col(cathode_df)
    ca_y = _detect_y_col(cathode_df, ca_x)

    an_clean = (anode_df[[an_x, an_y]].dropna()
                .sort_values(an_x).drop_duplicates(subset=[an_x]))
    ca_clean = (cathode_df[[ca_x, ca_y]].dropna()
                .sort_values(ca_x).drop_duplicates(subset=[ca_x]))

    an_spline = CubicSpline(an_clean[an_x].to_numpy(float),
                            an_clean[an_y].to_numpy(float))
    ca_spline = CubicSpline(ca_clean[ca_x].to_numpy(float),
                            ca_clean[ca_y].to_numpy(float))

    an_lo, an_hi = float(an_clean[an_x].min()), float(an_clean[an_x].max())
    ca_lo, ca_hi = float(ca_clean[ca_x].min()), float(ca_clean[ca_x].max())

    soc_grid = np.linspace(0.0, 1.0, n_grid)
    lith_a = np.clip(a_min + soc_grid * (a_max - a_min), an_lo, an_hi)
    lith_c = np.clip(c_max - soc_grid * (c_max - c_min), ca_lo, ca_hi)
    u_grid = ca_spline(lith_c) - an_spline(lith_a)

    if not np.all(np.diff(u_grid) > 0):
        order = np.argsort(u_grid)
        u_grid = u_grid[order]
        soc_grid = soc_grid[order]
        _, unique = np.unique(u_grid, return_index=True)
        u_grid = u_grid[unique]
        soc_grid = soc_grid[unique]

    return soc_grid, u_grid


def soc_from_ocv(
    u,
    soc_grid: np.ndarray,
    u_grid: np.ndarray,
) -> np.ndarray:
    u_arr = np.atleast_1d(np.asarray(u, dtype=float))
    return np.interp(u_arr, u_grid, soc_grid)


# --- Kurvenform-Kapazität ----------------------------------------------------


def _cumulative_ah(t: np.ndarray, i: np.ndarray) -> np.ndarray:
    """Kumulatives Ladungsintegral [Ah] an jedem Zeitpunkt (Trapezregel)."""
    dq = (t[1:] - t[:-1]) * (i[1:] + i[:-1]) / 2.0
    return np.concatenate([[0.0], np.cumsum(dq)]) / 3600.0


def _nan_result(n: int, return_details: bool):
    if not return_details:
        return float("nan")
    return {"capacity_mAh": float("nan"), "n_pairs": int(n), "soc_span": 0.0}


def estimate_capacity_curvefit_mAh(
    df: pd.DataFrame,
    anchors: pd.DataFrame,
    soc_ref: np.ndarray,
    u_ref: np.ndarray,
    *,
    time_col: str = "Test_Time[s]",
    current_col: str = "Current[A]",
    min_dsoc_pair: float = 0.02,
    max_cycling_ratio: float | None = 2.0,
    min_pairs: int = 3,
    return_details: bool = False,
):
    """
    Kapazität (mAh pro 100 % SOC) per kurvenbasiertem Multi-Punkt-Fit.

    Driftfrei: nutzt das Ladungsinkrement ``ΔQ`` *nur zwischen aufeinander-
    folgenden Ruhepausen* (nicht kumulativ über den ganzen Datensatz, was bei
    vielen Lade-/Entladezyklen durch Stromoffset wegdriften würde und die
    Kapazität explodieren ließe). Für jedes Paar liefert die Referenzkurve
    ``ΔSOC = soc(U_{k+1}) − soc(U_k)``; die Kapazität wird robust über alle
    Paare gefittet (Least-Squares durch den Ursprung, nach Hub-Größe
    gewichtet, Ausreißer-Paare per MAD verworfen). Funktioniert für einzelne
    Sweeps (Labor-iOCV) wie für Mehrzyklen-Chunks (Felddaten).

    Parameters
    ----------
    df : pd.DataFrame
        Zeitreihe mit ``time_col`` und ``current_col``.
    anchors : pd.DataFrame
        Ruhe-Anker mit ``t_start_s``, ``t_end_s`` und ``U`` (aus
        ``pauses_to_ocv_simple``).
    soc_ref, u_ref : np.ndarray
        Referenz-OCV-Kurve SOC → U (z. B. aus ``build_full_cell_ocv_curve``).
    time_col, current_col : str
        Spaltennamen in ``df``.
    min_dsoc_pair : float, default 0.02
        Mindest-SOC-Hub eines Paars; kleinere Hübe sind reines Rauschen.
    max_cycling_ratio : float or None, default 2.0
        Paare verwerfen, bei denen ``∫|I| dt`` zwischen den Pausen
        ``max_cycling_ratio · |∫ I dt|`` übersteigt (zu viel Hin-und-Her).
        ``None`` deaktiviert den Filter.
    min_pairs : int, default 3
        Weniger gültige Paare → NaN.
    return_details : bool, default False
        Wenn True, dict mit ``capacity_mAh``, ``n_pairs``, ``soc_span``.

    Returns
    -------
    float oder dict
        Kapazität in mAh pro 100 % SOC, oder NaN/NaN-dict.

    Raises
    ------
    ValueError
        Wenn eine erforderliche Spalte fehlt.
    """
    for c in (time_col, current_col):
        if c not in df.columns:
            raise ValueError(f"df fehlt Spalte {c!r}.")
    for c in ("t_start_s", "t_end_s", "U"):
        if c not in anchors.columns:
            raise ValueError(f"anchors fehlt Spalte {c!r}.")

    a = anchors.dropna(subset=["t_start_s", "t_end_s", "U"]).sort_values("t_end_s").reset_index(drop=True)
    if len(a) < min_pairs + 1:
        return _nan_result(len(a), return_details)

    # Referenzkurve nach SOC aufsteigend sortieren & deduplizieren
    order = np.argsort(np.asarray(soc_ref, float))
    s_ref = np.asarray(soc_ref, float)[order]
    v_ref = np.asarray(u_ref, float)[order]
    keep = np.concatenate([[True], np.diff(s_ref) > 0])
    s_ref, v_ref = s_ref[keep], v_ref[keep]
    if len(s_ref) < 2:
        return _nan_result(len(a), return_details)

    soc_a = np.asarray(soc_from_ocv(a["U"].to_numpy(float), s_ref, v_ref), float)

    d = df.dropna(subset=[time_col, current_col]).sort_values(time_col)
    t = d[time_col].to_numpy(float)
    i = d[current_col].to_numpy(float)
    if len(t) < 2:
        return _nan_result(len(a), return_details)
    q_cum = _cumulative_ah(t, i)
    qabs_cum = _cumulative_ah(t, np.abs(i))

    te = a["t_end_s"].to_numpy(float)
    ts = a["t_start_s"].to_numpy(float)
    dS, dQ = [], []
    # Akkumulierende Paare: ueber aufeinanderfolgende Anker summieren, bis der
    # SOC-Hub min_dsoc_pair erreicht ist. So fallen die vielen winzigen Stufen
    # einer iOCV/FUDS-Sequenz nicht durchs Raster, und grosse Feld-Spruenge
    # bilden je ein Paar. Die Spanne bleibt lokal (driftfrei).
    ref = 0
    for k in range(1, len(a)):
        ds = float(soc_a[k] - soc_a[ref])
        if abs(ds) < min_dsoc_pair:                       # noch nicht genug Hub
            continue
        t_lo, t_hi = te[ref], ts[k]
        if t_hi > t_lo:
            dq = float(np.interp(t_hi, t, q_cum) - np.interp(t_lo, t, q_cum))
            ok = True
            if max_cycling_ratio is not None and abs(dq) > 0:
                dqa = float(np.interp(t_hi, t, qabs_cum) - np.interp(t_lo, t, qabs_cum))
                if dqa > max_cycling_ratio * abs(dq):     # zu viel Hin-und-Her
                    ok = False
            if ok:
                dS.append(abs(ds))
                dQ.append(abs(dq))
        ref = k                                           # neuen Referenz-Anker setzen

    if len(dS) < min_pairs:
        return _nan_result(len(dS), return_details)

    dS = np.asarray(dS)
    dQ = np.asarray(dQ)
    ratio = dQ / dS
    med = float(np.median(ratio))
    mad = float(np.median(np.abs(ratio - med))) or 1e-12
    m = np.abs(ratio - med) <= 3.0 * mad                  # Ausreißer-Paare verwerfen
    C = float(np.sum(dS[m] * dQ[m]) / np.sum(dS[m] * dS[m]))   # Ah/100%SOC, hub-gewichtet
    cap = abs(C) * 1000.0

    if not return_details:
        return cap
    return {"capacity_mAh": cap, "n_pairs": int(m.sum()),
            "soc_span": float(np.max(soc_a) - np.min(soc_a))}


def estimate_capacity_curvefit_from_fit_mAh(
    df: pd.DataFrame,
    anchors: pd.DataFrame,
    fit_result: pd.DataFrame,
    anode_df: pd.DataFrame,
    cathode_df: pd.DataFrame,
    **kwargs,
):
    """
    Wie :func:`estimate_capacity_curvefit_mAh`, aber die Referenz-OCV-Kurve
    wird aus einem Halbzellen-Fit gebaut (:func:`build_full_cell_ocv_curve`).

    Praktisch für die Felddaten-Pipeline: ``fit_result`` ist der per-Chunk-Fit,
    ``anode_df``/``cathode_df`` die zugehörigen Referenz-Elektroden.
    """
    soc_ref, u_ref = build_full_cell_ocv_curve(fit_result, anode_df, cathode_df)
    return estimate_capacity_curvefit_mAh(df, anchors, soc_ref, u_ref, **kwargs)

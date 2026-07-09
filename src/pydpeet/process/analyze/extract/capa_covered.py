"""
Kapazität und LLI/LAM über den (gemeinsam) abgedeckten SOC-Bereich.

Die bestehenden Kapazitätsmethoden (``capa_real_ocv``, ``capa_curvefit``) geben
die Kapazität *pro 100 % SOC* zurück: Sie teilen die lokal gemessene Ladung
durch die durchfahrene SOC-Spanne und extrapolieren damit implizit auf
SOC 0→1. Bei FUDS-/Felddaten wird SOC 0→1 aber nicht durchfahren; Referenz- und
gealterter Zustand decken *unterschiedliche* SOC-Bereiche ab und extrapolieren
dort verschieden — das erzeugt einen Bias in SOH, LLI und LAM.

Dieses Modul misst die Kapazität stattdessen nur über den **gemeinsam
abgedeckten** SOC-Bereich beider Zustände. Beide werden so auf demselben
SOC-Fenster gemessen; die differenzielle Extrapolation entfällt. Bei
identischer voller Abdeckung (iOCV, SOC 0→1) fällt das Fenster mit dem vollen
Bereich zusammen und die Werte entsprechen den bisherigen.

Die Kapazität wird weiterhin *pro 100 % SOC* ausgedrückt — sie wird lediglich
aus den Ankern *innerhalb* des gemeinsamen Fensters bestimmt, statt aus dem
gesamten (unterschiedlich) befahrenen Bereich.

Grenze / Einordnung
-------------------
Die Elektroden-Aufteilung ``C_Anode = C_full / Δa`` in
:func:`calculate_electrode_quantities` nutzt weiterhin das Stöchiometrie-Fenster
``Δa`` des Halbzellen-Fits. Auf teilweise abgedeckten Daten ist ``Δa`` des
linearen Modells nur schwach identifizierbar. Dieses Modul behebt den
**Kapazitäts-Extrapolationsbias**, *nicht* die ``Δa``-Identifizierbarkeitsgrenze —
LAM bleibt entsprechend unsicherer als SOH/LLI.

Baut ausschließlich auf bestehenden Funktionen auf
(:func:`build_full_cell_ocv_curve`, :func:`soc_from_ocv`,
:func:`estimate_capacity_curvefit_mAh`, :func:`calculate_electrode_quantities`,
:func:`calculate_lli_lam`) und ändert keine anderen Module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pydpeet.process.analyze.extract.capa_curvefit import (
    build_full_cell_ocv_curve,
    soc_from_ocv,
    estimate_capacity_curvefit_mAh,
)
from pydpeet.process.analyze.extract.lithium_amount import calculate_electrode_quantities
from pydpeet.process.analyze.extract.lli_lam import calculate_lli_lam

_STATE_KEYS = ("df", "anchors", "fit_result", "anode_df", "cathode_df")


def _sorted_ref(soc_ref: np.ndarray, u_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Referenzkurve nach SOC aufsteigend sortieren & deduplizieren.

    Gleiche Aufbereitung wie intern in :func:`estimate_capacity_curvefit_mAh`,
    damit die hier abgeleiteten SOC-Werte konsistent sind.
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
    Abgedeckter SOC-Bereich ``[s_lo, s_hi]`` der Anker eines Zustands.

    Für jede Anker-Ruhespannung wird der SOC über die (invertierte)
    Referenzkurve bestimmt; zurückgegeben werden das Minimum und Maximum.

    Parameters
    ----------
    anchors : pd.DataFrame
        Ruhe-Anker mit der Spannungsspalte ``voltage_col`` (Default ``"U"``,
        aus :func:`pauses_to_ocv_simple`).
    soc_ref, u_ref : np.ndarray
        Referenz-OCV-Kurve SOC → U (z. B. aus
        :func:`build_full_cell_ocv_curve`).
    voltage_col : str, default "U"
        Name der Ruhespannungsspalte in ``anchors``.

    Returns
    -------
    (float, float)
        ``(s_lo, s_hi)``, oder ``(nan, nan)`` bei weniger als zwei gültigen
        Ankern.

    Raises
    ------
    ValueError
        Wenn ``voltage_col`` fehlt.
    """
    if voltage_col not in anchors.columns:
        raise ValueError(f"anchors fehlt Spalte {voltage_col!r}.")
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
    Kapazität (mAh pro 100 % SOC), gemessen nur über ein SOC-Fenster.

    Nur Anker, deren (aus der Referenzkurve abgelesener) SOC in
    ``soc_window`` liegt, gehen in den Kurvenform-Fit ein. Damit wird die
    Kapazität lokal über den gewünschten SOC-Bereich bestimmt, statt über den
    gesamten befahrenen Bereich extrapoliert. Die eigentliche Kapazitäts-
    berechnung erledigt :func:`estimate_capacity_curvefit_mAh`.

    Parameters
    ----------
    df : pd.DataFrame
        Zeitreihe mit ``time_col`` und ``current_col``.
    anchors : pd.DataFrame
        Ruhe-Anker mit ``t_start_s``, ``t_end_s`` und ``U``.
    soc_ref, u_ref : np.ndarray
        Referenz-OCV-Kurve SOC → U.
    soc_window : (float, float) or None, default None
        ``(s_lo, s_hi)`` — nur Anker mit SOC in diesem Fenster verwenden.
        ``None`` verwendet den vollen befahrenen Bereich (dann identisch zu
        :func:`estimate_capacity_curvefit_mAh`).
    time_col, current_col, min_dsoc_pair, max_cycling_ratio, min_pairs,
    return_details
        Wie in :func:`estimate_capacity_curvefit_mAh`.

    Returns
    -------
    float oder dict
        Kapazität in mAh pro 100 % SOC (bzw. NaN/NaN-dict), analog zu
        :func:`estimate_capacity_curvefit_mAh`.

    Raises
    ------
    ValueError
        Wenn eine erforderliche Spalte fehlt.
    """
    for c in ("t_start_s", "t_end_s", "U"):
        if c not in anchors.columns:
            raise ValueError(f"anchors fehlt Spalte {c!r}.")

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
    """Referenzkurve und abgedeckten SOC-Bereich eines Zustands ableiten."""
    for k in _STATE_KEYS:
        if k not in state:
            raise ValueError(f"Zustand {name!r} fehlt Key {k!r} (erwartet: {list(_STATE_KEYS)}).")
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
    SOH / LLI / LAM zwischen zwei Zuständen, Kapazität nur über den
    **gemeinsam abgedeckten** SOC-Bereich gemessen.

    Ablauf: Für jeden Zustand wird die Referenz-OCV-Kurve aus dem Halbzellen-Fit
    gebaut und der befahrene SOC-Bereich bestimmt. Das gemeinsame Fenster ist
    deren Schnittmenge. In diesem Fenster wird für beide Zustände die Kapazität
    (pro 100 % SOC) via :func:`estimate_capacity_covered_mAh` gemessen; daraus
    folgen über :func:`calculate_electrode_quantities` und
    :func:`calculate_lli_lam` die Degradationsgrößen.

    Parameters
    ----------
    ref, aged : dict
        Je ein Zustand (Referenz = frisch, aged = gealtert) mit den Keys
        ``"df"``, ``"anchors"``, ``"fit_result"``, ``"anode_df"``,
        ``"cathode_df"`` — wie sie auch die bestehende Pipeline erzeugt.
    min_dsoc_pair, max_cycling_ratio, min_pairs
        An :func:`estimate_capacity_covered_mAh` durchgereicht.

    Returns
    -------
    pd.DataFrame
        Das Ergebnis von :func:`calculate_lli_lam`, ergänzt um die Spalten
        ``SOC_window_lo``, ``SOC_window_hi`` (gemeinsames Fenster) sowie
        ``SOC_covered_ref_lo/hi`` und ``SOC_covered_aged_lo/hi``. Leerer
        DataFrame, wenn kein verwertbares gemeinsames Fenster existiert oder
        eine Kapazität nicht bestimmbar ist.

    Raises
    ------
    ValueError
        Wenn einem Zustand ein erforderlicher Key fehlt.
    """
    r = _prepare(ref, "ref")
    a = _prepare(aged, "aged")

    # Gemeinsames SOC-Fenster = Schnittmenge der befahrenen Bereiche
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

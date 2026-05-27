"""
Simple OCV extraction from a list of rest pauses — no relaxation fit.

The OCV anchor for each pause is taken as the **last measured sample** of
that pause: ``U_ocv = Voltage[V].iloc[-1]`` paired with ``SOC.iloc[-1]``.
This is the cheap baseline against which the relaxation-extrapolation
approach in
:mod:`pydpeet.process.analyze.extract.pause_to_ocv` should be compared.

For long laboratory rest pauses (≥ 30 min) the endpoint is practically
identical to the true OCV — the simple method is then sufficient. For
short pauses (seconds to minutes, as in FUDS or field data) the endpoint
is still relaxing and will bias subsequent half-cell fits; that is exactly
the case the relaxation extrapolation was built for.

The output schema matches
:func:`pydpeet.process.analyze.extract.pause_to_ocv.pauses_to_ocv` and
:func:`pydpeet.process.analyze.extract.relaxation.extract_relaxation_anchors`,
so the downstream pipeline (weighting, half-cell fit) does not need to
know which anchor source was used. Columns that have no meaning for the
endpoint method (``U_inf_std``, ``rmse``) are set to NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pauses_to_ocv_simple(
    pauses: list[pd.DataFrame],
    *,
    min_samples_per_pause: int = 2,
    time_col: str = "Test_Time[s]",
    voltage_col: str = "Voltage[V]",
    soc_col: str = "SOC",
) -> pd.DataFrame:
    """
    Endpoint-based OCV anchors: one row per pause, voltage and SOC of the
    last sample.

    Parameters
    ----------
    pauses : list of pd.DataFrame
        Output of :func:`extract_pauses`, or any list of rest-only slices.
        Each pause must contain ``time_col`` and ``voltage_col``;
        ``soc_col`` is optional (missing → NaN in output).
    min_samples_per_pause : int, default 2
        Pauses shorter than this are skipped — at least two samples are
        needed to report a meaningful start/end time.
    time_col, voltage_col, soc_col : str
        Column names inside each pause DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per qualifying pause with columns:

        =================== ====================================================
        pause_idx            Sequential index in the input list
        t_start_s, t_end_s   Test time of pause start / end
        pause_duration_s     End − start, in seconds
        n_samples            Number of samples in the pause
        SOC                  SOC at the last sample (NaN if column absent)
        U_inf                Voltage at the last sample, in V (= the OCV anchor)
        U_inf_std            NaN — no uncertainty for the endpoint method
        rmse                 NaN — no fit residual
        converged            ``True`` (the endpoint is always defined)
        model                ``"endpoint"``
        warnings             Empty string
        =================== ====================================================

    Raises
    ------
    ValueError
        If a pause is missing a required column or its ``time_col`` is not
        strictly increasing.
    """
    rows: list[dict] = []

    for k, p in enumerate(pauses):
        for c in (time_col, voltage_col):
            if c not in p.columns:
                raise ValueError(
                    f"Pause #{k} missing required column {c!r}."
                )
        if len(p) < min_samples_per_pause:
            continue

        p_sorted = p.sort_values(time_col)
        t = p_sorted[time_col].to_numpy(dtype=float)

        if np.any(np.diff(t) <= 0):
            raise ValueError(
                f"Pause #{k}: {time_col!r} must be strictly increasing."
            )

        u_end = float(p_sorted[voltage_col].iloc[-1])
        soc_end = (
            float(p_sorted[soc_col].iloc[-1])
            if soc_col in p_sorted.columns
            else float("nan")
        )

        rows.append({
            "pause_idx":        int(k),
            "t_start_s":        float(t[0]),
            "t_end_s":          float(t[-1]),
            "pause_duration_s": float(t[-1] - t[0]),
            "n_samples":        int(len(p_sorted)),
            "SOC":              soc_end,
            "U":                u_end,
            "U_std":        float("nan"),
            "rmse":             float("nan"),
            "converged":        True,
            "model":            "endpoint",
            "warnings":         "",
        })

    return pd.DataFrame(rows)

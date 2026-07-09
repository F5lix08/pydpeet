import numpy as np
import pandas as pd
from scipy import integrate

# Faraday-Konstante in mAh/mol
_FARADAY_mAh_per_mol = 26_800.0


def extract_full_cap_mAh(df: pd.DataFrame) -> float:
    """
    Extract the full-cell discharge capacity in mAh from a processed test DataFrame.

    Two strategies are tried in order:

    1. **Column-based** – if a ``Capacity[Ah]`` column exists and contains
       non-NaN values the maximum value is returned directly (assumes
       ``add_capacity`` has already been run).

    2. **Integration fallback** – if the column is absent or all-NaN the
       discharge current is integrated over time via the trapezoidal rule,
       separately for every contiguous discharge segment (run of consecutive
       samples with ``Current[A] < 0``); the largest segment capacity is
       returned. Integrating per segment keeps the trapezoidal rule from
       bridging the time gaps between separate discharge phases (e.g. iOCV
       pulses) with phantom charge, and the maximum picks the full
       discharge of the capacity test rather than the sum of all pulses.

    Parameters
    ----------
    df : pd.DataFrame
        Processed test DataFrame.  Must contain at least ``Current[A]`` and
        ``Test_Time[s]`` for the fallback path.

    Returns
    -------
    float
        Full-cell discharge capacity in mAh.

    Raises
    ------
    ValueError
        If neither strategy can produce a result.
    """
    # Strategy 1: pre-computed Capacity[Ah] column
    if "Capacity[Ah]" in df.columns:
        cap_series = df["Capacity[Ah]"].dropna()
        if not cap_series.empty:
            return float(cap_series.abs().max() * 1000)

    # Strategy 2: integrate discharge current
    required = {"Current[A]", "Test_Time[s]"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Cannot extract capacity: column(s) missing: "
            f"{missing}.  Either provide a DataFrame with a "
            "'Capacity[Ah]' column or with 'Current[A]' and 'Test_Time[s]'."
        )

    d = (df.dropna(subset=["Current[A]", "Test_Time[s]"])
           .sort_values("Test_Time[s]")
           .reset_index(drop=True))
    time_s = d["Test_Time[s]"].values.astype(float)
    current_a = d["Current[A]"].values.astype(float)

    is_discharge = current_a < 0
    if not is_discharge.any():
        raise ValueError(
            "Cannot extract capacity: no discharge samples (Current[A] < 0) found."
        )

    # Contiguous discharge segments: integrate each separately, take the
    # largest (= the full discharge of the capacity test).
    transitions = np.flatnonzero(np.diff(is_discharge.astype(np.int8))) + 1
    seg_starts = np.concatenate([[0], transitions])
    seg_ends = np.concatenate([transitions, [len(is_discharge)]])  # exclusive

    cap_ah = 0.0
    for s, e in zip(seg_starts, seg_ends):
        if not is_discharge[s] or (e - s) < 2:
            continue
        seg_ah = integrate.trapezoid(np.abs(current_a[s:e]), time_s[s:e]) / 3600.0
        cap_ah = max(cap_ah, seg_ah)

    if cap_ah == 0.0:
        raise ValueError(
            "Cannot extract capacity: no discharge segment with at least "
            "two samples found."
        )
    return float(cap_ah * 1000)


def calculate_electrode_quantities(
    fit_result: pd.DataFrame,
    full_cap_mAh: float = None,
    df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Calculate lithium inventory and electrode capacities from half-cell fitting results.

    Uses the stoichiometric window [a_min, a_max] / [c_min, c_max] determined by
    ``fit_half_cells`` together with the measured full-cell discharge capacity to
    derive absolute electrode capacities, the active lithium inventory, and the
    lithium stored in each electrode at the two SOC extremes.

    The capacity can be supplied in two ways — exactly one must be given:

    * **Explicit**: pass a float via ``full_cap_mAh``.
    * **Automatic**: pass the processed test DataFrame via ``df``; the capacity
      is then extracted automatically via :func:`extract_full_cap_mAh`.

    Parameters
    ----------
    fit_result : pd.DataFrame
        Single-row DataFrame returned by ``fit_half_cells`` or
        ``get_best_half_cell_fit``.  Must contain the columns
        ``Sol_Anode_Min``, ``Sol_Anode_Max``, ``Sol_Cathode_Min``,
        ``Sol_Cathode_Max``.
    full_cap_mAh : float, optional
        Measured full-cell discharge capacity in mAh.
    df : pd.DataFrame, optional
        Processed test DataFrame from which the capacity is extracted
        automatically (requires ``Capacity[Ah]`` or ``Current[A]`` +
        ``Test_Time[s]``).

    Returns
    -------
    pd.DataFrame
        One row per input row with the following columns:

        ========================== =============================================
        Column                     Description
        ========================== =============================================
        Full_Cell_Cap_mAh          Capacity used for the calculation
        C_Anode_mAh                Nominal anode capacity = C_full / Δlith_a
        C_Cathode_mAh              Nominal cathode capacity = C_full / Δlith_c
        NP_Ratio                   C_Anode / C_Cathode
        Li_Inventory_mAh           Lithium inventory from the electrode
                                   balance = C_Anode·a_min + C_Cathode·c_max
                                   (total Li in both electrodes at SOC 0;
                                   identical at SOC 100)
        Li_Inventory_mol           Lithium inventory in mol
        Anode_Lith_SOC0            Anode lithiation at SOC = 0 % (a_min)
        Anode_Lith_SOC100          Anode lithiation at SOC = 100 % (a_max)
        Cathode_Lith_SOC0          Cathode lithiation at SOC = 0 % (c_max)
        Cathode_Lith_SOC100        Cathode lithiation at SOC = 100 % (c_min)
        Li_Anode_SOC0_mAh          Li stored in anode at SOC = 0 %
        Li_Anode_SOC100_mAh        Li stored in anode at SOC = 100 %
        Li_Cathode_SOC0_mAh        Li stored in cathode at SOC = 0 %
        Li_Cathode_SOC100_mAh      Li stored in cathode at SOC = 100 %
        ========================== =============================================
    """
    if fit_result.empty:
        raise ValueError("fit_result is empty — run fit_half_cells first.")
    if full_cap_mAh is None and df is None:
        raise ValueError(
            "Provide either 'full_cap_mAh' (float) or 'df' (test DataFrame)."
        )
    if full_cap_mAh is not None and df is not None:
        raise ValueError(
            "Provide either 'full_cap_mAh' or 'df', not both."
        )

    if df is not None:
        full_cap_mAh = extract_full_cap_mAh(df)

    rows = []
    for _, row in fit_result.iterrows():
        a_min = float(row["Sol_Anode_Min"])
        a_max = float(row["Sol_Anode_Max"])
        c_min = float(row["Sol_Cathode_Min"])
        c_max = float(row["Sol_Cathode_Max"])

        delta_a = a_max - a_min
        delta_c = c_max - c_min

        if delta_a <= 0 or delta_c <= 0:
            raise ValueError(
                "Lithiation windows must be positive. "
                f"Got Δlith_a={delta_a:.4f}, Δlith_c={delta_c:.4f}."
            )

        C_anode   = full_cap_mAh / delta_a
        C_cathode = full_cap_mAh / delta_c

        # Lithium inventory from the electrode balance: total lithium held
        # in both electrodes at one full-cell state (here SOC 0). By charge
        # conservation the same value results at SOC 100. This is NOT the
        # full-cell capacity: it additionally counts the lithium that stays
        # in the electrodes at the SOC extremes (a_min > 0, c_min > 0).
        Li_inventory_mAh = a_min * C_anode + c_max * C_cathode
        Li_inventory_mol = Li_inventory_mAh / _FARADAY_mAh_per_mol

        rows.append({
            # ── full cell ────────────────────────────────────────────────
            "Full_Cell_Cap_mAh":     full_cap_mAh,
            # ── electrode capacities ─────────────────────────────────────
            "C_Anode_mAh":           C_anode,
            "C_Cathode_mAh":         C_cathode,
            "NP_Ratio":              C_anode / C_cathode,
            # ── lithium inventory ────────────────────────────────────────
            "Li_Inventory_mAh":      Li_inventory_mAh,
            "Li_Inventory_mol":      Li_inventory_mol,
            # ── stoichiometric endpoints ─────────────────────────────────
            "Anode_Lith_SOC0":       a_min,
            "Anode_Lith_SOC100":     a_max,
            "Cathode_Lith_SOC0":     c_max,
            "Cathode_Lith_SOC100":   c_min,
            # ── absolute lithium per electrode at SOC extremes ───────────
            "Li_Anode_SOC0_mAh":     a_min * C_anode,
            "Li_Anode_SOC100_mAh":   a_max * C_anode,
            "Li_Cathode_SOC0_mAh":   c_max * C_cathode,
            "Li_Cathode_SOC100_mAh": c_min * C_cathode,
        })

    return pd.DataFrame(rows)

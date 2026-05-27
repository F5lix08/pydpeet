"""
LLI / LAM calculation from half-cell fitting results.

NOTE: this file is named ``lli+lam.py`` for historical reasons.  Because the
``+`` character is not valid in a Python module name it cannot be imported with
a standard ``import`` statement.  Rename the file to ``lli_lam.py`` to use it
as a regular module.

Terminology
-----------
LLI  – Loss of Lithium Inventory
         Reduction in the amount of cyclable lithium between a reference and a
         current measurement.  Caused by SEI growth, lithium plating, etc.

LAM_Anode    – Loss of Active Material (Anode)
LAM_Cathode  – Loss of Active Material (Cathode)
         Reduction in the accessible electrode capacity.  Caused by particle
         cracking, binder decomposition, contact loss, etc.

All three quantities are calculated by comparing the output of
``calculate_electrode_quantities`` (from ``lithium_amount.py``) at two
different aging states.
"""

import pandas as pd


def calculate_lli_lam(
    ref_quantities: pd.DataFrame,
    aged_quantities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate LLI and LAM by comparing aged states to a reference state.

    Parameters
    ----------
    ref_quantities : pd.DataFrame
        Single-row DataFrame from ``calculate_electrode_quantities`` representing
        the reference / fresh-cell state (e.g. Checkup 1 / BOL).
    aged_quantities : pd.DataFrame
        One or more rows from ``calculate_electrode_quantities`` representing
        later aging states (e.g. Checkup 2, 3, …).  Each row is compared
        independently against the single reference row.

    Returns
    -------
    pd.DataFrame
        One row per aged state with the following columns:

        ========================== ==============================================
        Column                     Description
        ========================== ==============================================
        Full_Cell_Cap_ref_mAh      Reference cell capacity
        Full_Cell_Cap_mAh          Current cell capacity
        SOH_pct                    State of Health  = Cap_now / Cap_ref × 100
        LLI_mAh                    Absolute LLI in mAh
        LLI_pct                    Relative LLI in %
        LAM_Anode_mAh              Absolute LAM (Anode) in mAh
        LAM_Anode_pct              Relative LAM (Anode) in %
        LAM_Cathode_mAh            Absolute LAM (Cathode) in mAh
        LAM_Cathode_pct            Relative LAM (Cathode) in %
        C_Anode_ref_mAh            Reference anode capacity
        C_Anode_mAh                Current anode capacity
        C_Cathode_ref_mAh          Reference cathode capacity
        C_Cathode_mAh              Current cathode capacity
        Li_Inventory_ref_mAh       Reference lithium inventory
        Li_Inventory_mAh           Current lithium inventory
        ========================== ==============================================

    Notes
    -----
    Positive LLI / LAM values indicate degradation (loss).
    A negative value indicates a measurement artefact or fitting uncertainty.
    """
    if ref_quantities.empty:
        raise ValueError("ref_quantities is empty.")
    if aged_quantities.empty:
        raise ValueError("aged_quantities is empty.")
    if len(ref_quantities) != 1:
        raise ValueError(
            "ref_quantities must contain exactly one row (the reference state). "
            f"Got {len(ref_quantities)} rows."
        )

    ref = ref_quantities.iloc[0]

    # Reference values
    Li_ref  = float(ref["Li_Inventory_mAh"])
    Can_ref = float(ref["C_Anode_mAh"])
    Cca_ref = float(ref["C_Cathode_mAh"])
    cap_ref = float(ref["Full_Cell_Cap_mAh"])

    rows = []
    for _, aged in aged_quantities.iterrows():
        Li_now  = float(aged["Li_Inventory_mAh"])
        Can_now = float(aged["C_Anode_mAh"])
        Cca_now = float(aged["C_Cathode_mAh"])
        cap_now = float(aged["Full_Cell_Cap_mAh"])

        LLI_mAh        = Li_ref  - Li_now
        LAM_anode_mAh  = Can_ref - Can_now
        LAM_cathode_mAh = Cca_ref - Cca_now

        rows.append({
            # ── capacity / SOH ───────────────────────────────────────────
            "Full_Cell_Cap_ref_mAh":  cap_ref,
            "Full_Cell_Cap_mAh":      cap_now,
            "SOH_pct":                cap_now / cap_ref * 100,
            # ── LLI ─────────────────────────────────────────────────────
            "LLI_mAh":                LLI_mAh,
            "LLI_pct":                LLI_mAh / Li_ref * 100,
            # ── LAM Anode ────────────────────────────────────────────────
            "LAM_Anode_mAh":          LAM_anode_mAh,
            "LAM_Anode_pct":          LAM_anode_mAh / Can_ref * 100,
            # ── LAM Cathode ──────────────────────────────────────────────
            "LAM_Cathode_mAh":        LAM_cathode_mAh,
            "LAM_Cathode_pct":        LAM_cathode_mAh / Cca_ref * 100,
            # ── absolute values for reference ────────────────────────────
            "C_Anode_ref_mAh":        Can_ref,
            "C_Anode_mAh":            Can_now,
            "C_Cathode_ref_mAh":      Cca_ref,
            "C_Cathode_mAh":          Cca_now,
            "Li_Inventory_ref_mAh":   Li_ref,
            "Li_Inventory_mAh":       Li_now,
        })

    return pd.DataFrame(rows)

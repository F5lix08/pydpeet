
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline as CSP
from typing import List
from pathlib import Path


_FITTING_DATA_DIR = Path(__file__).resolve().parents[3] / "res" / "fitting_data"
dir_anode = str(_FITTING_DATA_DIR / "halfCellAnodes")
dir_cathode = str(_FITTING_DATA_DIR / "halfCellCathodes")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_y_col(df: pd.DataFrame, x_col: str) -> str:
    """Return the voltage/OCV column name in an electrode DataFrame."""
    for candidate in ['OCV', 'Voltage', 'Voltage[V]', 'V', 'U']:
        if candidate in df.columns:
            return candidate
    cols = [c for c in df.columns if c != x_col]
    return cols[0] if cols else df.columns[-1]


def _build_electrode_spline(df: pd.DataFrame, x_col: str, y_col: str) -> CSP:
    """Build a CubicSpline on the electrode's original lithiation coordinates."""
    df_c = df[[x_col, y_col]].copy().sort_values(by=x_col).drop_duplicates(subset=[x_col])
    return CSP(df_c[x_col].values.astype(float), df_c[y_col].values.astype(float))


def _objective_fn(params: np.ndarray,
                  soc_grid: np.ndarray, V_full: np.ndarray,
                  an_spline: CSP, an_domain: tuple,
                  cat_spline: CSP, cat_domain: tuple) -> float:
    """
    RMSE between measured and simulated full-cell voltage.

    params = [a_min, a_max, c_min, c_max] in original lithiation coordinates.
    Charge direction: anode lithiation increases, cathode lithiation decreases.
    """
    a_min, a_max, c_min, c_max = params
    lith_a = np.clip(a_min + soc_grid * (a_max - a_min), *an_domain)
    lith_c = np.clip(c_max - soc_grid * (c_max - c_min), *cat_domain)
    V_sim = cat_spline(lith_c) - an_spline(lith_a)
    return float(np.sqrt(np.mean((V_full - V_sim) ** 2)))

# ---------------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------------

def fit_half_cells(full_df: pd.DataFrame, anode_df: pd.DataFrame, cathode_df: pd.DataFrame,
                   methode: str = "multi-start",
                   full_cell_name: str = "FullCell",
                   anode_name: str = "Anode", cathode_name: str = "Cathode",
                   full_x_col: str = 'SOC', full_y_col: str = 'Voltage[V]') -> pd.DataFrame:

    # Detect electrode column names
    anode_x = 'Lithiation' if 'Lithiation' in anode_df.columns else anode_df.columns[0]
    anode_y = _detect_y_col(anode_df, anode_x)
    cathode_x = 'Lithiation' if 'Lithiation' in cathode_df.columns else cathode_df.columns[0]
    cathode_y = _detect_y_col(cathode_df, cathode_x)

    # Build electrode splines once on original lithiation coordinates
    an_spline = _build_electrode_spline(anode_df, anode_x, anode_y)
    cat_spline = _build_electrode_spline(cathode_df, cathode_x, cathode_y)
    an_domain = (float(anode_df[anode_x].min()), float(anode_df[anode_x].max()))
    cat_domain = (float(cathode_df[cathode_x].min()), float(cathode_df[cathode_x].max()))

    # Build full-cell reference on the measured SOC window
    fc = (full_df[[full_x_col, full_y_col]].dropna()
          .sort_values(by=full_x_col).drop_duplicates(subset=[full_x_col]))
    soc_raw = fc[full_x_col].values.astype(float)
    fc_spline = CSP(soc_raw, fc[full_y_col].values.astype(float))
    soc_grid = np.linspace(soc_raw.min(), soc_raw.max(), 300)
    V_full = fc_spline(soc_grid)

    # Optimizer bounds: [a_min, a_max, c_min, c_max] clamped to the actual data range
    # so the solution never requires extrapolation beyond the reference data.
    an_lo, an_hi = an_domain
    cat_lo, cat_hi = cat_domain
    bounds = [
        (max(0.0,  an_lo),  min(0.3,  an_hi)),   # a_min
        (max(0.6,  an_lo),  min(1.0,  an_hi)),   # a_max
        (max(0.0, cat_lo),  min(0.5, cat_hi)),   # c_min
        (max(0.6, cat_lo),  min(1.0, cat_hi)),   # c_max
    ]
    # Skip this pair if any bound is infeasible (data range too narrow)
    if any(lo >= hi for lo, hi in bounds):
        return pd.DataFrame()
    args = (soc_grid, V_full, an_spline, an_domain, cat_spline, cat_domain)

    # Diverse starting points for multi-start optimization
    _starts = [
        [0.02, 0.85, 0.05, 0.90],
        [0.05, 0.90, 0.10, 0.85],
        [0.10, 0.85, 0.05, 0.95],
        [0.10, 0.90, 0.20, 0.90],
        [0.05, 0.95, 0.05, 0.85],
        [0.15, 0.90, 0.05, 0.90],
        [0.05, 0.80, 0.10, 0.80],
        [0.20, 0.90, 0.30, 0.95],
        [0.05, 0.75, 0.05, 0.75],
        [0.25, 0.95, 0.15, 0.90],
        [0.00, 0.80, 0.05, 0.80],
        [0.10, 0.80, 0.20, 0.85],
        [0.15, 0.85, 0.10, 0.85],
        [0.05, 0.85, 0.30, 0.90],
        [0.20, 0.95, 0.05, 0.95],
        [0.02, 0.70, 0.05, 0.70],
    ]

    best_fun = np.inf
    best_x = None

    for x0 in _starts:
        x0c = [np.clip(x0[i], bounds[i][0], bounds[i][1]) for i in range(4)]
        try:
            res = minimize(
                _objective_fn, x0=x0c, args=args,
                method='L-BFGS-B', bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 500}
            )
            if res.fun < best_fun:
                best_fun = res.fun
                best_x = res.x.copy()
        except Exception:
            continue

    if best_x is not None:
        return pd.DataFrame({
            "RMSE[mV]": [best_fun * 1000],
            "FullCell": [full_cell_name],
            "Anode": [anode_name],
            "Cathode": [cathode_name],
            "Sol_Anode_Min": [best_x[0]],
            "Sol_Anode_Max": [best_x[1]],
            "Sol_Cathode_Min": [best_x[2]],
            "Sol_Cathode_Max": [best_x[3]],
            "Solution_Array": [list(best_x)]
        })
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Grid search over all electrode pairs
# ---------------------------------------------------------------------------

def find_best_half_cell_match(full_df: pd.DataFrame,
                               anodes_dir: str = dir_anode,
                               cathodes_dir: str = dir_cathode,
                               methode: str = "multi-start",
                               full_cell_name: str = "FullCell",
                               full_x_col: str = 'SOC',
                               full_y_col: str = 'Voltage[V]') -> pd.DataFrame:

    anode_files = list(Path(anodes_dir).glob("*.csv")) + list(Path(anodes_dir).glob("*.txt"))
    cathode_files = list(Path(cathodes_dir).glob("*.csv")) + list(Path(cathodes_dir).glob("*.txt"))

    if not anode_files or not cathode_files:
        print("Warnung: Es konnten keine Anoden- oder Kathoden-Dateien gefunden werden.")
        return pd.DataFrame()

    def read_data(filepath):
        try:
            df = pd.read_csv(filepath)
            if df.shape[1] < 2:
                df = pd.read_csv(filepath, sep=None, engine='python')
            return df
        except Exception as e:
            print(f"Fehler beim Einlesen von {filepath}: {e}")
            return None

    anodes_data = {f.name: read_data(f) for f in anode_files}
    anodes_data = {k: v for k, v in anodes_data.items() if v is not None and not v.empty}

    cathodes_data = {f.name: read_data(f) for f in cathode_files}
    cathodes_data = {k: v for k, v in cathodes_data.items() if v is not None and not v.empty}

    all_results = []
    for anode_name, anode_df in anodes_data.items():
        for cathode_name, cathode_df in cathodes_data.items():
            res_df = fit_half_cells(
                full_df=full_df,
                anode_df=anode_df,
                cathode_df=cathode_df,
                methode=methode,
                full_cell_name=full_cell_name,
                anode_name=anode_name,
                cathode_name=cathode_name,
                full_x_col=full_x_col,
                full_y_col=full_y_col,
            )
            if not res_df.empty:
                all_results.append(res_df)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True).sort_values(by="RMSE[mV]").reset_index(drop=True)


def get_best_half_cell_fit(full_df: pd.DataFrame,
                            anodes_dir: str = dir_anode,
                            cathodes_dir: str = dir_cathode,
                            methode: str = "multi-start",
                            full_cell_name: str = "FullCell",
                            full_x_col: str = 'SOC',
                            full_y_col: str = 'Voltage[V]') -> pd.DataFrame:

    all_results = find_best_half_cell_match(
        full_df=full_df,
        anodes_dir=anodes_dir,
        cathodes_dir=cathodes_dir,
        methode=methode,
        full_cell_name=full_cell_name,
        full_x_col=full_x_col,
        full_y_col=full_y_col,
    )

    if all_results.empty:
        return pd.DataFrame()
    return all_results.head(1)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt


def plot_half_cell_match(full_df: pd.DataFrame,
                          anode_df: pd.DataFrame,
                          cathode_df: pd.DataFrame,
                          solution_array: list,
                          full_cap_mAh: float = 2000,
                          filename: str = "Match_Plot.png"):

    anode_x = 'Lithiation' if 'Lithiation' in anode_df.columns else anode_df.columns[0]
    anode_y = _detect_y_col(anode_df, anode_x)
    cathode_x = 'Lithiation' if 'Lithiation' in cathode_df.columns else cathode_df.columns[0]
    cathode_y = _detect_y_col(cathode_df, cathode_x)

    full_x = 'SOC' if 'SOC' in full_df.columns else full_df.columns[0]
    full_y = _detect_y_col(full_df, full_x)

    a_min, a_max = solution_array[0], solution_array[1]
    c_min, c_max = solution_array[2], solution_array[3]

    an_cap = full_cap_mAh / (a_max - a_min) if (a_max - a_min) != 0 else 0
    cat_cap = full_cap_mAh / (c_max - c_min) if (c_max - c_min) != 0 else 0
    #print(f"Berechnete Skalierung -> Anode nominal: {an_cap:.2f} mAh | Kathode nominal: {cat_cap:.2f} mAh")

    # Build splines on ORIGINAL lithiation coordinates (no normalization)
    an_clean = anode_df[[anode_x, anode_y]].copy().sort_values(by=anode_x).drop_duplicates(subset=[anode_x])
    cat_clean = cathode_df[[cathode_x, cathode_y]].copy().sort_values(by=cathode_x).drop_duplicates(subset=[cathode_x])

    an_spline = CSP(an_clean[anode_x].values.astype(float), an_clean[anode_y].values.astype(float))
    cat_spline = CSP(cat_clean[cathode_x].values.astype(float), cat_clean[cathode_y].values.astype(float))

    soc_grid = np.linspace(full_df[full_x].min(), full_df[full_x].max(), 300)

    # Simulate: anode lithiation increases, cathode decreases during charge
    lith_a = a_min + soc_grid * (a_max - a_min)
    lith_c = c_max - soc_grid * (c_max - c_min)
    lith_a = np.clip(lith_a, an_clean[anode_x].min(), an_clean[anode_x].max())
    lith_c = np.clip(lith_c, cat_clean[cathode_x].min(), cat_clean[cathode_x].max())
    simulated_v = cat_spline(lith_c) - an_spline(lith_a)

    # Dense SOC grid for smooth electrode curves via spline
    an_lith_grid = np.linspace(an_clean[anode_x].min(), an_clean[anode_x].max(), 500)
    an_soc_grid = (an_lith_grid - a_min) / (a_max - a_min)

    cat_lith_grid = np.linspace(cat_clean[cathode_x].min(), cat_clean[cathode_x].max(), 500)
    cat_soc_grid = (c_max - cat_lith_grid) / (c_max - c_min)

    plt.figure(figsize=(10, 7))
    plt.plot(full_df[full_x], full_df[full_y], label='Measured iOCV', linewidth=2.5, color='blue')
    plt.plot(soc_grid, simulated_v, label='Simulated iOCV', linestyle='dashed', color='red', linewidth=2.5)
    plt.plot(an_soc_grid, an_spline(an_lith_grid), label='Fitted Anode', color='orange')
    plt.plot(cat_soc_grid, cat_spline(cat_lith_grid), label='Fitted Cathode', color='green')

    plt.xlabel("SOC")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit("Aufruf: python half_cell_fitting.py <Parquet-Datei mit iOCV-Daten>")
    df_full = pd.read_parquet(sys.argv[1])
    df_full = df_full[df_full["iOCV_type"] == "Charge"]

    best_fit_df = get_best_half_cell_fit(df_full)
    print(best_fit_df)

    best_anode_file = best_fit_df["Anode"].iloc[0]
    best_cathode_file = best_fit_df["Cathode"].iloc[0]
    solution_array = best_fit_df["Solution_Array"].iloc[0]

    anode_raw = pd.read_csv(f"{dir_anode}/{best_anode_file}")
    cathode_raw = pd.read_csv(f"{dir_cathode}/{best_cathode_file}")

    plot_half_cell_match(
        full_df=df_full,
        anode_df=anode_raw,
        cathode_df=cathode_raw,
        solution_array=solution_array,
        filename="BestMatch_Plot.png"
    )

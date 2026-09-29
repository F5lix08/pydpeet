import shutil
import unittest
from io import StringIO
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas

from pydpeet.process.analyze.extract.half_cell_fitting import (
    _build_electrode_spline,
    _detect_y_col,
    _objective_fn,
    find_best_half_cell_match,
    fit_half_cells,
    get_best_half_cell_fit,
    plot_half_cell_match,
)
from test.utils import TEMP_PATH

matplotlib.use("Agg", force=True)  # headless backend so plot_half_cell_match runs without a display

# Ground truth stoichiometric window used to generate the synthetic full cell.
# Chosen inside the optimizer bounds (a_min [0, 0.3], a_max [0.6, 1], c_min [0, 0.5], c_max [0.6, 1]).
TRUE_PARAMS = {"a_min": 0.05, "a_max": 0.85, "c_min": 0.30, "c_max": 0.90}

_LITHIATION = np.linspace(0.0, 1.0, 41)

# Graphite-like anode: potential drops steeply, then flattens
ANODE_DF = pandas.DataFrame({"Lithiation": _LITHIATION, "OCV": 0.08 + 0.60 * np.exp(-6.0 * _LITHIATION)})

# NMC-like cathode: potential falls monotonically with lithiation
CATHODE_DF = pandas.DataFrame({"Lithiation": _LITHIATION, "OCV": 4.25 - 0.90 * _LITHIATION**1.5})


def _make_full_cell_df(params=None, n=201, noise_mv=0.0, seed=42):
    # Forward model of the fit: during charge the anode lithiates (a_min -> a_max)
    # while the cathode delithiates (c_max -> c_min).
    params = TRUE_PARAMS if params is None else params
    anode_spline = _build_electrode_spline(ANODE_DF, "Lithiation", "OCV")
    cathode_spline = _build_electrode_spline(CATHODE_DF, "Lithiation", "OCV")
    soc = np.linspace(0.0, 1.0, n)
    lith_a = params["a_min"] + soc * (params["a_max"] - params["a_min"])
    lith_c = params["c_max"] - soc * (params["c_max"] - params["c_min"])
    voltage = cathode_spline(lith_c) - anode_spline(lith_a)
    if noise_mv > 0:
        voltage = voltage + np.random.default_rng(seed).normal(0.0, noise_mv / 1000.0, n)
    return pandas.DataFrame({"SOC": soc, "Voltage[V]": voltage})


FULL_CELL_DF = _make_full_cell_df()


class HalfCellFitTestCase(unittest.TestCase):
    def _assert_params_close(self, fit_row, params, tol):
        self.assertAlmostEqual(fit_row["Sol_Anode_Min"], params["a_min"], delta=tol)
        self.assertAlmostEqual(fit_row["Sol_Anode_Max"], params["a_max"], delta=tol)
        self.assertAlmostEqual(fit_row["Sol_Cathode_Min"], params["c_min"], delta=tol)
        self.assertAlmostEqual(fit_row["Sol_Cathode_Max"], params["c_max"], delta=tol)


class TestDetectYCol(unittest.TestCase):
    def test_finds_each_known_candidate(self):
        for candidate in ["OCV", "Voltage", "Voltage[V]", "V", "U"]:
            df = pandas.DataFrame({"Lithiation": [0.0, 1.0], candidate: [0.1, 0.2]})
            self.assertEqual(candidate, _detect_y_col(df, "Lithiation"))

    def test_priority_order(self):
        # "OCV" is listed before "Voltage" in the candidate list
        df = pandas.DataFrame({"Lithiation": [0.0, 1.0], "Voltage": [1, 2], "OCV": [3, 4]})
        self.assertEqual("OCV", _detect_y_col(df, "Lithiation"))

    def test_fallback_first_non_x_column(self):
        df = pandas.DataFrame({"Lithiation": [0.0, 1.0], "Spannung": [0.1, 0.2]})
        self.assertEqual("Spannung", _detect_y_col(df, "Lithiation"))


class TestBuildElectrodeSpline(unittest.TestCase):
    def test_spline_hits_knots(self):
        spline = _build_electrode_spline(ANODE_DF, "Lithiation", "OCV")
        np.testing.assert_allclose(spline(ANODE_DF["Lithiation"]), ANODE_DF["OCV"], atol=1e-12)

    def test_unsorted_and_duplicate_x_handled(self):
        df = pandas.DataFrame({"Lithiation": [0.5, 0.0, 1.0, 0.5], "OCV": [0.2, 0.6, 0.1, 0.2]})
        spline = _build_electrode_spline(df, "Lithiation", "OCV")
        self.assertAlmostEqual(0.6, float(spline(0.0)), places=12)
        self.assertAlmostEqual(0.2, float(spline(0.5)), places=12)
        self.assertAlmostEqual(0.1, float(spline(1.0)), places=12)


class TestObjectiveFn(unittest.TestCase):
    def setUp(self):
        self.anode_spline = _build_electrode_spline(ANODE_DF, "Lithiation", "OCV")
        self.cathode_spline = _build_electrode_spline(CATHODE_DF, "Lithiation", "OCV")
        self.soc_grid = np.linspace(0.0, 1.0, 300)
        lith_a = TRUE_PARAMS["a_min"] + self.soc_grid * (TRUE_PARAMS["a_max"] - TRUE_PARAMS["a_min"])
        lith_c = TRUE_PARAMS["c_max"] - self.soc_grid * (TRUE_PARAMS["c_max"] - TRUE_PARAMS["c_min"])
        self.v_full = self.cathode_spline(lith_c) - self.anode_spline(lith_a)
        self.args = (
            self.soc_grid,
            self.v_full,
            self.anode_spline,
            (0.0, 1.0),
            self.cathode_spline,
            (0.0, 1.0),
        )

    def test_zero_at_truth(self):
        params = np.array([TRUE_PARAMS["a_min"], TRUE_PARAMS["a_max"], TRUE_PARAMS["c_min"], TRUE_PARAMS["c_max"]])
        rmse = _objective_fn(params, *self.args)
        self.assertIsInstance(rmse, float)
        self.assertLess(rmse, 1e-10)

    def test_positive_for_wrong_params(self):
        rmse = _objective_fn(np.array([0.20, 0.70, 0.10, 0.75]), *self.args)
        self.assertGreater(rmse, 0.01)  # clearly above 10 mV

    def test_matches_manual_rmse(self):
        params = np.array([0.10, 0.80, 0.35, 0.85])
        lith_a = np.clip(params[0] + self.soc_grid * (params[1] - params[0]), 0.0, 1.0)
        lith_c = np.clip(params[3] - self.soc_grid * (params[3] - params[2]), 0.0, 1.0)
        v_sim = self.cathode_spline(lith_c) - self.anode_spline(lith_a)
        expected = float(np.sqrt(np.mean((self.v_full - v_sim) ** 2)))
        self.assertAlmostEqual(expected, _objective_fn(params, *self.args), places=12)


class TestFitHalfCells(HalfCellFitTestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit = fit_half_cells(
            FULL_CELL_DF,
            ANODE_DF,
            CATHODE_DF,
            full_cell_name="TestZelle",
            anode_name="TestAnode",
            cathode_name="TestKathode",
        )

    def test_recovers_ground_truth(self):
        self.assertEqual(1, len(self.fit))
        self.assertLess(self.fit["RMSE[mV]"].iloc[0], 0.5)  # noise-free -> close to zero
        self._assert_params_close(self.fit.iloc[0], TRUE_PARAMS, tol=0.01)

    def test_result_schema_and_names(self):
        expected_columns = {
            "RMSE[mV]",
            "FullCell",
            "Anode",
            "Cathode",
            "Sol_Anode_Min",
            "Sol_Anode_Max",
            "Sol_Cathode_Min",
            "Sol_Cathode_Max",
            "Solution_Array",
        }
        self.assertEqual(expected_columns, set(self.fit.columns))
        self.assertEqual("TestZelle", self.fit["FullCell"].iloc[0])
        self.assertEqual("TestAnode", self.fit["Anode"].iloc[0])
        self.assertEqual("TestKathode", self.fit["Cathode"].iloc[0])

    def test_solution_array_matches_columns(self):
        row = self.fit.iloc[0]
        expected = [row["Sol_Anode_Min"], row["Sol_Anode_Max"], row["Sol_Cathode_Min"], row["Sol_Cathode_Max"]]
        self.assertEqual(expected, row["Solution_Array"])

    def test_solution_within_bounds(self):
        row = self.fit.iloc[0]
        self.assertTrue(0.0 <= row["Sol_Anode_Min"] <= 0.3)
        self.assertTrue(0.6 <= row["Sol_Anode_Max"] <= 1.0)
        self.assertTrue(0.0 <= row["Sol_Cathode_Min"] <= 0.5)
        self.assertTrue(0.6 <= row["Sol_Cathode_Max"] <= 1.0)

    def test_recovers_truth_with_noise(self):
        full_cell_noisy = _make_full_cell_df(noise_mv=2.0)
        fit = fit_half_cells(full_cell_noisy, ANODE_DF, CATHODE_DF)
        self.assertEqual(1, len(fit))
        self._assert_params_close(fit.iloc[0], TRUE_PARAMS, tol=0.05)

    def test_custom_column_names(self):
        renamed = FULL_CELL_DF.rename(columns={"SOC": "soc_norm", "Voltage[V]": "U_cell"})
        fit = fit_half_cells(renamed, ANODE_DF, CATHODE_DF, full_x_col="soc_norm", full_y_col="U_cell")
        self.assertEqual(1, len(fit))
        self._assert_params_close(fit.iloc[0], TRUE_PARAMS, tol=0.01)

    def test_extra_columns_nans_and_duplicates_ignored(self):
        # Realistic input: extra columns, NaN rows and duplicated SOC values
        messy = FULL_CELL_DF.copy()
        messy["Current[A]"] = 0.0
        nan_row = pandas.DataFrame({"SOC": [np.nan], "Voltage[V]": [np.nan]})
        messy = pandas.concat([messy, messy.head(5), nan_row], ignore_index=True)
        fit = fit_half_cells(messy, ANODE_DF, CATHODE_DF)
        self.assertEqual(1, len(fit))
        self._assert_params_close(fit.iloc[0], TRUE_PARAMS, tol=0.01)

    def test_infeasible_bounds_return_empty(self):
        # Cathode domain [0.55, 0.58] makes the c_min bound (0.55, 0.5) empty
        lithiation = np.linspace(0.55, 0.58, 10)
        narrow_cathode = pandas.DataFrame({"Lithiation": lithiation, "OCV": 4.0 - lithiation})
        fit = fit_half_cells(FULL_CELL_DF, ANODE_DF, narrow_cathode)
        self.assertTrue(fit.empty)


class TestLibrarySearch(HalfCellFitTestCase):
    LIBRARY_PATH = TEMP_PATH / "half_cell_fit_library"
    ANODES_PATH = LIBRARY_PATH / "anodes"
    CATHODES_PATH = LIBRARY_PATH / "cathodes"
    EMPTY_PATH = LIBRARY_PATH / "empty"
    NO_FILES_WARNING = "Warnung: Es konnten keine Anoden- oder Kathoden-Dateien gefunden werden.\n"

    @classmethod
    def setUpClass(cls):
        # Temporary mini library: the true pair plus one clearly different distractor each
        cls.ANODES_PATH.mkdir(parents=True, exist_ok=True)
        cls.CATHODES_PATH.mkdir(parents=True, exist_ok=True)
        cls.EMPTY_PATH.mkdir(parents=True, exist_ok=True)
        ANODE_DF.to_csv(cls.ANODES_PATH / "Anode_wahr.csv", index=False)
        CATHODE_DF.to_csv(cls.CATHODES_PATH / "Kathode_wahr.csv", index=False)
        distractor_anode = pandas.DataFrame({"Lithiation": _LITHIATION, "OCV": 0.50 - 0.35 * _LITHIATION})
        distractor_anode.to_csv(cls.ANODES_PATH / "Anode_ablenker.csv", index=False)
        distractor_cathode = pandas.DataFrame({"Lithiation": _LITHIATION, "OCV": 4.40 - 1.40 * _LITHIATION**2})
        distractor_cathode.to_csv(cls.CATHODES_PATH / "Kathode_ablenker.csv", index=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.LIBRARY_PATH, ignore_errors=True)

    def test_ranking_sorted_and_true_pair_wins(self):
        ranking = find_best_half_cell_match(
            FULL_CELL_DF, anodes_dir=str(self.ANODES_PATH), cathodes_dir=str(self.CATHODES_PATH)
        )
        self.assertEqual(4, len(ranking))  # 2 anodes x 2 cathodes
        rmse = ranking["RMSE[mV]"].to_numpy()
        self.assertTrue(np.all(np.diff(rmse) >= 0))  # sorted ascending
        self.assertEqual("Anode_wahr.csv", ranking["Anode"].iloc[0])
        self.assertEqual("Kathode_wahr.csv", ranking["Cathode"].iloc[0])
        self.assertLess(rmse[0], 0.5)
        self.assertGreater(rmse[1], rmse[0])

    @patch("sys.stdout", new_callable=StringIO)
    def test_empty_library_returns_empty(self, mock_stdout):
        result = find_best_half_cell_match(
            FULL_CELL_DF, anodes_dir=str(self.EMPTY_PATH), cathodes_dir=str(self.EMPTY_PATH)
        )
        self.assertTrue(result.empty)
        self.assertEqual(self.NO_FILES_WARNING, mock_stdout.getvalue())

    def test_header_only_file_is_skipped(self):
        header_only_file = self.ANODES_PATH / "Anode_leer.csv"
        header_only_file.write_text("Lithiation,OCV\n")
        try:
            ranking = find_best_half_cell_match(
                FULL_CELL_DF, anodes_dir=str(self.ANODES_PATH), cathodes_dir=str(self.CATHODES_PATH)
            )
            # the empty file is dropped, the 4 valid pairs remain
            self.assertEqual(4, len(ranking))
            self.assertNotIn("Anode_leer.csv", set(ranking["Anode"]))
        finally:
            header_only_file.unlink()

    def test_get_best_returns_single_top_row(self):
        best = get_best_half_cell_fit(
            FULL_CELL_DF,
            anodes_dir=str(self.ANODES_PATH),
            cathodes_dir=str(self.CATHODES_PATH),
            full_cell_name="TestZelle",
        )
        self.assertEqual(1, len(best))
        self.assertEqual("Anode_wahr.csv", best["Anode"].iloc[0])
        self.assertEqual("Kathode_wahr.csv", best["Cathode"].iloc[0])
        self.assertEqual("TestZelle", best["FullCell"].iloc[0])
        self._assert_params_close(best.iloc[0], TRUE_PARAMS, tol=0.01)

    @patch("sys.stdout", new_callable=StringIO)
    def test_get_best_empty_library_returns_empty(self, mock_stdout):
        best = get_best_half_cell_fit(FULL_CELL_DF, anodes_dir=str(self.EMPTY_PATH), cathodes_dir=str(self.EMPTY_PATH))
        self.assertTrue(best.empty)
        self.assertEqual(self.NO_FILES_WARNING, mock_stdout.getvalue())


class TestPlotHalfCellMatch(unittest.TestCase):
    OUTPUT_PATH = TEMP_PATH / "half_cell_fit_plot"

    def test_creates_plot_file(self):
        fit = fit_half_cells(FULL_CELL_DF, ANODE_DF, CATHODE_DF)
        self.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        output_file = self.OUTPUT_PATH / "match.png"
        plot_half_cell_match(
            full_df=FULL_CELL_DF,
            anode_df=ANODE_DF,
            cathode_df=CATHODE_DF,
            solution_array=fit["Solution_Array"].iloc[0],
            filename=str(output_file),
        )
        self.assertTrue(output_file.exists())
        self.assertGreater(output_file.stat().st_size, 0)

    def tearDown(self):
        shutil.rmtree(self.OUTPUT_PATH, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

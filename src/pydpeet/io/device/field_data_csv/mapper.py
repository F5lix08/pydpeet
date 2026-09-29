# Map raw-data column names (left) to standardized column names (right)
# NOTE: "SOC" is not part of STANDARD_COLUMNS (pydpeet.io.configs.config) and is
# deliberately left unmapped here — the reader already names it "SOC", so it
# passes through mapping() unchanged. Callers must pass keep_all_additional_data=True
# to eet.read(), otherwise convert_file() drops it before the formatter runs.
COLUMN_MAP = {
    "terminaltime":         "Test_Time[s]",
    "current":              "Current[A]",
    "voltage_cell_avg":     "Voltage[V]",
    "temperature_cell_avg": "Temperature[°C]",
    "Date_Time":            "Date_Time",
    "Step_Count":           "Step_Count",
}

# Default columns of the standardized format
# which are not present in the raw data files.
MISSING_REQUIRED_COLUMNS = [
    "EIS_f[Hz]",
    "EIS_Z_Real[Ohm]",
    "EIS_Z_Imag[Ohm]",
    "EIS_DC[A]",
]

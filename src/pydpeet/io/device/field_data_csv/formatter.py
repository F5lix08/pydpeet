import pandas as pd

from pydpeet.io.utils.formatter_utils import typecast


def get_data_into_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format a DataFrame from a field-data CSV (vin schema) into standard format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be formatted.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame.
    """
    typecast(df, "Test_Time[s]", float)
    df["Test_Time[s]"] = df["Test_Time[s]"] - df["Test_Time[s]"].iloc[0]

    typecast(df, "SOC", float)
    df["SOC"] = df["SOC"].ffill() * 0.01

    typecast(df, "Step_Count", int)
    typecast(df, "Temperature[°C]", float)
    typecast(df, "Voltage[V]", float)
    typecast(df, "Current[A]", float)

    df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")

    return df

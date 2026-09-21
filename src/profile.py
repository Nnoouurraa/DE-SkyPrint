from __future__ import annotations
import pandas as pd

def dataframe_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Return the same compact profiling summary used in the cleaning notebook."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "null_count": df.isna().sum(),
        "null_pct": (df.isna().mean() * 100).round(2),
        "nunique": df.nunique(dropna=True),
    }).sort_values(["null_pct", "nunique"], ascending=[False, True])

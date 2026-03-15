import numpy as np
import pandas as pd

_SOURCE_ORDER = ["cleveland", "hungarian", "long_beach_va", "switzerland"]

_CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
_CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


def encode_source(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the 'source' string column with a numeric 'source_code' (1-based)."""
    df = df.copy()
    df["source"] = pd.Categorical(df["source"], categories=_SOURCE_ORDER, ordered=False)
    df["source_code"] = df["source"].cat.codes + 1
    df = df.drop(columns=["source"])
    return df


def drop_ca(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the 'ca' column (too many missing values across sources)."""
    return df.drop(columns=["ca"])


def binarize_num(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'num' to binary (0 = no disease, 1 = disease) and move it to the last column."""
    df = df.copy()
    df["num"] = (df["num"] > 0).astype(int)
    # Move num to last position
    cols = [c for c in df.columns if c != "num"] + ["num"]
    return df[cols]


def fix_zero_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physiologically impossible 0 values with NaN in chol and trestbps."""
    df = df.copy()
    df["chol"] = df["chol"].replace(0, np.nan)
    df["trestbps"] = df["trestbps"].replace(0, np.nan)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for continuous, mode for categorical."""
    df = df.copy()
    for col in _CONTINUOUS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in _CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline: encode source, drop ca, binarize target,
    fix zero-as-missing, impute remaining NaN values."""
    df = encode_source(df)
    df = drop_ca(df)
    df = binarize_num(df)
    df = fix_zero_missing(df)
    df = impute_missing(df)
    return df

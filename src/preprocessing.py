import pandas as pd

_SOURCE_ORDER = ["cleveland", "hungarian", "long_beach_va", "switzerland"]


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


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline: encode source, drop ca, binarize target."""
    df = encode_source(df)
    df = drop_ca(df)
    df = binarize_num(df)
    return df

import pandas as pd
from pathlib import Path

# Column names (standard 14 UCI features) 
_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Processed-file URLs from the UCI ML Repository 
_SOURCES = {
    "cleveland":    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "hungarian":    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "long_beach_va":"https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
    "switzerland":  "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
}

_CSV = Path(__file__).parent.parent / "data" / "raw" / "heart_disease.csv"


def load_heart_disease(force_download: bool = False) -> pd.DataFrame:
    """
    Load the UCI Heart Disease dataset (all 4 sources, 14 standard features + source).

    Downloads the four pre-processed files from the UCI ML Repository on the
    first call and caches the result to data/raw/heart_disease.csv.
    Subsequent calls load from the local CSV unless force_download=True.

    Returns
    -------
    DataFrame with 920 rows and 15 columns:
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal, num (0-4 disease severity), source.
    """
    if not force_download and _CSV.exists():
        return pd.read_csv(_CSV)

    frames = []
    for source, url in _SOURCES.items():
        df = pd.read_csv(url, header=None, names=_COLS, na_values="?")
        df["source"] = source
        frames.append(df)
        print(f"  {source:<15}: {len(df)} rows")

    combined = pd.concat(frames, ignore_index=True)

    _CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(_CSV, index=False)
    print(f"\nSaved to {_CSV}  ({len(combined)} rows total)")

    return combined

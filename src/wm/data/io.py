"""DataFrame storage with parquet→pickle fallback.

Parquet (pyarrow) is preferred, but pyarrow has no prebuilt wheels on some
platforms (notably Termux/Android). When parquet support is unavailable the
same data is stored as a .pkl file next to the requested path.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_df(df: pd.DataFrame, path: Path) -> Path:
    """Save df to path (parquet), falling back to pickle. Returns actual path used."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError, OSError):
        alt = path.with_suffix(".pkl")
        df.to_pickle(alt)
        return alt


def load_df(path: Path) -> pd.DataFrame:
    """Load df from path, trying parquet then the pickle fallback."""
    path = Path(path)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except (ImportError, ValueError, OSError):
            pass
    alt = path.with_suffix(".pkl")
    if alt.exists():
        return pd.read_pickle(alt)
    raise FileNotFoundError(f"Neither {path} nor {alt} exists (run the pipeline step that creates it)")

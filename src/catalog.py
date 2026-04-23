"""Seamount age catalogue loader.

The Hawaii–Emperor chain's bend story is told by about a hundred Ar-Ar dates
scattered along ~6000 km of seafloor. We store the compiled catalogue in a
single CSV (`data/seamount_ages.csv`) and expose a small loader that returns
a ``pandas.DataFrame`` with a predictable schema, sorted from youngest to
oldest.

Keeping the loader tiny lets the tests pin down exactly the invariants that
downstream geometry and plotting code relies on (longitudes in [0, 360),
age-ascending order, non-negative errors).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "name",
    "chain",
    "lon",
    "lat",
    "age_Ma",
    "age_err_Ma",
)


def load_seamount_catalog(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"catalog missing required columns: {missing}")

    df = df[list(REQUIRED_COLUMNS)].copy()
    df["lon"] = df["lon"].astype(float) % 360.0
    df["lat"] = df["lat"].astype(float)
    df["age_Ma"] = df["age_Ma"].astype(float)
    df["age_err_Ma"] = df["age_err_Ma"].astype(float).abs()

    df = df.sort_values("age_Ma", kind="mergesort").reset_index(drop=True)
    return df

"""Tests for src.catalog — Hawaii–Emperor seamount age CSV loader."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest

from src.catalog import REQUIRED_COLUMNS, load_seamount_catalog


@pytest.fixture()
def tiny_csv(tmp_path: Path) -> Path:
    """A CSV with the minimum schema we expect in the real data file."""
    csv = tmp_path / "sample.csv"
    csv.write_text(
        dedent(
            """\
            name,chain,lon,lat,age_Ma,age_err_Ma
            Kilauea,Hawaiian,-155.29,19.42,0.0,0.1
            # comment line that must be ignored
            Midway,Hawaiian,-177.37,28.20,27.7,0.3
            Daikakuji,Bend,-188.5,32.0,47.0,0.5
            Suiko,Emperor,-170.33,44.58,60.9,0.5
            """
        )
    )
    return csv


def test_required_columns_are_documented():
    # The loader advertises which columns downstream plotting depends on.
    assert {"name", "chain", "lon", "lat", "age_Ma", "age_err_Ma"} <= set(REQUIRED_COLUMNS)


def test_load_returns_dataframe_with_required_columns(tiny_csv: Path):
    df = load_seamount_catalog(tiny_csv)
    assert isinstance(df, pd.DataFrame)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"missing column: {col}"


def test_comments_and_blank_lines_are_skipped(tiny_csv: Path):
    df = load_seamount_catalog(tiny_csv)
    # 4 real rows, the '# comment' line must not appear.
    assert len(df) == 4
    assert "Kilauea" in set(df["name"])
    assert not df["name"].str.startswith("#").any()


def test_longitudes_are_normalised_to_0_360(tiny_csv: Path):
    """Plotting a chain that straddles the antimeridian is much easier when
    longitudes are expressed in [0, 360)."""
    df = load_seamount_catalog(tiny_csv)
    assert df["lon"].between(0.0, 360.0, inclusive="left").all()
    # Kilauea (-155.29) should become ~204.71.
    kilauea_lon = float(df.loc[df["name"] == "Kilauea", "lon"].iloc[0])
    assert kilauea_lon == pytest.approx(204.71, abs=0.01)


def test_sorted_by_age_ascending(tiny_csv: Path):
    df = load_seamount_catalog(tiny_csv)
    ages = df["age_Ma"].to_numpy()
    assert (ages[:-1] <= ages[1:]).all(), "catalog must be sorted by age"


def test_ages_non_negative_and_errors_non_negative(tiny_csv: Path):
    df = load_seamount_catalog(tiny_csv)
    assert (df["age_Ma"] >= 0).all()
    assert (df["age_err_Ma"] >= 0).all()


def test_chain_values_are_in_allowed_set(tiny_csv: Path):
    df = load_seamount_catalog(tiny_csv)
    allowed = {"Hawaiian", "Emperor", "Bend"}
    assert set(df["chain"]).issubset(allowed)


def test_raises_on_missing_column(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("name,lon,lat,age_Ma\nKilauea,-155.29,19.42,0.0\n")
    with pytest.raises(ValueError, match="missing"):
        load_seamount_catalog(bad)

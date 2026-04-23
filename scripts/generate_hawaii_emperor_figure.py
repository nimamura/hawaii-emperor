#!/usr/bin/env python
"""Generate outputs/hawaii_emperor_bend.png — the X-post figure.

Reads data/seamount_ages.csv, builds the four-panel matplotlib figure,
and writes the PNG. Also prints one-line segment stats so the caption
numbers can be sanity-checked before posting.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless write-only.

import matplotlib.pyplot as plt  # noqa: E402

from src.catalog import load_seamount_catalog  # noqa: E402
from src.geometry import (  # noqa: E402
    chain_azimuth_deg,
    chain_distance_via_bend_km,
    fit_broken_stick,
)
from src.plotting import plot_hawaii_emperor_figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "seamount_ages.csv"
OUT_PATH = REPO_ROOT / "outputs" / "hawaii_emperor_bend.png"
BEND_AGE_MA = 47.0


def _segment_stats(df) -> None:
    ages = df["age_Ma"].to_numpy()
    d = chain_distance_via_bend_km(df["lat"].to_numpy(), df["lon"].to_numpy(), chain=df["chain"].to_numpy())

    res = fit_broken_stick(ages, d, break_bounds=(35.0, 60.0))
    print(f"broken-stick fit: bend at {res.break_age_Ma:.1f} Ma, "
          f"Hawaiian {res.slope_young*0.1:.2f} cm/yr, Emperor {res.slope_old*0.1:.2f} cm/yr")

    # Mean chain azimuth on each side of the bend (rough).
    is_hawaiian = df["chain"].isin(["Hawaiian", "Bend"])
    is_emperor = df["chain"] == "Emperor"
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()

    def mean_az(mask):
        idx = np.where(mask)[0]
        if idx.size < 2:
            return float("nan")
        azs = chain_azimuth_deg(lats[idx[:-1]], lons[idx[:-1]], lats[idx[1:]], lons[idx[1:]])
        return float(np.mod(np.rad2deg(np.arctan2(np.mean(np.sin(np.deg2rad(azs))),
                                                  np.mean(np.cos(np.deg2rad(azs))))), 360.0))

    print(f"mean chain azimuth — Hawaiian: {mean_az(is_hawaiian.to_numpy()):.0f}°, "
          f"Emperor: {mean_az(is_emperor.to_numpy()):.0f}°")


def main() -> None:
    df = load_seamount_catalog(DATA_PATH)
    print(f"loaded {len(df)} seamounts from {DATA_PATH.relative_to(REPO_ROOT)}")
    _segment_stats(df)

    fig = plot_hawaii_emperor_figure(df, bend_age_Ma=BEND_AGE_MA)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

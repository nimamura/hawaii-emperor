"""2×2 Hawaii–Emperor bend figure.

Layout:
  A  map of the North Pacific with seamounts coloured by age and a
     great-circle bend marker at Daikakuji.
  B  age (Ma) vs along-chain distance from Kilauea (km). Two OLS lines
     are fit independently to Hawaiian+Bend and Emperor seamounts; the
     published ~47 Ma bend age is drawn as a vertical reference.
  C  chain azimuth (°) as a function of age, computed from a sliding
     window of consecutive seamounts in age order. The 47 Ma step is
     the core of the story.
  D  apparent plate speed (cm/yr) along the chain, from a running slope
     of distance vs age. Noisier than B/C but makes the point that the
     speed is broadly comparable on both sides of the bend.

Plain matplotlib only — no cartopy. The geographical domain is narrow
enough (roughly 150°E–210°E, 15°N–60°N) that a rectangular PlateCarrée
rendering with a hand-drawn Pacific rim reads cleanly for an X-post
audience.
"""
from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.geometry import (
    BEND_LAT,
    BEND_LON,
    KILAUEA_LAT,
    KILAUEA_LON,
    apparent_speed_cm_per_yr,
    chain_azimuth_deg,
    chain_distance_via_bend_km,
    great_circle_distance_km,
)

# Labels to annotate in the map panel. (name, corner of label relative to point).
MAP_LABELS: list[tuple[str, str]] = [
    ("Kilauea", "tr"),
    ("Midway", "tr"),
    ("Suiko South", "tr"),
    ("Meiji", "tr"),
]

# Approximate Pacific-rim coastline polyline (sparse, hand-traced) for
# the map panel. Longitudes are in [140, 240] (east-positive, crossing the
# dateline), latitudes in degrees north.
_RIM = np.array(
    [
        # Japan / Kuril / Kamchatka
        (140.0, 35.0), (141.0, 39.0), (141.5, 41.5), (143.0, 43.5),
        (146.0, 45.0), (150.0, 46.5), (153.0, 49.0), (156.0, 51.0),
        (159.0, 53.0), (162.0, 54.5), (165.0, 56.0), (168.0, 57.5),
        (172.0, 59.0),
        # Aleutians
        (180.0, 53.0), (190.0, 52.5), (200.0, 54.0), (210.0, 56.0),
        (215.0, 58.0),
    ]
)


def _draw_pacific_rim(ax: plt.Axes) -> None:
    # Close the polygon into the top-left so the land fill sits cleanly in
    # the NW corner of the map without bleeding into the open ocean.
    poly_lon = np.concatenate([_RIM[:, 0], [_RIM[-1, 0], _RIM[0, 0]]])
    poly_lat = np.concatenate([_RIM[:, 1], [65.0, 65.0]])
    ax.fill(poly_lon, poly_lat, color="#f1ede2", edgecolor="#b7b0a0",
            linewidth=0.7, zorder=1)


def _fit_segment(ages: np.ndarray, dists: np.ndarray) -> tuple[float, float]:
    """OLS slope and intercept of ``dists = a + s * ages``."""
    n = ages.size
    if n < 2:
        return 0.0, 0.0
    A = np.column_stack([np.ones(n), ages])
    (a, s), *_ = np.linalg.lstsq(A, dists, rcond=None)
    return float(s), float(a)


def _running_azimuth(
    ages_Ma: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    age_window_Myr: float = 10.0,
) -> np.ndarray:
    """Bearing averaged over seamounts within ±``age_window_Myr`` of each point.

    Using an age-window (rather than a fixed index window) keeps the
    azimuth estimate stable in the densely-sampled young end of the chain
    where many Big-Island-scale points sit at almost identical ages.
    """
    n = ages_Ma.size
    az = np.full(n, np.nan)
    for i in range(n):
        lo_mask = ages_Ma <= ages_Ma[i] - age_window_Myr / 2
        hi_mask = ages_Ma >= ages_Ma[i] + age_window_Myr / 2
        lo_idx = np.where(lo_mask)[0]
        hi_idx = np.where(hi_mask)[0]
        lo = lo_idx[-1] if lo_idx.size else max(0, i - 2)
        hi = hi_idx[0] if hi_idx.size else min(n - 1, i + 2)
        if hi <= lo:
            continue
        az[i] = chain_azimuth_deg(lats[lo], lons[lo], lats[hi], lons[hi])
    return az


def _smoothed_speed_cm_per_yr(
    ages_Ma: np.ndarray,
    dists_km: np.ndarray,
    age_window_Myr: float = 10.0,
) -> np.ndarray:
    """Apparent speed from a sliding-window linear fit of dist vs age.

    The direct numerical derivative in ``apparent_speed_cm_per_yr`` is
    dominated by tiny age gaps between volcanoes on the same island (e.g.
    Kilauea/Mauna Loa/Hualalai). Fitting a line in a ±5 Myr window averages
    that scatter out.
    """
    n = ages_Ma.size
    out = np.full(n, np.nan)
    for i in range(n):
        mask = np.abs(ages_Ma - ages_Ma[i]) <= age_window_Myr / 2
        if mask.sum() < 3:
            # Fall back to nearest-neighbours when the window is sparse.
            k = 3
            lo = max(0, i - k)
            hi = min(n, i + k + 1)
            mask = np.zeros(n, dtype=bool)
            mask[lo:hi] = True
        x = ages_Ma[mask]
        y = dists_km[mask]
        A = np.column_stack([np.ones_like(x), x])
        (a, s), *_ = np.linalg.lstsq(A, y, rcond=None)
        out[i] = abs(float(s)) * 0.1  # km/Myr → cm/yr
    return out


def plot_hawaii_emperor_figure(
    df: pd.DataFrame,
    *,
    bend_age_Ma: float = 47.0,
    cmap: str = "plasma",
    title: str | None = (
        "Hawaii–Emperor Bend — the 47 Ma kink in a 6000 km hotspot track"
    ),
) -> plt.Figure:
    """Build the 2×2 figure. ``df`` must follow ``src.catalog`` schema."""
    required = {"name", "chain", "lon", "lat", "age_Ma", "age_err_Ma"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    df = df.sort_values("age_Ma").reset_index(drop=True)
    ages = df["age_Ma"].to_numpy()
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    chains = df["chain"].to_numpy()

    dist_km = chain_distance_via_bend_km(lats, lons, chain=chains)

    fig = plt.figure(figsize=(13.0, 10.0), facecolor="white")
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        left=0.06, right=0.97, top=0.93, bottom=0.07,
        wspace=0.24, hspace=0.32,
    )

    ax_map = fig.add_subplot(gs[0, 0], label="A")
    ax_dist = fig.add_subplot(gs[0, 1], label="B")
    ax_azi = fig.add_subplot(gs[1, 0], label="C")
    ax_vel = fig.add_subplot(gs[1, 1], label="D")

    vmin = 0.0
    vmax = float(np.ceil(ages.max() / 10) * 10)

    _draw_panel_map(ax_map, df, vmin=vmin, vmax=vmax, cmap=cmap)
    _draw_panel_distance(ax_dist, ages, dist_km, chains, bend_age_Ma, cmap=cmap, vmax=vmax)
    _draw_panel_azimuth(ax_azi, ages, lats, lons, bend_age_Ma)
    _draw_panel_speed(ax_vel, ages, dist_km, bend_age_Ma)

    for ax, letter, loc in [
        (ax_map, "A", "tl"),
        (ax_dist, "B", "tl"),
        (ax_azi, "C", "tl"),
        (ax_vel, "D", "tl"),
    ]:
        _panel_tag(ax, letter, loc)

    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.985)

    return fig


# ---- individual panels ------------------------------------------------------


def _draw_panel_map(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
) -> None:
    ages = df["age_Ma"].to_numpy()
    lons = df["lon"].to_numpy()
    lats = df["lat"].to_numpy()

    ax.set_facecolor("#f6f8fc")
    ax.set_xlim(142.0, 215.0)
    ax.set_ylim(15.0, 60.0)

    _draw_pacific_rim(ax)

    # Great-circle from Kilauea to bend to Meiji-ish (reference polyline).
    ref_lons = np.concatenate(
        [
            _gc_sample(KILAUEA_LAT, KILAUEA_LON, BEND_LAT, BEND_LON, n=40)[1],
            _gc_sample(BEND_LAT, BEND_LON, 53.4, 164.4, n=40)[1],
        ]
    )
    ref_lats = np.concatenate(
        [
            _gc_sample(KILAUEA_LAT, KILAUEA_LON, BEND_LAT, BEND_LON, n=40)[0],
            _gc_sample(BEND_LAT, BEND_LON, 53.4, 164.4, n=40)[0],
        ]
    )
    ax.plot(ref_lons, ref_lats, color="#bbb", lw=1.0, ls="--", zorder=1)

    sc = ax.scatter(
        lons, lats, c=ages, cmap=cmap, vmin=vmin, vmax=vmax,
        s=28, edgecolor="white", linewidth=0.5, zorder=3,
    )

    # Bend marker.
    ax.plot(
        [BEND_LON], [BEND_LAT],
        marker="*", ms=16, color="#cc3344", mec="white", mew=1.0, zorder=4,
    )
    ax.annotate(
        "47 Ma bend (Daikakuji)",
        xy=(BEND_LON, BEND_LAT),
        xytext=(-10, -20),
        textcoords="offset points",
        fontsize=10,
        color="#cc3344",
        fontweight="bold",
        ha="right",
    )

    # A handful of named seamounts.
    for name, anchor in MAP_LABELS:
        match = df[df["name"] == name]
        if match.empty:
            continue
        row = match.iloc[0]
        dx, dy, ha, va = _anchor_offset(anchor)
        ax.annotate(
            row["name"],
            xy=(row["lon"], row["lat"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            ha=ha,
            va=va,
            color="#333",
        )

    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    ax.grid(True, color="#dddddd", lw=0.5)

    cbar = ax.figure.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("seamount age (Ma)")


def _draw_panel_distance(
    ax: plt.Axes,
    ages: np.ndarray,
    dists_km: np.ndarray,
    chains: np.ndarray,
    bend_age_Ma: float,
    *,
    cmap: str,
    vmax: float,
) -> None:
    ax.scatter(
        ages, dists_km, c=ages, cmap=cmap, vmin=0.0, vmax=vmax,
        s=30, edgecolor="white", linewidth=0.5, zorder=3,
    )

    haw_mask = np.isin(chains, ["Hawaiian", "Bend"])
    emp_mask = chains == "Emperor"

    if haw_mask.any():
        s_h, a_h = _fit_segment(ages[haw_mask], dists_km[haw_mask])
        x = np.linspace(0.0, max(ages[haw_mask].max(), bend_age_Ma), 50)
        ax.plot(x, a_h + s_h * x, color="#2c6fb0", lw=2.0, label=f"Hawaiian: {s_h*0.1:.1f} cm/yr")
    if emp_mask.any():
        s_e, a_e = _fit_segment(ages[emp_mask], dists_km[emp_mask])
        x = np.linspace(bend_age_Ma, ages.max(), 50)
        ax.plot(x, a_e + s_e * x, color="#b01a3a", lw=2.0, label=f"Emperor: {s_e*0.1:.1f} cm/yr")

    ax.axvline(bend_age_Ma, color="#888", ls=":", lw=1.0)
    ax.text(
        bend_age_Ma + 0.8, ax.get_ylim()[1] * 0.05,
        f"{bend_age_Ma:.0f} Ma", color="#666", fontsize=9, va="bottom",
    )

    ax.set_xlabel("age (Ma)")
    ax.set_ylabel("along-chain distance from Kilauea (km)")
    ax.set_xlim(-2, ages.max() * 1.04)
    ax.set_ylim(0, dists_km.max() * 1.07)
    ax.grid(True, color="#dddddd", lw=0.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9)


def _draw_panel_azimuth(
    ax: plt.Axes,
    ages: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    bend_age_Ma: float,
) -> None:
    az = _running_azimuth(ages, lats, lons, age_window_Myr=12.0)
    ax.plot(ages, az, marker="o", ms=4, lw=1.2, color="#333")

    ax.axvline(bend_age_Ma, color="#888", ls=":", lw=1.0)
    ax.axhspan(285, 310, color="#2c6fb0", alpha=0.14, zorder=0)
    ax.axhspan(340, 360, color="#b01a3a", alpha=0.14, zorder=0)

    ax.text(5.0, 303, "Hawaiian chain\nWNW (~300°)", color="#2c6fb0", fontsize=9)
    ax.text(70.0, 353, "Emperor chain\n~N (~350°)", color="#b01a3a",
            fontsize=9, ha="center")
    ax.text(bend_age_Ma + 0.8, 270, f"{bend_age_Ma:.0f} Ma",
            color="#666", fontsize=9, va="bottom")

    ax.set_xlabel("age (Ma)")
    ax.set_ylabel("chain azimuth (°)")
    ax.set_xlim(-2, ages.max() * 1.04)
    ax.set_ylim(265, 365)
    ax.grid(True, color="#dddddd", lw=0.5)


def _draw_panel_speed(
    ax: plt.Axes,
    ages: np.ndarray,
    dists_km: np.ndarray,
    bend_age_Ma: float,
) -> None:
    # Noisy raw derivative as a background trace.
    v_raw = apparent_speed_cm_per_yr(ages, dists_km)
    ax.plot(ages, v_raw, color="#cccccc", lw=0.6, zorder=1)

    # Smoothed estimate via sliding-window OLS.
    v = _smoothed_speed_cm_per_yr(ages, dists_km, age_window_Myr=12.0)
    ax.plot(ages, v, marker="o", ms=4, lw=1.4, color="#333", zorder=3)

    ax.axhline(8.0, color="#999", ls="--", lw=0.8)
    ax.text(ages.max() * 0.99, 8.1, "~8 cm/yr", color="#666",
            va="bottom", ha="right", fontsize=9)
    ax.axvline(bend_age_Ma, color="#888", ls=":", lw=1.0)
    ax.text(bend_age_Ma + 0.8, 0.4, f"{bend_age_Ma:.0f} Ma",
            color="#666", fontsize=9, va="bottom")

    ax.set_xlabel("age (Ma)")
    ax.set_ylabel("apparent plate speed (cm/yr)")
    ax.set_xlim(-2, ages.max() * 1.04)
    ax.set_ylim(0, 14.0)
    ax.grid(True, color="#dddddd", lw=0.5)


# ---- helpers ---------------------------------------------------------------


def _gc_sample(
    lat1: float, lon1: float, lat2: float, lon2: float, n: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a great-circle path between two endpoints (for drawing)."""
    phi1, phi2 = np.deg2rad([lat1, lat2])
    lam1, lam2 = np.deg2rad([lon1, lon2])
    d = great_circle_distance_km(lat1, lon1, lat2, lon2) / 6371.0088
    if d < 1e-9:
        return np.array([lat1]), np.array([lon1])
    f = np.linspace(0.0, 1.0, n)
    a = np.sin((1 - f) * d) / np.sin(d)
    b = np.sin(f * d) / np.sin(d)
    x = a * np.cos(phi1) * np.cos(lam1) + b * np.cos(phi2) * np.cos(lam2)
    y = a * np.cos(phi1) * np.sin(lam1) + b * np.cos(phi2) * np.sin(lam2)
    z = a * np.sin(phi1) + b * np.sin(phi2)
    lat = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
    lon = np.rad2deg(np.arctan2(y, x)) % 360.0
    return lat, lon


def _anchor_offset(anchor: Literal["tl", "tr", "bl", "br"]) -> tuple[int, int, str, str]:
    return {
        "tl": (-6, 6, "right", "bottom"),
        "tr": (6, 6, "left", "bottom"),
        "bl": (-6, -6, "right", "top"),
        "br": (6, -6, "left", "top"),
    }[anchor]


def _panel_tag(ax: plt.Axes, letter: str, loc: str = "tl") -> None:
    x, y, ha, va = {
        "tl": (0.015, 0.965, "left", "top"),
        "tr": (0.985, 0.965, "right", "top"),
        "bl": (0.015, 0.035, "left", "bottom"),
        "br": (0.985, 0.035, "right", "bottom"),
    }[loc]
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=14, fontweight="bold", color="#222",
        ha=ha, va=va,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5, "alpha": 0.85},
    )

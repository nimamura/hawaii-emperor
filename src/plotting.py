"""2×2 Hawaii–Emperor bend figure.

Layout:
  A  map of the North Pacific with ETOPO 2022 bathymetry as backdrop,
     seamounts coloured by age, plate-motion arrows for each segment,
     and the 47 Ma bend starred at Daikakuji.
  B  age (Ma) vs along-chain distance from Kilauea (km). Two OLS lines
     are fit independently to Hawaiian+Bend and Emperor seamounts; the
     published ~47 Ma bend age is drawn as a vertical reference.
  C  chain azimuth (°) as a function of age. The 47 Ma step from
     ~300° (WNW) to ~350° (~N) is the core of the story.
  D  apparent plate speed (cm/yr). Smoothed sliding-window slope of
     distance vs age, with a noisier raw derivative in the background.

Japanese subtitles on each panel are there so a non-specialist reader
can parse the figure without reading any body text.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.patches import FancyArrowPatch

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

# Use a Japanese-capable sans-serif if it's installed (Hiragino Sans on macOS).
# Falls back silently to DejaVu Sans so tests still run on CI.
_PREFERRED_JP_FONTS = ["Hiragino Sans", "Hiragino Maru Gothic Pro", "YuGothic", "Noto Sans CJK JP"]
mpl.rcParams["font.sans-serif"] = _PREFERRED_JP_FONTS + list(mpl.rcParams["font.sans-serif"])
mpl.rcParams["axes.unicode_minus"] = False

# Default path for the pre-cropped ETOPO 2022 tile. Created by
# ``scripts/prepare_etopo_crop.py``. If missing, the map panel falls back
# to a coastline-only rendering.
DEFAULT_ETOPO_PATH = Path(__file__).resolve().parent.parent / "data" / "etopo_north_pacific.npz"

# Labels to annotate on the map. (seamount_name, anchor_corner).
MAP_LABELS: list[tuple[str, str]] = [
    ("Kilauea", "tr"),
    ("Midway", "tr"),
    ("Suiko South", "tr"),
    ("Meiji", "tr"),
]


# ---------- bathymetry colour map ------------------------------------------

def _bathymetry_cmap() -> LinearSegmentedColormap:
    """Ocean-biased GMT-style colour ramp from deep blue → tan land."""
    # Breakpoints chosen for North Pacific seamount topography. Values are
    # ``(normalised_elevation ∈ [0, 1], colour)`` with 0 = -9000 m and
    # 1 = +4000 m, so sea level sits at ~0.69.
    stops = [
        (0.00, "#0b1e3a"),  # trench / abyssal floor
        (0.18, "#102c52"),
        (0.35, "#1a4576"),
        (0.55, "#3b78b0"),  # mid-ocean ridges
        (0.69, "#b7dff5"),  # shelf / sea level
        (0.70, "#f1e9cf"),  # coastal
        (0.80, "#c2b488"),
        (1.00, "#7a6948"),  # mountains
    ]
    return LinearSegmentedColormap.from_list("hawaii_bathy", stops, N=512)


# ---------- small numerical helpers ----------------------------------------

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
    age_window_Myr: float = 12.0,
) -> np.ndarray:
    """Bearing averaged over seamounts within ±``age_window_Myr`` of each point."""
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
    age_window_Myr: float = 12.0,
) -> np.ndarray:
    """Apparent speed from a sliding-window linear fit of dist vs age."""
    n = ages_Ma.size
    out = np.full(n, np.nan)
    for i in range(n):
        mask = np.abs(ages_Ma - ages_Ma[i]) <= age_window_Myr / 2
        if mask.sum() < 3:
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


# ---------- figure entry point ----------------------------------------------

def plot_hawaii_emperor_figure(
    df: pd.DataFrame,
    *,
    bend_age_Ma: float = 47.0,
    cmap: str = "plasma",
    etopo_path: Path | None = None,
    title: str | None = (
        "ハワイ–皇帝海山列：47 Ma に太平洋プレートが進路を 60° 変えた瞬間の化石\n"
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

    fig = plt.figure(figsize=(14.5, 10.5), facecolor="white")
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        left=0.055, right=0.97, top=0.90, bottom=0.075,
        wspace=0.22, hspace=0.42,
        width_ratios=[1.25, 1.0],
    )

    ax_map = fig.add_subplot(gs[0, 0], label="A")
    ax_dist = fig.add_subplot(gs[0, 1], label="B")
    ax_azi = fig.add_subplot(gs[1, 0], label="C")
    ax_vel = fig.add_subplot(gs[1, 1], label="D")

    vmin = 0.0
    vmax = float(np.ceil(ages.max() / 10) * 10)

    if etopo_path is None:
        etopo_path = DEFAULT_ETOPO_PATH

    _draw_panel_map(
        ax_map, df, vmin=vmin, vmax=vmax, cmap=cmap,
        bend_age_Ma=bend_age_Ma, etopo_path=etopo_path,
    )
    _draw_panel_distance(ax_dist, ages, dist_km, chains, bend_age_Ma, cmap=cmap, vmax=vmax)
    _draw_panel_azimuth(ax_azi, ages, lats, lons, bend_age_Ma)
    _draw_panel_speed(ax_vel, ages, dist_km, bend_age_Ma)

    for ax, letter in [(ax_map, "A"), (ax_dist, "B"), (ax_azi, "C"), (ax_vel, "D")]:
        _panel_tag(ax, letter)

    if title is not None:
        fig.suptitle(title, fontsize=13.5, y=0.975)

    return fig


# ---------- map (panel A) ---------------------------------------------------

def _load_etopo(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return ``(lon, lat, z)`` or ``None`` if the cached tile is missing."""
    if not path.exists():
        return None
    with np.load(path) as npz:
        return npz["lon"].copy(), npz["lat"].copy(), npz["z"].copy()


def _draw_panel_map(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    bend_age_Ma: float,
    etopo_path: Path,
) -> None:
    ages = df["age_Ma"].to_numpy()
    lons = df["lon"].to_numpy()
    lats = df["lat"].to_numpy()

    ax.set_xlim(142.0, 215.0)
    ax.set_ylim(15.0, 60.0)

    bathy = _load_etopo(etopo_path)
    if bathy is not None:
        bathy_lon, bathy_lat, z = bathy
        # Normalise to [0, 1] with sea level at the land/sea break.
        z_vmin, z_vmax = -9000.0, 4000.0
        norm_z = np.clip((z - z_vmin) / (z_vmax - z_vmin), 0.0, 1.0)

        # Add a hillshade so the seafloor texture reads as 3-D. The
        # vertical exaggeration is tuned so seamounts "pop" off the
        # abyssal plain without the Kuril Trench over-saturating.
        ls = LightSource(azdeg=315, altdeg=40)
        shaded = ls.shade(
            norm_z, cmap=_bathymetry_cmap(),
            blend_mode="overlay", vert_exag=90, fraction=1.0,
        )
        ax.imshow(
            shaded,
            extent=[bathy_lon.min(), bathy_lon.max(), bathy_lat.min(), bathy_lat.max()],
            origin="lower",
            zorder=0,
            interpolation="bilinear",
        )
    else:
        ax.set_facecolor("#e8eef8")

    # Great-circle reference line through the bend (faint guide).
    ref_lats_a, ref_lons_a = _gc_sample(KILAUEA_LAT, KILAUEA_LON, BEND_LAT, BEND_LON, n=60)
    ref_lats_b, ref_lons_b = _gc_sample(BEND_LAT, BEND_LON, 53.4, 164.4, n=60)
    ax.plot(ref_lons_a, ref_lats_a, color="white", lw=1.6, alpha=0.6, zorder=2)
    ax.plot(ref_lons_b, ref_lats_b, color="white", lw=1.6, alpha=0.6, zorder=2)

    # Seamount points, coloured by age.
    sc = ax.scatter(
        lons, lats, c=ages, cmap=cmap, vmin=vmin, vmax=vmax,
        s=40, edgecolor="white", linewidth=0.7, zorder=4,
    )

    # Bend marker.
    ax.plot(
        [BEND_LON], [BEND_LAT],
        marker="*", ms=20, color="#ffd44b", mec="#7a3a00", mew=1.2, zorder=5,
    )
    ax.annotate(
        f"{bend_age_Ma:.0f} Ma bend\n(Daikakuji)",
        xy=(BEND_LON, BEND_LAT),
        xytext=(-12, -26),
        textcoords="offset points",
        fontsize=10,
        color="#ffe27a",
        fontweight="bold",
        ha="right",
        bbox={"facecolor": "#3a1a00", "edgecolor": "none", "pad": 3.0, "alpha": 0.75},
    )

    # Plate-motion arrows, offset into open ocean so they don't cover
    # the chain itself. Each arrow points in the direction the Pacific
    # plate was moving during that era (the direction in which young
    # volcanoes "age away" along the chain).
    _plate_arrow(
        ax,
        start=(208.0, 17.0),
        end=(192.0, 21.0),
        color="#ffffff",
        label="47 Ma→現在\nプレートは WNW へ\n~8 cm/yr",
        label_offset=(0.0, -3.5),
    )
    _plate_arrow(
        ax,
        start=(150.5, 35.0),
        end=(150.5, 50.5),
        color="#ffffff",
        label="82 → 47 Ma\nプレートは北へ\n~6 cm/yr",
        label_offset=(2.5, 0.0),
        label_anchor="left",
    )

    # Small legend in the top-right corner.
    ax.text(
        0.988, 0.015,
        "→ は古代のプレート進行方向\narrows: past plate motion",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        color="white",
        bbox={"facecolor": "#222222", "edgecolor": "none", "pad": 3.0, "alpha": 0.8},
    )

    # Named seamounts.
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
            color="white",
            path_effects=[],
            bbox={"facecolor": "#000000", "edgecolor": "none", "pad": 1.8, "alpha": 0.55},
        )

    ax.set_xlabel("経度 longitude (°E)")
    ax.set_ylabel("緯度 latitude (°N)")
    ax.set_title("北太平洋の海底地形と海山の年代 (ETOPO 2022 bathymetry)",
                 fontsize=11, pad=5)
    ax.grid(True, color="white", lw=0.3, alpha=0.20)

    # Slim colorbar so it doesn't eat into the map area.
    cbar = ax.figure.colorbar(sc, ax=ax, pad=0.015, fraction=0.032, aspect=28)
    cbar.set_label("海山の年代 age (Ma)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def _plate_arrow(
    ax: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
    label_anchor: Literal["center", "left", "right"] = "center",
) -> None:
    # Halo first (bigger, darker) so the white arrow pops against any background.
    halo = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=30,
        color="#1a1a1a", lw=6.0, zorder=5.5, alpha=0.7,
    )
    ax.add_patch(halo)
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=26,
        color=color, lw=3.2, zorder=6,
    )
    ax.add_patch(arrow)

    if label is not None:
        lon = (start[0] + end[0]) / 2 + label_offset[0]
        lat = (start[1] + end[1]) / 2 + label_offset[1]
        ha = {"center": "center", "left": "left", "right": "right"}[label_anchor]
        ax.text(
            lon, lat, label,
            color="white", fontsize=9.5, ha=ha, va="center",
            fontweight="bold",
            bbox={"facecolor": "#1a1a1a", "edgecolor": "none",
                  "pad": 3.5, "alpha": 0.82},
            zorder=7,
        )


# ---------- age-distance (panel B) -----------------------------------------

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
        s=34, edgecolor="white", linewidth=0.5, zorder=3,
    )

    haw_mask = np.isin(chains, ["Hawaiian", "Bend"])
    emp_mask = chains == "Emperor"

    if haw_mask.any():
        s_h, a_h = _fit_segment(ages[haw_mask], dists_km[haw_mask])
        x = np.linspace(0.0, bend_age_Ma, 50)
        ax.plot(x, a_h + s_h * x, color="#2c6fb0", lw=2.2,
                label=f"Hawaiian chain: {s_h*0.1:.1f} cm/yr")
    if emp_mask.any():
        s_e, a_e = _fit_segment(ages[emp_mask], dists_km[emp_mask])
        x = np.linspace(bend_age_Ma, ages.max(), 50)
        ax.plot(x, a_e + s_e * x, color="#b01a3a", lw=2.2,
                label=f"Emperor chain: {s_e*0.1:.1f} cm/yr")

    ax.axvline(bend_age_Ma, color="#cc3344", ls="--", lw=1.2, alpha=0.7, zorder=2)

    # Arrow highlighting the kink.
    ymax = dists_km.max() * 1.07
    ax.annotate(
        "ここで折れ曲がる\n(the kink)",
        xy=(bend_age_Ma, 3500),
        xytext=(bend_age_Ma + 15, 1800),
        textcoords="data",
        fontsize=10, color="#cc3344", fontweight="bold",
        ha="left", va="center",
        arrowprops={"arrowstyle": "->", "color": "#cc3344", "lw": 1.4},
    )

    ax.set_xlabel("年代 age (Ma)")
    ax.set_ylabel("Kilaueaからの距離 along-chain distance (km)")
    ax.set_title("海山の年代 vs 距離   Age vs along-chain distance",
                 fontsize=10.5, pad=5)
    ax.set_xlim(-2, ages.max() * 1.04)
    ax.set_ylim(0, ymax)
    ax.grid(True, color="#dddddd", lw=0.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    ax.text(bend_age_Ma + 0.8, 150, f"{bend_age_Ma:.0f} Ma",
            color="#cc3344", fontsize=9, va="bottom")


# ---------- azimuth (panel C) ----------------------------------------------

def _draw_panel_azimuth(
    ax: plt.Axes,
    ages: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    bend_age_Ma: float,
) -> None:
    az = _running_azimuth(ages, lats, lons, age_window_Myr=12.0)
    ax.plot(ages, az, marker="o", ms=4, lw=1.4, color="#222")

    ax.axvline(bend_age_Ma, color="#cc3344", ls="--", lw=1.2, alpha=0.7)

    ax.axhspan(285, 315, color="#2c6fb0", alpha=0.16, zorder=0)
    ax.axhspan(340, 365, color="#b01a3a", alpha=0.16, zorder=0)

    # Compass hint on the left margin.
    for deg, label in [(270, "W"), (300, "WNW"), (330, "NNW"), (360, "N")]:
        ax.axhline(deg, color="#cccccc", lw=0.5, zorder=1)
        ax.text(-1.2, deg, label, color="#888", fontsize=8, va="center", ha="right")

    ax.text(8.0, 302, "Hawaiian = WNW ~ 300°\n(プレートが西微北へ動いた)",
            color="#2c6fb0", fontsize=9.5, fontweight="bold")
    ax.text(55.0, 352, "Emperor ~ N ~ 350°\n(プレートがほぼ北へ動いた)",
            color="#b01a3a", fontsize=9.5, fontweight="bold", ha="left")

    # "Direction jump" arrow.
    ax.annotate(
        "",
        xy=(bend_age_Ma - 0.5, 350), xytext=(bend_age_Ma - 0.5, 305),
        arrowprops={"arrowstyle": "-|>", "color": "#cc3344", "lw": 2.0},
    )
    ax.text(bend_age_Ma + 1.5, 325, "47 Ma に\n向きが急変\ndirection jumps",
            color="#cc3344", fontsize=9.5, fontweight="bold",
            ha="left", va="center")

    ax.set_xlabel("年代 age (Ma)")
    ax.set_ylabel("列の方位 chain azimuth (°)")
    ax.set_title("鎖状火山列の方位角   Chain direction vs age",
                 fontsize=10.5, pad=5)
    ax.set_xlim(-2, ages.max() * 1.04)
    ax.set_ylim(265, 370)
    ax.grid(True, color="#dddddd", lw=0.5)


# ---------- speed (panel D) ------------------------------------------------

def _draw_panel_speed(
    ax: plt.Axes,
    ages: np.ndarray,
    dists_km: np.ndarray,
    bend_age_Ma: float,
) -> None:
    v_raw = apparent_speed_cm_per_yr(ages, dists_km)
    ax.plot(ages, v_raw, color="#cccccc", lw=0.6, zorder=1,
            label="raw derivative")

    v = _smoothed_speed_cm_per_yr(ages, dists_km, age_window_Myr=12.0)
    ax.plot(ages, v, marker="o", ms=4, lw=1.6, color="#222", zorder=3,
            label="12 Myr window")

    ax.axhspan(5, 11, color="#d6c94a", alpha=0.15, zorder=0)
    ax.axhline(8.0, color="#a88b00", ls="--", lw=1.0, zorder=2)
    ax.text(ages.max() * 0.99, 8.15, "~ 8 cm/yr (指の爪が伸びる速さ)",
            color="#7a6a00", va="bottom", ha="right", fontsize=9)

    ax.axvline(bend_age_Ma, color="#cc3344", ls="--", lw=1.2, alpha=0.7)

    ax.annotate(
        "方向は急変したが\n速さはほぼそのまま",
        xy=(bend_age_Ma, 6.0),
        xytext=(bend_age_Ma + 10, 12.4),
        textcoords="data",
        fontsize=10, color="#222", fontweight="bold",
        ha="left", va="top",
        arrowprops={"arrowstyle": "->", "color": "#888", "lw": 1.0},
    )
    ax.text(bend_age_Ma + 0.8, 0.3, f"{bend_age_Ma:.0f} Ma",
            color="#cc3344", fontsize=9, va="bottom")

    ax.set_xlabel("年代 age (Ma)")
    ax.set_ylabel("見かけのプレート速度 speed (cm/yr)")
    ax.set_title("プレート速度はどう変わったか   Apparent plate speed vs age",
                 fontsize=10.5, pad=5)
    ax.set_xlim(-2, ages.max() * 1.04)
    ax.set_ylim(0, 14.0)
    ax.grid(True, color="#dddddd", lw=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)


# ---------- helpers --------------------------------------------------------

def _gc_sample(
    lat1: float, lon1: float, lat2: float, lon2: float, n: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a great-circle path between two endpoints."""
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


def _panel_tag(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.012, 0.972, letter,
        transform=ax.transAxes,
        fontsize=15, fontweight="bold", color="#111",
        ha="left", va="top",
        bbox={"facecolor": "white", "edgecolor": "#888888",
              "pad": 3.2, "linewidth": 0.7, "alpha": 0.95},
    )

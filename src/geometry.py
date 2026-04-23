"""Great-circle geometry + broken-stick changepoint fit.

All functions are pure numpy — no dependency on cartopy. Longitudes may be
given in either ±180 or [0, 360) form; trig handles the difference.

The broken-stick fit is the core of the 47 Ma story: we search for the age at
which a two-segment linear model of ``distance_km ~ age_Ma`` minimises its
residual sum-of-squares. Because the function is non-smooth in the breakpoint,
we use a 1-D grid search over ``break_bounds`` — this is accurate enough given
~100 data points and cheaper than a general-purpose optimiser.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Earth radius (mean, km) — enough precision for a 6000 km track.
EARTH_RADIUS_KM: float = 6371.0088

# Kilauea, the currently active Hawaiian volcano and the natural distance origin.
KILAUEA_LAT: float = 19.4069
KILAUEA_LON: float = 204.7104  # = -155.2896E in [0, 360).

# Daikakuji Seamount, the classical "bend" marker in the Hawaii–Emperor chain.
# O'Connor et al. (2013) place the morphological bend within a handful of
# seamounts around 32°N, 172.3°E.
BEND_LAT: float = 32.08
BEND_LON: float = 172.30


def _to_rad(x: float | np.ndarray) -> np.ndarray:
    return np.deg2rad(np.asarray(x, dtype=float))


def great_circle_distance_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Haversine distance between ``(lat1, lon1)`` and ``(lat2, lon2)`` in km."""
    phi1 = _to_rad(lat1)
    phi2 = _to_rad(lat2)
    dphi = phi2 - phi1
    dlam = _to_rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


def chain_azimuth_deg(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Great-circle bearing (°) from point 1 to point 2, returned in [0, 360)."""
    phi1 = _to_rad(lat1)
    phi2 = _to_rad(lat2)
    dlam = _to_rad(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    az = np.rad2deg(np.arctan2(x, y))
    return np.mod(az, 360.0)


def chain_distance_via_bend_km(
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    *,
    chain: str | np.ndarray,
) -> float | np.ndarray:
    """Along-chain distance from Kilauea (km), routed via Daikakuji for Emperor.

    For Hawaiian and Bend seamounts (age ≲ 48 Ma) the chain is well
    approximated by a great circle, so direct ``great_circle_distance_km``
    from Kilauea is correct. Emperor seamounts sit on the other limb of the
    ~60° bend, so we use ``dist(Kilauea, bend) + dist(bend, seamount)`` which
    is the classical "along-chain" measure plotted in Morgan 1971 and
    O'Connor 2013 against age.
    """
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    chain_arr = np.asarray(chain)

    direct = great_circle_distance_km(KILAUEA_LAT, KILAUEA_LON, lat_arr, lon_arr)
    leg1 = great_circle_distance_km(KILAUEA_LAT, KILAUEA_LON, BEND_LAT, BEND_LON)
    leg2 = great_circle_distance_km(BEND_LAT, BEND_LON, lat_arr, lon_arr)
    via = leg1 + leg2

    is_emperor = chain_arr == "Emperor"
    out = np.where(is_emperor, via, direct)
    # Preserve scalar return for scalar input.
    if out.ndim == 0:
        return float(out)
    return out


@dataclass(frozen=True)
class BrokenStickFit:
    break_age_Ma: float
    dist_at_break_km: float
    slope_young: float  # km per Myr on the young (Hawaiian) side
    slope_old: float    # km per Myr on the old (Emperor) side
    rss: float          # residual sum of squares
    residuals: np.ndarray


def fit_broken_stick(
    ages_Ma: np.ndarray,
    dists_km: np.ndarray,
    *,
    break_bounds: tuple[float, float] = (25.0, 70.0),
    break_grid_step_Ma: float = 0.25,
) -> BrokenStickFit:
    """Fit a continuous piecewise-linear model with one interior break.

    The model is ``d = a + s_y * age`` for ``age <= tau`` and
    ``d = a + s_y * tau + s_o * (age - tau)`` for ``age > tau``. For each
    candidate ``tau`` we solve a 3-parameter OLS and pick the ``tau`` with
    smallest residual sum of squares.
    """
    ages = np.asarray(ages_Ma, dtype=float)
    dists = np.asarray(dists_km, dtype=float)
    if ages.shape != dists.shape:
        raise ValueError("ages_Ma and dists_km must have the same shape")

    taus = np.arange(break_bounds[0], break_bounds[1] + 1e-9, break_grid_step_Ma)

    best: tuple[float, np.ndarray, float] | None = None  # (rss, beta, tau)
    for tau in taus:
        # Design matrix columns: 1, age, max(age - tau, 0). Continuous by construction.
        x1 = ages
        x2 = np.maximum(ages - tau, 0.0)
        X = np.column_stack([np.ones_like(ages), x1, x2])
        beta, residuals, rank, _ = np.linalg.lstsq(X, dists, rcond=None)
        pred = X @ beta
        rss = float(np.sum((dists - pred) ** 2))
        if best is None or rss < best[0]:
            best = (rss, beta, float(tau))

    assert best is not None
    rss, beta, tau = best
    intercept, slope_young, delta = beta
    slope_old = slope_young + delta
    dist_at_break = intercept + slope_young * tau

    # Recompute residuals at the winning tau.
    x1 = ages
    x2 = np.maximum(ages - tau, 0.0)
    X = np.column_stack([np.ones_like(ages), x1, x2])
    pred = X @ beta
    residuals = dists - pred

    return BrokenStickFit(
        break_age_Ma=tau,
        dist_at_break_km=float(dist_at_break),
        slope_young=float(slope_young),
        slope_old=float(slope_old),
        rss=rss,
        residuals=residuals,
    )


def apparent_speed_cm_per_yr(ages_Ma: np.ndarray, dists_km: np.ndarray) -> np.ndarray:
    """Numerical derivative |d(distance)/d(age)|, converted from km/Myr to cm/yr.

    Uses centred differences on the interior points and one-sided differences
    at the endpoints. Returns an array of the same length as the inputs.
    """
    ages = np.asarray(ages_Ma, dtype=float)
    dists = np.asarray(dists_km, dtype=float)
    if ages.shape != dists.shape:
        raise ValueError("ages_Ma and dists_km must have the same shape")

    # np.gradient returns d(dist)/d(age) with the correct 1-sided/centred behaviour.
    slope_km_per_Myr = np.gradient(dists, ages)
    # 1 km/Myr = 100 000 cm / 1 000 000 yr = 0.1 cm/yr.
    return np.abs(slope_km_per_Myr) * 0.1

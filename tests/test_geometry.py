"""Tests for src.geometry — great-circle math + broken-stick age-distance fit."""
from __future__ import annotations

import numpy as np
import pytest

from src.geometry import (
    BEND_LAT,
    BEND_LON,
    KILAUEA_LAT,
    KILAUEA_LON,
    apparent_speed_cm_per_yr,
    chain_azimuth_deg,
    chain_distance_via_bend_km,
    fit_broken_stick,
    great_circle_distance_km,
)

EARTH_RADIUS_KM = 6371.0088


# ---------- great_circle_distance_km ----------------------------------------


def test_distance_to_self_is_zero():
    assert great_circle_distance_km(20.0, 200.0, 20.0, 200.0) == pytest.approx(0.0, abs=1e-9)


def test_distance_antipodal_is_half_circumference():
    d = great_circle_distance_km(0.0, 0.0, 0.0, 180.0)
    assert d == pytest.approx(np.pi * EARTH_RADIUS_KM, rel=1e-3)


def test_distance_vectorised_matches_scalar():
    lon1 = np.array([200.0, 190.0, 170.0])
    lat1 = np.array([20.0, 30.0, 45.0])
    out = great_circle_distance_km(KILAUEA_LAT, KILAUEA_LON, lat1, lon1)
    assert out.shape == lon1.shape
    expected = np.array(
        [great_circle_distance_km(KILAUEA_LAT, KILAUEA_LON, la, lo) for la, lo in zip(lat1, lon1)]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-9)


def test_kilauea_to_midway_distance_is_about_2400_km():
    # Midway Atoll ~28.20N, 182.63E (-177.37W). Published chain distance ~2400 km.
    d = great_circle_distance_km(KILAUEA_LAT, KILAUEA_LON, 28.20, 182.63)
    assert d == pytest.approx(2420.0, abs=80.0)


# ---------- chain_azimuth_deg ------------------------------------------------


def test_azimuth_due_north_is_zero():
    az = chain_azimuth_deg(0.0, 0.0, 10.0, 0.0)
    assert az == pytest.approx(0.0, abs=1e-6)


def test_azimuth_due_east_is_ninety():
    az = chain_azimuth_deg(0.0, 0.0, 0.0, 10.0)
    assert az == pytest.approx(90.0, abs=1e-6)


def test_azimuth_wraps_into_0_360():
    az = chain_azimuth_deg(0.0, 0.0, -10.0, 0.0)  # due south
    assert az == pytest.approx(180.0, abs=1e-6)
    az2 = chain_azimuth_deg(0.0, 0.0, 0.0, -10.0)  # due west
    assert az2 == pytest.approx(270.0, abs=1e-6)


def test_hawaiian_chain_points_WNW_from_kilauea():
    # From Kilauea toward Midway the chain trends roughly WNW → azimuth ~290–305°.
    az = chain_azimuth_deg(KILAUEA_LAT, KILAUEA_LON, 28.20, 182.63)
    assert 280.0 < az < 315.0


def test_emperor_chain_points_approximately_north():
    # Daikakuji (32N, ~172E) to Suiko (44.6N, ~170E): azimuth near 350–010°.
    az = chain_azimuth_deg(32.0, 172.0, 44.58, 170.33)
    assert az > 340.0 or az < 20.0


# ---------- chain_distance_via_bend_km --------------------------------------


def test_chain_distance_hawaiian_matches_direct_great_circle():
    # For post-bend seamounts the chain is a near-great-circle from Kilauea,
    # so chain distance should equal radial distance.
    lat, lon = 28.20, 182.63  # Midway
    direct = great_circle_distance_km(KILAUEA_LAT, KILAUEA_LON, lat, lon)
    via = chain_distance_via_bend_km(lat, lon, chain="Hawaiian")
    assert via == pytest.approx(direct, rel=1e-12)


def test_chain_distance_emperor_is_sum_of_two_legs():
    lat, lon = 53.40, 164.40  # Meiji
    leg1 = great_circle_distance_km(KILAUEA_LAT, KILAUEA_LON, BEND_LAT, BEND_LON)
    leg2 = great_circle_distance_km(BEND_LAT, BEND_LON, lat, lon)
    via = chain_distance_via_bend_km(lat, lon, chain="Emperor")
    assert via == pytest.approx(leg1 + leg2, rel=1e-12)
    # Sanity: chain distance for Meiji should be larger than the direct
    # great-circle distance from Kilauea, because the chain bends.
    assert via > great_circle_distance_km(KILAUEA_LAT, KILAUEA_LON, lat, lon)


def test_chain_distance_bend_zone_uses_direct_distance():
    # "Bend" seamounts sit right at the kink; treat them as the Hawaiian side.
    lat, lon = 32.08, 172.30
    direct = great_circle_distance_km(KILAUEA_LAT, KILAUEA_LON, lat, lon)
    via = chain_distance_via_bend_km(lat, lon, chain="Bend")
    assert via == pytest.approx(direct, rel=1e-12)


def test_chain_distance_is_vectorised():
    lats = np.array([28.20, 35.26, 53.40])
    lons = np.array([182.63, 171.59, 164.40])
    chains = np.array(["Hawaiian", "Bend", "Emperor"])
    out = chain_distance_via_bend_km(lats, lons, chain=chains)
    assert out.shape == lats.shape
    # Each element should match the scalar implementation.
    expected = np.array(
        [chain_distance_via_bend_km(la, lo, chain=c) for la, lo, c in zip(lats, lons, chains)]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)


# ---------- fit_broken_stick -------------------------------------------------


def test_broken_stick_recovers_planted_breakpoint():
    """Synthetic: two linear segments joined at t=47, with different slopes.
    The fit should recover the breakpoint within ~2 Myr."""
    rng = np.random.default_rng(0)
    ages_young = np.linspace(0, 47, 30)
    ages_old = np.linspace(47, 80, 30)
    dist_young = 95.0 * ages_young  # 9.5 cm/yr → 95 km/Myr
    dist_old = 95.0 * 47 + 80.0 * (ages_old - 47)  # slower in Emperor segment
    ages = np.concatenate([ages_young, ages_old])
    dists = np.concatenate([dist_young, dist_old]) + rng.normal(0, 20, size=ages.size)
    res = fit_broken_stick(ages, dists, break_bounds=(20.0, 70.0))
    assert res.break_age_Ma == pytest.approx(47.0, abs=2.5)
    # Slopes should come back roughly right (order-of-magnitude sanity).
    assert 80.0 < res.slope_young < 110.0
    assert 65.0 < res.slope_old < 95.0


def test_broken_stick_returns_residuals_with_same_length_as_input():
    ages = np.linspace(0, 80, 40)
    dists = 90.0 * ages + 10.0 * np.sin(ages)
    res = fit_broken_stick(ages, dists, break_bounds=(20.0, 70.0))
    assert res.residuals.shape == ages.shape


# ---------- apparent_speed_cm_per_yr ----------------------------------------


def test_apparent_speed_from_constant_slope_is_constant():
    # 95 km / Myr  ==  9.5 cm / yr.
    ages = np.linspace(1.0, 60.0, 30)
    dists = 95.0 * ages
    v = apparent_speed_cm_per_yr(ages, dists)
    assert v.shape == ages.shape
    # Centre of the array should sit very close to 9.5 cm/yr.
    np.testing.assert_allclose(v[5:-5], 9.5, rtol=1e-6)


def test_apparent_speed_handles_age_zero_start():
    ages = np.array([0.0, 1.0, 2.0, 3.0])
    dists = np.array([0.0, 95.0, 190.0, 285.0])
    v = apparent_speed_cm_per_yr(ages, dists)
    assert np.isfinite(v).all()
    np.testing.assert_allclose(v[1:-1], 9.5, rtol=1e-6)

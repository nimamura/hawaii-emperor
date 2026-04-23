"""Smoke tests for src.plotting.

These tests don't try to verify the look of the figure — that's reviewed by
eye in ``outputs/hawaii_emperor_bend.png``. What they pin down is that the
figure assembles without exception from a realistic in-memory catalogue and
exposes the four panels in the documented order so callers (and the
end-to-end script) can rely on it.
"""
from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless backend for CI / test runs.

import matplotlib.pyplot as plt  # noqa: E402

from src.plotting import plot_hawaii_emperor_figure  # noqa: E402


def _tiny_catalog() -> pd.DataFrame:
    """A 10-row spoof of data/seamount_ages.csv already in [0, 360) lon."""
    rows = [
        ("Kilauea",  "Hawaiian", 204.71, 19.42,  0.0,  0.1),
        ("Kauai",    "Hawaiian", 200.50, 22.06,  5.1,  0.2),
        ("Nihoa",    "Hawaiian", 198.07, 23.07,  7.2,  0.3),
        ("Necker",   "Hawaiian", 195.29, 23.58, 10.3, 0.3),
        ("Midway",   "Hawaiian", 182.63, 28.20, 27.7, 0.3),
        ("Daikakuji","Bend",     172.30, 32.08, 46.7, 0.3),
        ("Koko",     "Bend",     171.59, 35.26, 48.1, 0.3),
        ("Ojin",     "Emperor",  170.40, 37.97, 55.2, 0.2),
        ("Suiko",    "Emperor",  170.33, 44.58, 60.9, 0.2),
        ("Meiji",    "Emperor",  164.40, 53.40, 82.0, 0.5),
    ]
    return pd.DataFrame(
        rows,
        columns=["name", "chain", "lon", "lat", "age_Ma", "age_err_Ma"],
    )


def test_figure_builds_and_has_four_labelled_panels():
    df = _tiny_catalog()
    fig = plot_hawaii_emperor_figure(df)
    assert isinstance(fig, plt.Figure)
    # 2 × 2 grid → four axes, each tagged with a panel letter via set_label.
    labels = {ax.get_label() for ax in fig.axes if ax.get_label() in {"A", "B", "C", "D"}}
    assert labels == {"A", "B", "C", "D"}
    plt.close(fig)


def test_figure_has_white_background():
    df = _tiny_catalog()
    fig = plot_hawaii_emperor_figure(df)
    # X-post convention: solid white canvas.
    assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    plt.close(fig)


def test_figure_accepts_bend_age_override():
    df = _tiny_catalog()
    fig = plot_hawaii_emperor_figure(df, bend_age_Ma=47.0)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

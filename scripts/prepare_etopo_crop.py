#!/usr/bin/env python
"""Pre-crop and downsample ETOPO 2022 to the North Pacific bbox we plot.

The full ETOPO 2022 60-arcsec grid (~933 MB) is sitting in the
planetary-hypsometry project; this script reads it with xarray, crops
to the Hawaii–Emperor map domain, downsamples by factor 4 (so output is
~15-arcmin resolution, plenty for a slide-sized panel), re-expresses
longitude in [0, 360) so the dateline doesn't split the array, and
saves the result as a compact ``.npz`` bundled with the repository.

Run once; the generated ``data/etopo_north_pacific.npz`` is force-added
to git.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_NC = Path(
    "/Users/nimamura/work_raithing/planetary-hypsometry/data/earth/"
    "ETOPO_2022_v1_60s_N90W180_surface.nc"
)
OUT_PATH = REPO_ROOT / "data" / "etopo_north_pacific.npz"

# Map domain that Panel A renders.
LAT_MIN, LAT_MAX = 14.0, 61.0
LON_MIN_0_360, LON_MAX_0_360 = 140.0, 216.0  # 0..360 convention

DOWNSAMPLE = 4  # 60s × 4 = 4 arcmin effective resolution.


def main() -> None:
    if not SRC_NC.exists():
        raise FileNotFoundError(f"ETOPO NC file not found: {SRC_NC}")
    ds = xr.open_dataset(SRC_NC)

    # Native lon is in [-180, 180); shift to [0, 360) so the Pacific is contiguous.
    lon = ds.lon.to_numpy()
    lon360 = np.mod(lon, 360.0)
    order = np.argsort(lon360)
    lon360_sorted = lon360[order]
    z_reordered = ds.z.to_numpy()[:, order]
    lat = ds.lat.to_numpy()

    # Select the bbox.
    lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lon_mask = (lon360_sorted >= LON_MIN_0_360) & (lon360_sorted <= LON_MAX_0_360)

    lat_crop = lat[lat_mask]
    lon_crop = lon360_sorted[lon_mask]
    z_crop = z_reordered[np.ix_(lat_mask, lon_mask)]

    # Downsample.
    z_down = z_crop[::DOWNSAMPLE, ::DOWNSAMPLE].astype(np.float32)
    lat_down = lat_crop[::DOWNSAMPLE]
    lon_down = lon_crop[::DOWNSAMPLE]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, lat=lat_down, lon=lon_down, z=z_down)

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(
        f"saved {OUT_PATH.relative_to(REPO_ROOT)}  "
        f"shape={z_down.shape}  size={size_mb:.2f} MB  "
        f"z range [{float(z_down.min()):.0f}, {float(z_down.max()):.0f}] m"
    )


if __name__ == "__main__":
    main()

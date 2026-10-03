"""
Build an (N, T, F) satellite/climate time-series dataset from a set of monthly
parquet extracts that all share the same point grid.

Each input parquet file is one month of observations for a fixed set of
points (Sentinel-1/2 bands, NDVI, weather, terrain, land-cover, label).
This script aligns the months on `point_id`, stacks them into a single
tensor, fills short gaps via linear interpolation, drops points that are
too incomplete to trust, and writes the result as .npy arrays plus a JSON
manifest describing what happened.

Usage:
    python build_timeseries_dataset.py \
        --raw-dir data/raw \
        --output-dir data/processed_tumakuru \
        --prefix Tumakuru \
        --months 2025_06 2025_07 2025_08 2025_09 2025_10

Run with --help for all options.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "VV", "VH", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
    "B11", "B12", "NDVI", "temperature_2m", "total_precipitation",
    "elevation", "slope",
]
META_COLUMNS = ["point_id", "latitude", "longitude", "month", "dynamic_world", "label"]
KEEP_COLUMNS = FEATURE_COLUMNS + META_COLUMNS

NODATA_SENTINEL = -9999
DEFAULT_NAN_THRESHOLD = 0.1
DEFAULT_CHUNK_SIZE = 500_000

logger = logging.getLogger("build_timeseries_dataset")


@dataclass
class Config:
    raw_dir: Path
    output_dir: Path
    prefix: str
    months: list[str]
    nan_threshold: float = DEFAULT_NAN_THRESHOLD
    chunk_size: int = DEFAULT_CHUNK_SIZE
    file_suffix: str = "_formatted.parquet"

    @property
    def input_paths(self) -> list[Path]:
        return [self.raw_dir / f"{self.prefix}_{m}{self.file_suffix}" for m in self.months]

    @property
    def n_timesteps(self) -> int:
        return len(self.months)


@dataclass
class Stats:
    """Running record of what happened, written out as a manifest at the end."""
    per_month_rows: dict[str, int] = field(default_factory=dict)
    n_points_total: int = 0
    n_dropped_nan_threshold: int = 0
    n_dropped_post_interpolation: int = 0
    n_points_final: int = 0
    elapsed_seconds: float = 0.0


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------

def load_and_sort_month(path: Path, n_timesteps: int, stats: Stats) -> dict:
    """Load one month's parquet file, sanitize sentinel values, and sort by point_id."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    t0 = time.time()
    df = pd.read_parquet(path, columns=KEEP_COLUMNS)

    missing_cols = set(KEEP_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path.name} is missing expected columns: {sorted(missing_cols)}")

    if not df["point_id"].is_unique:
        n_dupes = int(df["point_id"].duplicated().sum())
        raise ValueError(
            f"{path.name} has {n_dupes} duplicate point_id values. "
            "Stacking assumes exactly one row per point per month — "
            "resolve duplicates upstream before running this script."
        )

    # -9999 is the nodata sentinel for these feature rasters. Sanity-check
    # that we're not accidentally nuking legitimate values (e.g. an NDVI or
    # temperature reading that happens to fall near -9999, which would be
    # a red flag rather than nodata).
    near_sentinel = ((df[FEATURE_COLUMNS] > NODATA_SENTINEL - 1)
                      & (df[FEATURE_COLUMNS] < NODATA_SENTINEL + 1)
                      & (df[FEATURE_COLUMNS] != NODATA_SENTINEL))
    if near_sentinel.to_numpy().any():
        logger.warning(
            "%s: found values very close to the %d sentinel but not equal to it — "
            "double check these aren't legitimate readings.", path.name, NODATA_SENTINEL,
        )
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(NODATA_SENTINEL, np.nan)

    df = df.sort_values("point_id", kind="mergesort").reset_index(drop=True)
    stats.per_month_rows[path.name] = len(df)

    logger.info("  %s: %d rows loaded + sorted (%.1fs)", path.name, len(df), time.time() - t0)
    return {
        "point_id": df["point_id"].to_numpy(),
        "features": df[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        "latitude": df["latitude"].to_numpy(dtype=np.float32),
        "longitude": df["longitude"].to_numpy(dtype=np.float32),
        "dynamic_world": df["dynamic_world"].to_numpy(dtype=np.float32),
        "month": df["month"].to_numpy(),
        "label": df["label"].to_numpy(dtype=np.float32),
    }


def verify_alignment(months_data: list[dict]) -> np.ndarray:
    """Confirm every month has the identical point_id ordering; return that ordering."""
    base_pid = months_data[0]["point_id"]
    for i, month in enumerate(months_data[1:], start=1):
        if not np.array_equal(base_pid, month["point_id"]):
            raise RuntimeError(
                f"Month index {i} has a different point_id set/order than month 0 "
                "after sorting. Files don't share an identical point grid — an "
                "explicit merge (e.g. pd.merge on point_id) is needed instead of "
                "direct stacking."
            )
    return base_pid


def interpolate_gaps(raw_all: np.ndarray, nan_mask_all: np.ndarray, keep_mask: np.ndarray,
                      chunk_size: int, t0: float) -> np.ndarray:
    """
    Linearly interpolate along the time axis for points that have some but not
    all timesteps missing, processed in chunks to bound peak memory. Any gap
    at the start/end of the series (which linear interpolation can't fill) is
    covered by limit_direction='both'.
    """
    n = raw_all.shape[0]
    n_features = raw_all.shape[2]
    n_timesteps = raw_all.shape[1]
    x_final = raw_all.copy()
    n_chunks = (n + chunk_size - 1) // chunk_size

    for chunk_i in range(n_chunks):
        s, e = chunk_i * chunk_size, min((chunk_i + 1) * chunk_size, n)
        chunk = raw_all[s:e]
        chunk_nanmask = nan_mask_all[s:e]
        needs_fix = chunk_nanmask.any(axis=(1, 2)) & keep_mask[s:e]

        if needs_fix.any():
            problem_pts = chunk[needs_fix]  # (n_problem, T, F)
            # Interpolate along the time axis without the deprecated
            # DataFrame.interpolate(axis=1) path: put time on axis 0 instead.
            flat = problem_pts.transpose(1, 0, 2).reshape(n_timesteps, -1)  # (T, n_problem*F)
            filled = (
                pd.DataFrame(flat)
                .interpolate(method="linear", axis=0, limit_direction="both")
                .to_numpy(dtype=np.float32)
            )
            fixed = filled.reshape(n_timesteps, -1, n_features).transpose(1, 0, 2)
            x_final[s:e][needs_fix] = fixed

        if chunk_i % 5 == 0 or chunk_i == n_chunks - 1:
            logger.info("  interpolation chunk %d/%d (%.1fs elapsed)",
                        chunk_i + 1, n_chunks, time.time() - t0)

    return x_final


def build_dataset(cfg: Config) -> Stats:
    t0 = time.time()
    stats = Stats()

    logger.info("Loading + sorting %d monthly files...", len(cfg.input_paths))
    months_data = [load_and_sort_month(p, cfg.n_timesteps, stats) for p in cfg.input_paths]

    logger.info("Verifying all months share an identical point grid...")
    point_ids = verify_alignment(months_data)
    n = len(point_ids)
    stats.n_points_total = n
    logger.info("All %d months aligned on %d points (%.1fs)", cfg.n_timesteps, n, time.time() - t0)

    raw_all = np.stack([m["features"] for m in months_data], axis=1)  # (N, T, F)
    logger.info("Stacked feature tensor: %s", raw_all.shape)

    nan_mask_all = np.isnan(raw_all)
    nan_fraction = nan_mask_all.mean(axis=(1, 2))
    keep_mask = nan_fraction <= cfg.nan_threshold
    stats.n_dropped_nan_threshold = int(n - keep_mask.sum())
    logger.info("Points within NaN threshold (<=%.0f%% missing): %d / %d",
                cfg.nan_threshold * 100, keep_mask.sum(), n)

    logger.info("Interpolating remaining gaps (chunked)...")
    x_final = interpolate_gaps(raw_all, nan_mask_all, keep_mask, cfg.chunk_size, t0)

    still_nan = np.isnan(x_final).any(axis=(1, 2))
    final_keep = keep_mask & ~still_nan
    stats.n_dropped_post_interpolation = int(keep_mask.sum() - final_keep.sum())
    stats.n_points_final = int(final_keep.sum())
    logger.info("Final points kept: %d / %d (%.1fs elapsed)",
                final_keep.sum(), n, time.time() - t0)

    x = x_final[final_keep]
    nan_masks = nan_mask_all[final_keep]

    kept_point_ids = point_ids[final_keep]
    lat = months_data[0]["latitude"][final_keep]
    lon = months_data[0]["longitude"][final_keep]
    latlons = np.stack([lat, lon], axis=1).astype(np.float32)

    dws = np.stack([m["dynamic_world"] for m in months_data], axis=1)[final_keep].astype(np.int64)
    months_arr = np.stack([m["month"] for m in months_data], axis=1)[final_keep].astype(np.int64)

    label_month0 = months_data[0]["label"][final_keep]
    y = np.where(pd.isna(label_month0), -1, label_month0).astype(np.int64)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_dir / "X.npy", x)
    np.save(cfg.output_dir / "y.npy", y)
    np.save(cfg.output_dir / "point_ids.npy", kept_point_ids)
    np.save(cfg.output_dir / "location_ids.npy", kept_point_ids.astype(str))
    np.save(cfg.output_dir / "latlons.npy", latlons)
    np.save(cfg.output_dir / "dws.npy", dws)
    np.save(cfg.output_dir / "months.npy", months_arr)
    np.save(cfg.output_dir / "nan_mask.npy", nan_masks)

    stats.elapsed_seconds = round(time.time() - t0, 1)
    manifest = {
        "config": {
            "raw_dir": str(cfg.raw_dir),
            "output_dir": str(cfg.output_dir),
            "prefix": cfg.prefix,
            "months": cfg.months,
            "nan_threshold": cfg.nan_threshold,
            "chunk_size": cfg.chunk_size,
            "feature_columns": FEATURE_COLUMNS,
        },
        "stats": stats.__dict__,
        "output_shapes": {
            "X": list(x.shape),
            "y": list(y.shape),
        },
    }
    with open(cfg.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("Final X shape: %s", x.shape)
    logger.info("Dropped (NaN threshold): %d | Dropped (post-interpolation): %d",
                stats.n_dropped_nan_threshold, stats.n_dropped_post_interpolation)
    logger.info("Total elapsed: %.1fs", stats.elapsed_seconds)
    logger.info("Saved -> %s", cfg.output_dir)

    return stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base_dir = Path(__file__).resolve().parent.parent
    p.add_argument("--raw-dir", type=Path, default=base_dir / "data" / "raw",
                    help="Directory containing the monthly parquet files.")
    p.add_argument("--output-dir", type=Path, default=base_dir / "data" / "processed_tumakuru",
                    help="Directory to write the .npy outputs and manifest.json.")
    p.add_argument("--prefix", type=str, default="Tumakuru",
                    help="Filename prefix, e.g. 'Tumakuru' for Tumakuru_2025_06_formatted.parquet.")
    p.add_argument("--months", nargs="+", default=["2025_06", "2025_07", "2025_08", "2025_09", "2025_10"],
                    help="Month tokens matching filenames, e.g. 2025_06 2025_07 ...")
    p.add_argument("--nan-threshold", type=float, default=DEFAULT_NAN_THRESHOLD,
                    help="Max fraction of missing values allowed per point before it's dropped.")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                    help="Number of points processed per interpolation chunk (memory control).")
    args = p.parse_args(argv)
    return Config(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        prefix=args.prefix,
        months=args.months,
        nan_threshold=args.nan_threshold,
        chunk_size=args.chunk_size,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    cfg = parse_args(argv)
    try:
        build_dataset(cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
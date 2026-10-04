"""
The following code is implemented for Mandya can be used for other districts as well.
MANDYA RICE - FILL MISSING VALUES (INTERPOLATION + MEDIAN FALLBACK)
------------------------------------------------------------------------------
Approach:
  1. PRIMARY: linear interpolation along the month axis, PER POINT_ID.
     This respects each point's own trajectory (its actual growth curve)
     instead of pulling every point toward one flat number.
  2. FALLBACK: for any point that still has NaN after interpolation (e.g. it
     had no valid data at all that year, or was missing at the very start/end
     with nothing to anchor to), fill with the MEDIAN of that column FOR THAT
     MONTH across all other points. Median (not mean) because every parameter
     showed at least moderate skew in the bell-curve check, and EVI in
     particular was extremely skewed due to formula outlier artifacts.

Order matters: interpolation always runs first and covers most gaps. The
monthly median only touches whatever interpolation could not fix.

EDIT BEFORE RUNNING:
  - CSV_PATHS: point these at your 5 downloaded CSVs.

"""
import pandas as pd
import numpy as np
import os

# =========================== CONFIG ==========================================
CSV_PATHS = {
    2019: "mandya_rice_points_2019_usednow.csv",   # <-- CHANGE to your actual file paths
    2020: "mandya_rice_points_2020_usednow.csv",
    2021: "mandya_rice_points_2021_usednow.csv",
    2022: "mandya_rice_points_2022_usednow.csv",
    2023: "mandya_rice_points_2023_usednow.csv",
}

OUTPUT_DIR = "mandya_rice_filled"
FILL_COLUMNS = ["NDVI", "LSWI", "EVI", "VV", "VH"]

CLIP_EVI = True
EVI_MIN, EVI_MAX = -1.0, 2.5   # standard valid EVI range; values outside this are formula artifacts

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fill_missing(df):
    df = df.copy()

    # Step 1: restore true NaN wherever GEE flagged the cell as filled
    for col in FILL_COLUMNS:
        if col in df.columns:
            df.loc[df["is_missing"] == 1, col] = np.nan

    # Step 2 (optional): clip EVI outlier artifacts before anything else touches it
    if CLIP_EVI and "EVI" in df.columns:
        df["EVI"] = df["EVI"].clip(lower=EVI_MIN, upper=EVI_MAX)

    df = df.sort_values(["point_id", "month"])

    # Step 3 (PRIMARY): linear interpolation along month, per point_id.
    # limit_direction='forward' only interpolates BETWEEN known points -
    # it will NOT extrapolate off the ends here, so leading/trailing gaps
    # for a point are deliberately left NaN and handled by the fallback below,
    # rather than being guessed by extending the nearest known value.
    for col in FILL_COLUMNS:
        df[col] = (
            df.groupby("point_id")[col]
            .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
        )
        # note: limit_direction='both' does allow edge-fill from the single
        # nearest known value when only one side has data - if you'd rather
        # leave pure edge gaps for the median fallback instead, change this
        # to limit_direction='forward' and interpolate with limit_area='inside'

    n_after_interp = {col: int(df[col].isna().sum()) for col in FILL_COLUMNS}

    # Step 4 (FALLBACK): whatever interpolation couldn't fix, fill with the
    # median of that column FOR THAT MONTH across all other points
    for col in FILL_COLUMNS:
        monthly_median = df.groupby("month")[col].transform("median")
        df[col] = df[col].fillna(monthly_median)

    # Final safety net: if an entire month has no valid values for a column
    # at all (monthly_median itself NaN), fall back to overall column median
    for col in FILL_COLUMNS:
        overall_median = df[col].median()
        df[col] = df[col].fillna(overall_median)

    return df, n_after_interp


# =========================== RUN FOR EACH YEAR ================================
for year, path in CSV_PATHS.items():
    print(f"\nProcessing {year} ({path}) ...")
    df = pd.read_csv(path)

    n_missing_before = int(df["is_missing"].sum()) if "is_missing" in df.columns else "N/A"
    print(f"  Rows flagged missing before fill: {n_missing_before} / {len(df)}")

    df_filled, n_after_interp = fill_missing(df)

    print(f"  Remaining gaps after interpolation (before median fallback): {n_after_interp}")

    out_path = os.path.join(OUTPUT_DIR, f"mandya_rice_points_{year}_filled.csv")
    df_filled.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

print("\nDone. Filled CSVs are in the 'mandya_rice_filled' folder.")



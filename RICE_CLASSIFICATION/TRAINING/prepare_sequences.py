import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_CSVS = [
    os.path.join(BASE_DIR, "data", "raw", "Karnataka_Presto_Training_2019_imputed.csv"),
    os.path.join(BASE_DIR, "data", "raw", "Karnataka_Presto_Training_2020_imputed.csv"),
    os.path.join(BASE_DIR, "data", "raw", "Karnataka_Presto_Training_2021_imputed.csv"),
    os.path.join(BASE_DIR, "data", "raw", "Karnataka_Presto_Training_2022_imputed.csv"),
    os.path.join(BASE_DIR, "data", "raw", "Karnataka_Presto_Training_2023_imputed.csv"),
]

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NDVI added to S2_COLS — passed to PRESTO as an S2 band
FEATURE_COLUMNS = [
    "VV", "VH",
    "B2", "B3", "B4",
    "B5", "B6", "B7",
    "B8", "B8A",
    "B11", "B12",
    "NDVI",                    # ← added here, grouped with S2
    "temperature_2m", "total_precipitation",
    "elevation", "slope",
]

EXPECTED_TIMESTEPS = 5
NAN_THRESHOLD      = 0.1

# ==========================================================
# Load and combine
# ==========================================================

print("=" * 60)
print("Loading datasets...")

dfs = []
for path in INPUT_CSVS:
    if not os.path.exists(path):
        print(f"  WARNING: File not found, skipping → {path}")
        continue
    temp        = pd.read_csv(path)
    source_name = os.path.basename(path).replace(".csv", "")
    print(f"\nChecking {source_name}")

    missing = temp[FEATURE_COLUMNS].replace(-9999, np.nan).isna().sum()
    print(missing.sort_values(ascending=False))

    temp["location_id"] = temp["point_id"].astype(str)
    temp["point_id"]    = source_name + "_" + temp["point_id"].astype(str)

    print(f"  {source_name}: {len(temp)} rows, {temp['point_id'].nunique()} unique point-years, "
          f"{temp['location_id'].nunique()} unique physical locations")
    dfs.append(temp)

if not dfs:
    raise RuntimeError("No CSV files loaded. Check INPUT_CSVS paths.")

df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined rows            : {len(df)}")
print(f"Combined point-years     : {df['point_id'].nunique()}")
print(f"Combined unique locations: {df['location_id'].nunique()}")

# ==========================================================
# Replace sentinel → NaN
# ==========================================================

df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(-9999, np.nan)
print("\nMissing values per feature (raw):")
print(df[FEATURE_COLUMNS].isna().sum().sort_values(ascending=False))

# ==========================================================
# Sort chronologically
# ==========================================================

df = df.sort_values(["point_id", "year", "month"]).reset_index(drop=True)

# ==========================================================
# Filter to points with exactly EXPECTED_TIMESTEPS rows
# ==========================================================

counts       = df.groupby("point_id").size()
valid_points = counts[counts == EXPECTED_TIMESTEPS].index
dropped      = counts[counts != EXPECTED_TIMESTEPS].index

print("=" * 60)
print(f"Valid point-years   : {len(valid_points)}")
print(f"Dropped point-years : {len(dropped)} (wrong timestep count)")

df = df[df["point_id"].isin(valid_points)].copy()

# ==========================================================
# Build sequences
# ==========================================================

X            = []
y            = []
point_ids    = []
location_ids = []
latlons      = []
dws          = []
months       = []
nan_masks    = []

skipped_nan            = 0
skipped_label_conflict = 0

print("\nBuilding sequences...")

for point_id, group in df.groupby("point_id"):

    group = group.sort_values(["year", "month"]).reset_index(drop=True)

    # ── Label consistency ────────────────────────────────────
    unique_labels = group["label"].unique()
    if len(unique_labels) > 1:
        skipped_label_conflict += 1
        continue

    label = int(unique_labels[0])

    # ── Raw sequence (may contain NaN) ──────────────────────
    raw_sequence = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)  # (T, F)

    # ── NaN mask BEFORE interpolation ───────────────────────
    nan_mask_seq = np.isnan(raw_sequence)  # (T, F)

    # ── NaN threshold check ──────────────────────────────────
    nan_fraction = np.isnan(raw_sequence).mean()
    if nan_fraction > NAN_THRESHOLD:
        skipped_nan += 1
        continue

    # ── Interpolate remaining NaNs ───────────────────────────
    if nan_fraction > 0:
        seq_df = pd.DataFrame(raw_sequence, columns=FEATURE_COLUMNS)
        seq_df = seq_df.interpolate(method="linear", axis=0)
        seq_df = seq_df.bfill().ffill()
        sequence = seq_df.to_numpy(dtype=np.float32)
    else:
        sequence = raw_sequence

    if np.isnan(sequence).any():
        skipped_nan += 1
        continue

    # ── Shape guard ──────────────────────────────────────────
    if sequence.shape != (EXPECTED_TIMESTEPS, len(FEATURE_COLUMNS)):
        continue

    sequence_norm = sequence.astype(np.float32)

    # ── Collect ──────────────────────────────────────────────
    X.append(sequence_norm)
    y.append(label)
    point_ids.append(point_id)
    location_ids.append(group["location_id"].iloc[0])
    latlons.append([group["latitude"].iloc[0], group["longitude"].iloc[0]])
    dws.append(group["dynamic_world"].tolist())   # ← per-timestep DW, not just iloc[0]
    months.append(group["month"].tolist())
    nan_masks.append(nan_mask_seq)

    assert len(X) == len(y) == len(months) == len(latlons) == len(dws) == len(point_ids) == len(location_ids) == len(nan_masks)

# ==========================================================
# Convert to arrays
# ==========================================================

X            = np.array(X,            dtype=np.float32)   # (N, T, F)
y            = np.array(y,            dtype=np.int64)
point_ids    = np.array(point_ids)
location_ids = np.array(location_ids)
latlons      = np.array(latlons,      dtype=np.float32)
dws          = np.array(dws,          dtype=np.int64)      # (N, T) ← now per-timestep
months       = np.array(months,       dtype=np.int64)      # (N, T)
nan_masks    = np.array(nan_masks,    dtype=bool)          # (N, T, F)

# ==========================================================
# Shuffle
# ==========================================================

rng  = np.random.default_rng(seed=42)
perm = rng.permutation(len(X))

X            = X           [perm]
y            = y           [perm]
point_ids    = point_ids   [perm]
location_ids = location_ids[perm]
latlons      = latlons     [perm]
dws          = dws         [perm]
months       = months      [perm]
nan_masks    = nan_masks   [perm]

# ==========================================================
# Sanity check
# ==========================================================

print("=" * 60)
print("Raw feature ranges:")
for i, name in enumerate(FEATURE_COLUMNS):
    fmin = X[:, :, i].min()
    fmax = X[:, :, i].max()
    print(f"  {name:<22}: [{fmin:9.3f}, {fmax:9.3f}]")

# ==========================================================
# Summary
# ==========================================================

print("=" * 60)
print(f"Skipped — label conflict : {skipped_label_conflict}")
print(f"Skipped — too many NaNs  : {skipped_nan}")
print("=" * 60)
print(f"X shape       : {X.shape}  (N, T, F) — F={len(FEATURE_COLUMNS)} including NDVI")
print(f"y shape       : {y.shape}")
print(f"dws shape     : {dws.shape}  (N, T) — per-timestep dynamic world")
print(f"months shape  : {months.shape}")
print(f"Rice          : {(y == 1).sum()}")
print(f"Non-rice      : {(y == 0).sum()}")

# ==========================================================
# Save
# ==========================================================

np.save(os.path.join(OUTPUT_DIR, "X.npy"),            X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"),            y)
np.save(os.path.join(OUTPUT_DIR, "point_ids.npy"),    point_ids)
np.save(os.path.join(OUTPUT_DIR, "location_ids.npy"), location_ids)
np.save(os.path.join(OUTPUT_DIR, "latlons.npy"),      latlons)
np.save(os.path.join(OUTPUT_DIR, "dws.npy"),          dws)
np.save(os.path.join(OUTPUT_DIR, "months.npy"),       months)
np.save(os.path.join(OUTPUT_DIR, "nan_mask.npy"),     nan_masks)

print("=" * 60)
print(f"All files saved → {OUTPUT_DIR}")
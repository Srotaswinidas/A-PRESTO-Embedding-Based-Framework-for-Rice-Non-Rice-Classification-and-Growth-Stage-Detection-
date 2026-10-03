"""
generate_embeddings_tumakuru.py

Tumakuru-specific version of generate_embeddings.py.

Only real change vs the reference script: DATA_DIR now points at
data/processed_tumakuru (the output_dir produced by
build_timeseries_dataset.py --prefix Tumakuru --months 2025_06 2025_07
2025_08 2025_09 2025_10), instead of the generic data/processed.

Everything else — checkpoint path, NDVI-excluded-from-PRESTO handling,
per-timestep dynamic_world, and the "keep all raw monthly NDVI values"
concat step — is carried over unchanged from generate_embeddings.py,
since Tumakuru's X.npy has the same FEATURE_COLUMNS order and the same
5 timesteps (T=5), so the final embeddings.npy is (N, 128 + 5) = (N, 133),
same as before.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CHANGED: Tumakuru's processed arrays live in their own output dir
# (this is the --output-dir used when running build_timeseries_dataset.py
# with --prefix Tumakuru). Point everything at that instead of the
# generic data/processed used by the reference script.
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_tumakuru")
MODELS_DIR = os.path.join(BASE_DIR, "models")

PRESTO_REPO_DIR = os.path.join(os.path.dirname(BASE_DIR), "presto-worldcereal")
if not os.path.exists(PRESTO_REPO_DIR):
    PRESTO_REPO_DIR = os.path.join(BASE_DIR, "presto-worldcereal")

print("PRESTO_REPO_DIR:", PRESTO_REPO_DIR)
print("Exists         :", os.path.exists(PRESTO_REPO_DIR))
print("DATA_DIR       :", DATA_DIR)

if PRESTO_REPO_DIR not in sys.path:
    sys.path.insert(0, PRESTO_REPO_DIR)

from presto.dataops import DynamicWorld2020_2021
from presto.presto import Presto
from presto.utils import construct_single_presto_input

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8192

print(f"Device: {DEVICE}")

# ── Band columns as stored in X.npy — must match build_timeseries_dataset.py's
# FEATURE_COLUMNS exactly. This list still includes NDVI, because that's
# how X.npy is laid out on disk; we just don't hand NDVI to PRESTO below. ──
S1_COLS   = ["VV", "VH"]
S2_COLS   = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
ERA5_COLS = ["temperature_2m", "total_precipitation"]
SRTM_COLS = ["elevation", "slope"]
FEATURE_COLUMNS = S1_COLS + S2_COLS + ERA5_COLS + SRTM_COLS

# The subset of S2_COLS that PRESTO actually recognizes as valid band names.
# NDVI is deliberately excluded here — it gets handled separately below.
S2_COLS_FOR_PRESTO = [c for c in S2_COLS if c != "NDVI"]

# ---------------------------------------------------------------------------
# 1. Load processed arrays (Tumakuru: 5 months, so T=5)
# ---------------------------------------------------------------------------
print("\nLoading processed data...")
X            = np.load(os.path.join(DATA_DIR, "X.npy"))
y            = np.load(os.path.join(DATA_DIR, "y.npy"))
point_ids    = np.load(os.path.join(DATA_DIR, "point_ids.npy"),    allow_pickle=True)
location_ids = np.load(os.path.join(DATA_DIR, "location_ids.npy"), allow_pickle=True)
latlons      = np.load(os.path.join(DATA_DIR, "latlons.npy"))
dws          = np.load(os.path.join(DATA_DIR, "dws.npy"))      # (N, T) per-timestep
months       = np.load(os.path.join(DATA_DIR, "months.npy"))   # (N, T) 1-indexed

N, T, F = X.shape
print(f"X shape : {X.shape}  (F should be {len(FEATURE_COLUMNS)}, T should be 5 for Tumakuru)")
print(f"dws shape: {dws.shape}")

assert F == len(FEATURE_COLUMNS), (
    f"X.npy has {F} features but FEATURE_COLUMNS has {len(FEATURE_COLUMNS)}. "
    f"Rerun build_timeseries_dataset.py first."
)

# ---------------------------------------------------------------------------
# 2. Unit sanity check
# ---------------------------------------------------------------------------
col_idx = {name: i for i, name in enumerate(FEATURE_COLUMNS)}

s2_max    = X[:, :, [col_idx[b] for b in ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]]].max()
temp_mean = X[:, :, col_idx["temperature_2m"]].mean()
prec_max  = X[:, :, col_idx["total_precipitation"]].max()
ndvi_min  = X[:, :, col_idx["NDVI"]].min()
ndvi_max  = X[:, :, col_idx["NDVI"]].max()

print(f"\nUnit check:")
print(f"  S2 max   : {s2_max:.1f}    (expect 5000-16000)")
print(f"  Temp mean: {temp_mean:.2f}K (expect ~295-310K)")
print(f"  Precip   : {prec_max:.4f}m (expect <1.0m)")
print(f"  NDVI     : [{ndvi_min:.3f}, {ndvi_max:.3f}] (expect [-1, 1])")

if s2_max < 10:
    raise ValueError("S2 values look pre-normalized. Must be raw 0-10000 scale.")
if temp_mean < 200:
    raise ValueError("Temperature looks like Celsius, not Kelvin.")
print("Units look correct \u2713")

# ---------------------------------------------------------------------------
# 2b. Load PRESTO model early so we can sanitize dynamic_world against its
#     actual embedding table size before building any tensors.
#
#     WHY THIS IS NEEDED: build_timeseries_dataset.py only replaces the
#     NODATA_SENTINEL (-9999) inside FEATURE_COLUMNS. "dynamic_world" is a
#     META_COLUMN, so any point with a missing/nodata DW reading keeps its
#     raw sentinel value (-9999, or a raw DW nodata code) straight through
#     to dws.npy. Casting that to .long() and feeding it into dw_embed
#     (an nn.Embedding with a fixed number of rows) throws:
#         IndexError: index out of range in self
#     because -9999 (or 255, etc.) is nowhere near a valid row index.
# ---------------------------------------------------------------------------
print("\nLoading PRESTO model...")

finetuned_checkpoint = os.path.join(MODELS_DIR, "presto_finetuned.pt")

if os.path.exists(finetuned_checkpoint):
    print("Using fine-tuned checkpoint:", finetuned_checkpoint)
    presto_model = Presto.load_pretrained(finetuned_checkpoint, strict=False)
else:
    print(f"WARNING: fine-tuned checkpoint not found at {finetuned_checkpoint}")
    print("Falling back to default pretrained weights — this is probably NOT what you want.")
    presto_model = Presto.load_pretrained()

presto_model = presto_model.to(DEVICE).eval()
print("Model loaded \u2713")

# Ask the model itself how many DW classes it has room for, rather than
# hardcoding a number — this stays correct regardless of which checkpoint
# is loaded. The last row of a PRESTO dw_embed table is conventionally the
# "unknown / no data" class, so that's what we remap bad values to.
dw_num_embeddings = presto_model.encoder.dw_embed.num_embeddings
dw_unknown_idx = dw_num_embeddings - 1

invalid_dw = (dws < 0) | (dws >= dw_num_embeddings)
n_invalid_dw = int(invalid_dw.sum())
if n_invalid_dw > 0:
    n_points_affected = int(invalid_dw.any(axis=1).sum())
    print(
        f"\nWARNING: found {n_invalid_dw} invalid dynamic_world entries "
        f"(out of range [0, {dw_num_embeddings})) across {n_points_affected} points — "
        f"e.g. leftover nodata sentinel values. Remapping them to the "
        f"'unknown' class (index {dw_unknown_idx}) so the embedding lookup "
        f"doesn't crash."
    )
    dws = np.where(invalid_dw, dw_unknown_idx, dws)
else:
    print(f"\ndynamic_world values all within valid range [0, {dw_num_embeddings}) \u2713")

# ---------------------------------------------------------------------------
# 3. Slice each source
# ---------------------------------------------------------------------------
def slice_cols(cols):
    idx = [col_idx[c] for c in cols]
    return X[:, :, idx]

s1   = slice_cols(S1_COLS)             # (N, T, 2)
s2   = slice_cols(S2_COLS_FOR_PRESTO)  # (N, T, 10) — NDVI excluded, PRESTO doesn't know it
era5 = slice_cols(ERA5_COLS)           # (N, T, 2)
srtm = slice_cols(SRTM_COLS)           # (N, T, 2)

# NDVI handled separately: keep ALL raw monthly values instead of
# collapsing to a derived summary, so no phenological detail is discarded.
ndvi_seq = slice_cols(["NDVI"])[:, :, 0]        # (N, T) — one column per month, in month order

print(f"\nNDVI raw monthly values — shape {ndvi_seq.shape} "
      f"(kept as-is, one column per timestep, no summarization):")
for t in range(T):
    print(f"  timestep {t} (month col {t}) — min: {ndvi_seq[:, t].min():.4f}, "
          f"max: {ndvi_seq[:, t].max():.4f}, mean: {ndvi_seq[:, t].mean():.4f}")

months0 = months - 1  # 0-indexed for PRESTO

# ---------------------------------------------------------------------------
# 4. Build PRESTO input tensor + mask (vectorized)
# ---------------------------------------------------------------------------
print("\nBuilding PRESTO input tensors...")

x_flat, mask_flat, _ = construct_single_presto_input(
    s1=torch.from_numpy(s1.reshape(N * T, -1)).float(),
    s1_bands=S1_COLS,
    s2=torch.from_numpy(s2.reshape(N * T, -1)).float(),
    s2_bands=S2_COLS_FOR_PRESTO,
    era5=torch.from_numpy(era5.reshape(N * T, -1)).float(),
    era5_bands=ERA5_COLS,
    srtm=torch.from_numpy(srtm.reshape(N * T, -1)).float(),
    srtm_bands=SRTM_COLS,
    dynamic_world=torch.from_numpy(dws.reshape(N * T)).long(),  # per-timestep DW
    normalize=True,
)

num_bands = x_flat.shape[-1]

x_tensor    = torch.from_numpy(x_flat.reshape(N, T, num_bands).cpu().numpy()).float()
mask_tensor = torch.from_numpy(mask_flat.reshape(N, T, num_bands).cpu().numpy()).bool()

# DW tensor — (N, T) per-timestep
dw_tensor      = torch.from_numpy(dws).long()          # (N, T)
months_tensor  = torch.from_numpy(months0).long()      # (N, T)
latlons_tensor = torch.from_numpy(latlons).float()     # (N, 2)

print(f"x_tensor shape   : {x_tensor.shape}")
print(f"mask_tensor shape: {mask_tensor.shape}")
print(f"dw_tensor shape  : {dw_tensor.shape}")

# ---------------------------------------------------------------------------
# 5. (model already loaded in step 2b, above, so dws could be sanitized
#     against its real embedding-table size before tensors were built)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6. Generate embeddings in batches
# ---------------------------------------------------------------------------
print(f"\nGenerating embeddings (batch_size={BATCH_SIZE})...")

dataset = TensorDataset(x_tensor, dw_tensor, latlons_tensor, months_tensor, mask_tensor)
dl      = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

embed_dim   = presto_model.encoder.embedding_size          # 128
presto_embeddings = np.empty((N, embed_dim), dtype=np.float32)

with torch.no_grad():
    start = 0
    for batch_num, (x_b, dw_b, ll_b, m_b, mask_b) in enumerate(dl):
        x_b    = x_b.to(DEVICE)
        dw_b   = dw_b.to(DEVICE)
        ll_b   = ll_b.to(DEVICE)
        m_b    = m_b.to(DEVICE)
        mask_b = mask_b.to(DEVICE)

        out = presto_model.encoder(
            x_b,
            dynamic_world=dw_b.long(),
            mask=mask_b,
            latlons=ll_b,
            month=m_b,
        )

        n = out.shape[0]
        presto_embeddings[start:start + n] = out.cpu().numpy()
        start += n

        if batch_num % 5 == 0:
            print(f"  {start}/{N} processed...")

print(f"Done.")

# ---------------------------------------------------------------------------
# 6b. Concatenate all raw monthly NDVI values onto the PRESTO embedding
#     -> final embeddings.npy is (N, 128 + T) = (N, 133) for Tumakuru's
#     5 timesteps: 128 PRESTO dims + 5 raw NDVI dims (one per month, in
#     month order).
# ---------------------------------------------------------------------------
embeddings = np.concatenate(
    [presto_embeddings, ndvi_seq.astype(np.float32)],
    axis=1,
)
print(f"\nFinal embeddings shape (PRESTO + raw NDVI[{T} months]): {embeddings.shape}  "
      f"(expected ({N}, {embed_dim + T}))")

if embed_dim + T != 133:
    print(
        f"WARNING: expected 133 total dims (128 + 5 months) for Tumakuru, "
        f"got {embed_dim + T}. Check that Tumakuru's months.npy really has T=5 "
        f"timesteps and that the PRESTO checkpoint's embedding_size is 128."
    )

# ---------------------------------------------------------------------------
# 7. Verify
# ---------------------------------------------------------------------------
print(f"\nEmbedding check:")
print(f"  Shape    : {embeddings.shape}")
print(f"  All zeros: {(embeddings == 0).all()}")
print(f"  Mean     : {embeddings.mean():.4f}")
print(f"  Std      : {embeddings.std():.4f}")
for t in range(T):
    col = embed_dim + t
    print(f"  NDVI month-col {t} (index {col}) — min: {embeddings[:, col].min():.4f}, "
          f"max: {embeddings[:, col].max():.4f}")

if (embeddings == 0).all():
    raise ValueError("All embeddings are zero — PRESTO encoder did not run.")

# ---------------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------------
np.save(os.path.join(DATA_DIR, "embeddings.npy"),   embeddings)
np.save(os.path.join(DATA_DIR, "y.npy"),            y)
np.save(os.path.join(DATA_DIR, "point_ids.npy"),    point_ids)
np.save(os.path.join(DATA_DIR, "location_ids.npy"), location_ids)
np.save(os.path.join(DATA_DIR, "latlons.npy"),      latlons)

print(f"\nAll files saved \u2192 {DATA_DIR}")
print(f"embeddings.npy is now (N, {embed_dim + T}) — 128 PRESTO dims + {T} raw NDVI dims (one per month).")
print("Next step: RE-RUN train_mlp.py to retrain on the new embedding shape")
print(f"(the old mlp_rice_classifier.pt was trained on a different input_dim and is now stale).")
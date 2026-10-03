"""
generate_embeddings.py

Changes vs previous version:
  - FIXED: checkpoint path. Previous versions looked for
    data/processed/finetuned_model.pt, which does not exist and was
    silently falling back to the generic pretrained PRESTO weights every
    time. The real fine-tuned checkpoint is models/presto_finetuned.pt.
  - dws is now (N, T) per-timestep instead of a scalar repeated.
  - FIXED: NDVI was being listed in S2_COLS and passed straight into
    construct_single_presto_input(), but PRESTO's S2 band vocabulary is
    fixed to ['B1'..'B12'] — it does NOT know "NDVI" as a band name, so it
    was being silently dropped every time embeddings were generated. This
    was caught via SHAP analysis: NDVI's contribution came out as exactly
    0.0 for every sample, which led to tracing it back here.
    FIX: NDVI is no longer passed into PRESTO at all. Instead, it is
    concatenated onto the 128-dim PRESTO embedding AFTER encoding, giving
    the MLP direct access to NDVI.
  - UPDATED (this version): instead of collapsing NDVI down to 3 derived
    stats (early/peak/late mean), we now keep ALL 5 raw monthly NDVI
    values as-is. The early/peak/late summary was a lossy compression of
    the same 5 numbers — this keeps everything, including month-to-month
    detail the summary discarded, at the cost of 2 extra embedding
    dimensions (negligible next to the 128 PRESTO dims).
    Final embeddings.npy is now (N, 133): 128 PRESTO dims + 5 raw NDVI
    dims (one per month, in month order).
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

PRESTO_REPO_DIR = os.path.join(os.path.dirname(BASE_DIR), "presto-worldcereal")
if not os.path.exists(PRESTO_REPO_DIR):
    PRESTO_REPO_DIR = os.path.join(BASE_DIR, "presto-worldcereal")

print("PRESTO_REPO_DIR:", PRESTO_REPO_DIR)
print("Exists         :", os.path.exists(PRESTO_REPO_DIR))

if PRESTO_REPO_DIR not in sys.path:
    sys.path.insert(0, PRESTO_REPO_DIR)

from presto.dataops import DynamicWorld2020_2021
from presto.presto import Presto
from presto.utils import construct_single_presto_input

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8192

print(f"Device: {DEVICE}")

# ── Band columns as stored in X.npy — must match prepare_sequences.py
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
# 1. Load processed arrays
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
print(f"X shape : {X.shape}  (F should be {len(FEATURE_COLUMNS)})")
print(f"dws shape: {dws.shape}")

assert F == len(FEATURE_COLUMNS), (
    f"X.npy has {F} features but FEATURE_COLUMNS has {len(FEATURE_COLUMNS)}. "
    f"Rerun prepare_sequences.py first."
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
# 5. Load YOUR fine-tuned PRESTO model (models/presto_finetuned.pt)
# ---------------------------------------------------------------------------
print("\nLoading PRESTO model...")

# FIXED: this used to point at data_dir/finetuned_model.pt, which never
# existed. Your real fine-tuned checkpoint is here:
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
#     -> final embeddings.npy is (N, 128 + T): 128 PRESTO dims + T raw NDVI
#     dims (one per month, in month order). This replaces the previous
#     early/peak/late 3-stat summary with the full, uncompressed sequence.
# ---------------------------------------------------------------------------
embeddings = np.concatenate(
    [presto_embeddings, ndvi_seq.astype(np.float32)],
    axis=1,
)
print(f"\nFinal embeddings shape (PRESTO + raw NDVI[{T} months]): {embeddings.shape}  "
      f"(expected ({N}, {embed_dim + T}))")

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
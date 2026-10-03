"""
test_mlp_tumakuru.py

Runs the ALREADY-TRAINED MLP rice classifier (mlp_rice_classifier.pt +
mlp_scaler.joblib, produced by your train_mlp.py) on Tumakuru's embeddings
and produces a rice vs non-rice plot.

This is inference only — nothing here is fit/trained on Tumakuru data:
  - RiceMLP is rebuilt using the input_dim/hidden_dims/dropout stored
    INSIDE the checkpoint (not hardcoded), then its state_dict is loaded.
  - The StandardScaler from training is loaded and only .transform()'d on
    Tumakuru's embeddings — never re-fit. Re-fitting a scaler on new data
    would silently shift the feature distribution the model was actually
    trained on, making its learned weights meaningless.
  - At Tumakuru's scale (~1 crore / 10M points, ~5GB embeddings.npy),
    embeddings.npy is memory-mapped and processed in small batches — the
    full array (and a second full-size scaled copy) is never held in RAM
    at once, and the rice/non-rice map uses hexbin (aggregated bins)
    instead of a per-point scatter.
  - If Tumakuru's y.npy contains real labels (not all -1, per the label
    fallback in build_timeseries_dataset.py), the usual metrics/confusion
    matrix are computed on that labeled subset. If it's all -1 (no ground
    truth), metrics are skipped and only predictions + the plot are produced.

NOTE: an earlier version of this script added an Otsu-recalibrated
threshold on top of the fixed 0.5 cutoff. That step was removed after
inspecting the actual probability histogram on Tumakuru: the distribution
is a single sharp peak near 0 with a long decaying tail (not two separated
groups), so Otsu was just finding an inflection point in one continuous
curve and calling it a "threshold" — it moved the rice count from ~41K to
~800K points (a ~20x jump) with no evidence that was more correct. The
fixed 0.5 threshold from training is used throughout instead. The
geographic plausibility mask (elevation/slope/land-cover) is unaffected by
this and is applied directly to the 0.5-threshold predictions.

Usage:
    python test_mlp_tumakuru.py
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — never blocks waiting on a GUI window
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score,
)

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data", "processed_mandya")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
OUTPUT_DIR    = MODEL_DIR  # plots/predictions land alongside the model, like train_mlp.py does

MODEL_PATH  = os.path.join(MODEL_DIR, "mlp_rice_classifier.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "mlp_scaler.joblib")

# Must match build_timeseries_dataset.py's FEATURE_COLUMNS order exactly —
# this is how X.npy's last axis is laid out. Only used here to locate the
# elevation/slope columns for the geographic plausibility mask.
FEATURE_COLUMNS = [
    "VV", "VH", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
    "B11", "B12", "NDVI", "temperature_2m", "total_precipitation",
    "elevation", "slope",
]


# -----------------------------
# 1. Same MLP architecture as train_mlp.py (must match exactly for
#    load_state_dict to work)
# -----------------------------
class RiceMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # binary output, no sigmoid here
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_trained_model(device):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run train_mlp.py first."
        )
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = RiceMLP(
        input_dim=ckpt["input_dim"],
        hidden_dims=ckpt["hidden_dims"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {MODEL_PATH}  (input_dim={ckpt['input_dim']}, "
          f"hidden_dims={ckpt['hidden_dims']}, dropout={ckpt['dropout']})")
    return model, ckpt["input_dim"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -----------------------------
    # 2. Load Mandya data
    # -----------------------------
    print("\nLMandya embeddings...")
    # mmap_mode='r' means the ~5GB embeddings array (10M rows x 133 dims at
    # float32) is NOT pulled fully into RAM here — pages are read from disk
    # lazily as they're sliced below. y/location_ids/latlons are small
    # (tens-hundreds of MB at 10M rows) and safe to load fully.
    embeddings   = np.load(os.path.join(DATA_DIR, "embeddings.npy"), mmap_mode="r")
    y            = np.load(os.path.join(DATA_DIR, "y.npy"))
    location_ids = np.load(os.path.join(DATA_DIR, "location_ids.npy"), allow_pickle=True)
    latlons      = np.load(os.path.join(DATA_DIR, "latlons.npy"))  # (N, 2) -> [lat, lon]

    print(f"embeddings shape: {embeddings.shape}")
    print(f"y shape         : {y.shape}  (unique values: {np.unique(y)})")

    # -----------------------------
    # 3. Load model + scaler, sanity-check dims match
    # -----------------------------
    model, expected_dim = load_trained_model(device)
    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Tumakuru embeddings have {embeddings.shape[1]} dims but the trained "
            f"model expects {expected_dim}. Did the NDVI-months count or PRESTO "
            f"checkpoint change between training and this generation run?"
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run train_mlp.py first.")
    scaler = joblib.load(SCALER_PATH)
    # NOTE: scaler.transform() is now called per-chunk inside the inference
    # loop below, not on the whole array here — at ~10M rows, transforming
    # everything up front would materialize a second ~5GB array on top of
    # the mmap'd embeddings, which is what likely caused the hang.

    # -----------------------------
    # 4. Run inference — BATCHED, not all N points in one forward pass.
    #    Pushing the whole dataset through at once is what was likely
    #    hanging your system: BatchNorm1d + a large N creates big
    #    intermediate tensors, and once you start swapping to disk the
    #    machine looks frozen even though nothing crashed. Batching keeps
    #    peak memory bounded and prints progress so it's clear it's working.
    # -----------------------------
    N = embeddings.shape[0]
    INFER_BATCH_SIZE = 8192
    print(f"\nRunning inference on {N} points in batches of {INFER_BATCH_SIZE}...")

    probs = np.empty(N, dtype=np.float32)  # 10M floats ≈ 40MB — trivial, kept fully in memory
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N, INFER_BATCH_SIZE):
            end = min(start + INFER_BATCH_SIZE, N)
            # Reads only this chunk off disk from the mmap'd array, scales
            # just this chunk (small, temporary array), then discards it —
            # never holds the full embeddings or a full scaled copy in RAM.
            chunk = np.asarray(embeddings[start:end])  # pulls chunk into RAM
            chunk_scaled = scaler.transform(chunk)
            batch = torch.tensor(chunk_scaled, dtype=torch.float32).to(device)
            logits = model(batch)
            probs[start:end] = torch.sigmoid(logits).cpu().numpy()
            if (start // INFER_BATCH_SIZE) % 50 == 0:
                print(f"  {end}/{N} processed ({time.time() - t0:.1f}s elapsed)")
    print(f"Inference done in {time.time() - t0:.1f}s")

    preds = (probs > 0.5).astype(int)

    n_rice, n_non_rice = int(preds.sum()), int((preds == 0).sum())
    print(f"Predicted: {n_rice} rice points, {n_non_rice} non-rice points "
          f"({n_rice / len(preds):.1%} rice)")

    # Confidence = how far the model's probability sits from the 0.5 decision
    # boundary, for each class. For rice predictions that's just the raw
    # probability (closer to 1.0 = more confident); for non-rice predictions
    # it's (1 - probability), since a low probability is "confident non-rice".
    rice_confidences     = probs[preds == 1]
    non_rice_confidences = 1 - probs[preds == 0]

    print("\nOverall prediction confidence:")
    if n_rice > 0:
        print(f"  Rice     — mean confidence: {rice_confidences.mean():.4f}  "
              f"(min: {rice_confidences.min():.4f}, max: {rice_confidences.max():.4f})")
    else:
        print("  Rice     — no points predicted rice, nothing to average.")
    if n_non_rice > 0:
        print(f"  Non-rice — mean confidence: {non_rice_confidences.mean():.4f}  "
              f"(min: {non_rice_confidences.min():.4f}, max: {non_rice_confidences.max():.4f})")
    else:
        print("  Non-rice — no points predicted non-rice, nothing to average.")
    print(f"  Overall  — mean confidence across all points: "
          f"{np.concatenate([rice_confidences, non_rice_confidences]).mean():.4f}")

    # ---------------------------------------------------------------------
    # 4b. Geographic plausibility mask. Doesn't need Tumakuru ground-truth
    #     labels: it uses terrain/land-cover features already sitting in
    #     X.npy/dws.npy to veto predictions rice can't physically be on,
    #     applied directly to the fixed 0.5-threshold predictions.
    # ---------------------------------------------------------------------
    print("\nApplying geographic plausibility mask (elevation/slope/land-cover)...")
    X_mmap = np.load(os.path.join(DATA_DIR, "X.npy"), mmap_mode="r")
    ftr_idx = {"elevation": FEATURE_COLUMNS.index("elevation"), "slope": FEATURE_COLUMNS.index("slope")}
    # elevation/slope are static per point across months, so timestep 0 is representative
    elevation = np.asarray(X_mmap[:, 0, ftr_idx["elevation"]])
    slope     = np.asarray(X_mmap[:, 0, ftr_idx["slope"]])

    dws_raw = np.load(os.path.join(DATA_DIR, "dws.npy"))  # (N, T), raw dynamic_world codes
    NUM_DW_CLASSES = 9  # Google Dynamic World: water,trees,grass,flooded_veg,crops,shrub,built,bare,snow_ice
    # Forward-fill across timesteps to get one representative class per
    # point, skipping any leftover nodata/invalid codes (see the IndexError
    # fix from generate_embeddings_tumakuru.py — the same sentinel issue
    # applies here since this reads dws.npy directly, not the sanitized
    # in-memory version from that script).
    dw_repr = dws_raw[:, 0].copy()
    invalid = (dw_repr < 0) | (dw_repr >= NUM_DW_CLASSES)
    for t in range(1, dws_raw.shape[1]):
        dw_repr = np.where(invalid, dws_raw[:, t], dw_repr)
        invalid = (dw_repr < 0) | (dw_repr >= NUM_DW_CLASSES)
    dw_unknown = invalid  # every timestep was invalid for this point — don't mask, we don't know

    # CAVEAT: verify these two assumptions against your actual data before
    # trusting the mask — (1) that dynamic_world class codes follow Google's
    # standard 9-class Dynamic World ordering, and (2) that "slope" is in
    # degrees, not a percent-grade or radians. Adjust the sets/threshold
    # below if either doesn't hold.
    IMPLAUSIBLE_DW_CLASSES = {6, 8}  # built-up, snow_and_ice — deliberately conservative
    MAX_PLAUSIBLE_SLOPE_DEG = 15.0    # generous cutoff so real terraced paddy isn't wrongly cut

    implausible_landcover = np.isin(dw_repr, list(IMPLAUSIBLE_DW_CLASSES)) & ~dw_unknown
    implausible_slope     = slope > MAX_PLAUSIBLE_SLOPE_DEG
    implausible = implausible_landcover | implausible_slope

    preds_final = np.where((preds == 1) & implausible, 0, preds)
    n_overridden = int(((preds == 1) & implausible).sum())

    print(f"  Implausible land-cover (built-up/snow-ice): {int(implausible_landcover.sum())} points")
    print(f"  Implausible slope (> {MAX_PLAUSIBLE_SLOPE_DEG}°)         : {int(implausible_slope.sum())} points")
    print(f"  Rice predictions overridden to non-rice     : {n_overridden}")
    print(f"  Final rice count: {int(preds_final.sum())}  "
          f"(0.5 threshold: {n_rice} → after geo-mask: {int(preds_final.sum())})")

    # -----------------------------
    # 5. Metrics, only if Tumakuru actually has ground-truth labels
    #    (build_timeseries_dataset.py sets y = -1 for unlabeled points)
    # -----------------------------
    has_labels = np.any(y != -1)
    if has_labels:
        labeled_mask = y != -1
        y_true  = y[labeled_mask]
        y_pred  = preds[labeled_mask]
        y_proba = probs[labeled_mask]
        n_labeled = labeled_mask.sum()

        print(f"\nFound ground-truth labels on {n_labeled}/{len(y)} points — computing metrics on those.")
        acc       = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall    = recall_score(y_true, y_pred)
        f1        = f1_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = float("nan")  # only one class present in the labeled subset

        print("\n" + "=" * 60)
        print("Tumakuru metrics (labeled subset only)")
        print("=" * 60)
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 score : {f1:.4f}")
        print(f"ROC AUC  : {auc:.4f}")

        print("\nConfusion matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(pd.DataFrame(
            cm,
            index=["True Non-rice", "True Rice"],
            columns=["Pred Non-rice", "Pred Rice"],
        ))
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=["non-rice", "rice"]))
    else:
        print("\nNo ground-truth labels found in Tumakuru's y.npy (all -1) — "
              "skipping metrics, showing predictions only.")

    # -----------------------------
    # 6. Save predictions CSV — written in chunks, not as one giant
    #    in-memory DataFrame. At ~10M rows, a single DataFrame (especially
    #    with string location_ids) adds real memory overhead on top of
    #    everything else; streaming to disk avoids that.
    #
    #    y_pred_05    = fixed 0.5 threshold
    #    y_pred_final = after geographic plausibility mask (use this one)
    # -----------------------------
    results_path = os.path.join(OUTPUT_DIR, "tumakuru_predictions_mlp.csv")
    CSV_CHUNK = 200_000
    for i, start in enumerate(range(0, N, CSV_CHUNK)):
        end = min(start + CSV_CHUNK, N)
        chunk_df = pd.DataFrame({
            "location_id": location_ids[start:end],
            "latitude":    latlons[start:end, 0],
            "longitude":   latlons[start:end, 1],
            "y_true":      y[start:end],              # -1 where unlabeled
            "y_proba":     probs[start:end],
            "y_pred_05":   preds[start:end],
            "y_pred_final": preds_final[start:end],
        })
        chunk_df.to_csv(results_path, mode="w" if i == 0 else "a", header=(i == 0), index=False)
    print(f"\nPredictions saved → {results_path}")

    # -----------------------------
    # 7. Rice vs non-rice spatial plot — hexbin instead of scatter.
    #    At ~10M points, scatter would mean drawing 10M individual markers
    #    (slow, and requires downsampling to stay usable). hexbin instead
    #    bins points into hexagonal cells and aggregates, so it comfortably
    #    handles the full dataset in a couple of seconds. Each cell is
    #    colored by the fraction of points in it predicted as rice.
    # -----------------------------
    lat = latlons[:, 0]
    lon = latlons[:, 1]
    is_rice = (preds_final == 1).astype(np.float32)
    n_rice_final, n_non_rice_final = int(preds_final.sum()), int((preds_final == 0).sum())

    plt.figure(figsize=(9, 8))
    hb = plt.hexbin(
        lon, lat,
        C=is_rice,
        reduce_C_function=np.mean,   # each hex cell = fraction of its points predicted rice
        gridsize=200,
        cmap="RdYlGn",                # red = non-rice-dominant, green = rice-dominant
        vmin=0, vmax=1,
        mincnt=1,                     # only color cells that actually contain points
    )
    cbar = plt.colorbar(hb)
    cbar.set_label("Fraction predicted rice")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Mandya — Predicted Rice vs Non-Rice (MLP, 0.5 threshold + geo-mask)\n"
              f"{n_rice_final:,} rice / {n_non_rice_final:,} non-rice points")
    plt.gca().set_aspect("equal", adjustable="datalim")
    plot_path = os.path.join(OUTPUT_DIR, "mandya_rice_vs_nonrice_map.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Rice vs non-rice map saved → {plot_path}")


if __name__ == "__main__":
    main()
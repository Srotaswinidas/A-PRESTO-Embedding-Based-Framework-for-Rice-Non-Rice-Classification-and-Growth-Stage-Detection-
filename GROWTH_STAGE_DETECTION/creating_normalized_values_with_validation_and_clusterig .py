"""
GROWTH STAGE CLASSIFICATION VIA K-MEANS (PER-POINT NORMALIZED)
------------------------------------------------------------------------------
This method normalizes each point's own values relative to ITS OWN
  seasonal min and max (per point_id, per year) before clustering:
    normalized = (value - point's own min that year) / (point's own max - min)
  So every point's own June-November trajectory gets rescaled to roughly
  0 (that point's own seasonal low) -> 1 (that point's own seasonal high),
  regardless of whether the field is generally bright or dim overall.
  Clustering on THIS is what should actually recover growth PHASE - early,
  middle, late - since it's now measuring "where is this field right now,
  relative to its own season" rather than "how bright is this field vs
  every other field in the district."

INPUT:
  Expects your 5 *_filled.csv files (from the interpolation+median fill step).
  The following code is implemented for Mandya can be used for other districts as well.

EDIT BEFORE RUNNING:
  - FILLED_CSV_PATHS: your 5 filled CSV file paths.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# =========================== CONFIG ==========================================
FILLED_CSV_PATHS = {
    2019: "mandya_rice_filled/mandya_rice_points_2019_filled.csv",   # <-- CHANGE if different
    2020: "mandya_rice_filled/mandya_rice_points_2020_filled.csv",
    2021: "mandya_rice_filled/mandya_rice_points_2021_filled.csv",
    2022: "mandya_rice_filled/mandya_rice_points_2022_filled.csv",
    2023: "mandya_rice_filled/mandya_rice_points_2023_filled.csv",
}

CLUSTER_FEATURES = ["NDVI", "LSWI", "VV", "VH"]
N_CLUSTERS = 3
RANDOM_STATE = 42

# =========================== STEP 1: LOAD + COMBINE ALL YEARS ================
dfs = []
for year, path in FILLED_CSV_PATHS.items():
    d = pd.read_csv(path)
    d["year"] = year
    dfs.append(d)

df = pd.concat(dfs, ignore_index=True)
print(f"Combined dataset: {len(df)} rows across {len(FILLED_CSV_PATHS)} years")

# Drop June - keeping July-November only. June was where double-cropped
# fields (finishing an old crop while others are freshly transplanting)
# created ambiguous Late<->Early readings that don't reflect one clean
# growth arc for the June-November window.
df = df[df["month"] >= 7].copy()
print(f"After dropping June, keeping July-November: {len(df)} rows")

# =========================== STEP 2: PER-POINT, PER-YEAR NORMALIZATION ========
# For each (point_id, year), rescale each feature to its own seasonal
# min-max range. This is what turns "absolute brightness" into
# "relative position within this field's own season."
#
# Vectorized via groupby().transform() - fast, no per-group Python calls.
print("Starting per-point/per-year normalization...")
group_keys = ["point_id", "year"]
grouped = df.groupby(group_keys)

norm_cols = []
for col in CLUSTER_FEATURES:
    col_min = grouped[col].transform("min")
    col_max = grouped[col].transform("max")
    span = col_max - col_min
    norm_col = col + "_norm"
    # span == 0: field had identical values all season (e.g. all-filled edge
    # case) - normalized position is undefined, set to 0.5 (neutral).
    df[norm_col] = np.where(span == 0, 0.5, (df[col] - col_min) / span)
    norm_cols.append(norm_col)

# VV and VH are in dB (negative values, less negative = more backscatter).
# The min-max normalization above already handles direction correctly since
# it's relative to each point's own min/max, so no extra sign-flipping needed.

print("Normalization done.")

# =========================== STEP 3: CLUSTER ON NORMALIZED FEATURES ============
features_norm = df[norm_cols].values

print("Starting KMeans...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
df["cluster"] = kmeans.fit_predict(features_norm)
print("KMeans done.")

# =========================== STEP 3b: CLUSTER VALIDITY METRICS =================
# These evaluate how well-separated/compact the KMeans clusters are in the
# normalized NDVI/LSWI/VV/VH feature space (independent of the early/middle/
# late naming applied later - the labels are just a renaming of `cluster`).
#
# NOTE ON SILHOUETTE SCORE: by default this computes a full pairwise distance
# matrix (n_rows x n_rows). For 250,000 rows that's ~62.5 BILLION entries -
# effectively never finishes and can exhaust available RAM. We instead pass
# `sample_size` so sklearn estimates the score from a random subsample. 10,000
# points gives a stable estimate without the blow-up. davies_bouldin_score and
# calinski_harabasz_score are linear/near-linear and don't need sampling.
print("Computing cluster validity metrics (silhouette on a sample)...")
sil_score = silhouette_score(
    features_norm, df["cluster"], sample_size=10000, random_state=RANDOM_STATE
)
dbi_score = davies_bouldin_score(features_norm, df["cluster"])
chi_score = calinski_harabasz_score(features_norm, df["cluster"])
print("Validity metrics done.")

print("\nCluster validity metrics (on normalized NDVI, LSWI, VV, VH):")
print(f"  Silhouette Score:         {sil_score:.4f}  (higher is better, range -1 to 1)")
print(f"  Davies-Bouldin Index:     {dbi_score:.4f}  (lower is better, 0 = best)")
print(f"  Calinski-Harabasz Index:  {chi_score:.4f}  (higher is better)")

# =========================== STEP 4: LABEL CLUSTERS AS GROWTH STAGES ===========
# Middle = highest combined normalized score (this field is near ITS OWN
# seasonal peak right now).
agg_dict = {f"mean_{c}": (c, "mean") for c in norm_cols}
cluster_stats = df.groupby("cluster").agg(
    **agg_dict,
    mean_month=("month", "mean"),
    n=("NDVI", "size"),
)
print("\nCluster stats (normalized features, before labeling):")
print(cluster_stats)

combined_score = pd.Series(0.0, index=cluster_stats.index)
for col in norm_cols:
    mcol = f"mean_{col}"
    combined_score += cluster_stats[mcol]

middle_cluster = combined_score.idxmax()

remaining = [c for c in cluster_stats.index if c != middle_cluster]
# Early vs late: earlier average month = early, later average month = late.
# Since features are now normalized per-point, this split should now also
# roughly correspond to which side of the peak the cluster's mean_month
# actually falls on, which is the behavior we want.
remaining_sorted = cluster_stats.loc[remaining].sort_values("mean_month")
early_cluster = remaining_sorted.index[0]
late_cluster = remaining_sorted.index[1]

stage_map = {
    early_cluster: "Early (Vegetative)",
    middle_cluster: "Middle (Reproductive)",
    late_cluster: "Late (Maturity/Senescence)",
}
df["growth_stage"] = df["cluster"].map(stage_map)

print("\nCluster -> stage mapping:")
for c, stage in stage_map.items():
    stats_row = cluster_stats.loc[c]
    detail = ", ".join(f"{feat}={stats_row[f'mean_{feat}_norm']:.3f}" for feat in CLUSTER_FEATURES)
    print(f"  Cluster {c}: {stage}  ({detail}, mean month={stats_row['mean_month']:.1f}, n={stats_row['n']:.0f})")

# =========================== STEP 5: SAVE LABELED DATA =========================
OUTPUT_DIR = "growth_stage_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df.to_csv(os.path.join(OUTPUT_DIR, "mandya_rice_growth_stages_normalized.csv"), index=False)
print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'mandya_rice_growth_stages_normalized.csv')}")

# =========================== STEP 6: VALIDATE - PCA SCATTER ====================
# NOTE: plt.show() below will open a plot window and BLOCK the script until
# you close that window. If the script looks "stuck" after this point, check
# for a plot window (it may be behind other windows or off-screen).
print("Building PCA scatter plot (script will pause here until you close the plot window)...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
proj = pca.fit_transform(features_norm)
df["pca1"], df["pca2"] = proj[:, 0], proj[:, 1]

fig, ax = plt.subplots(figsize=(7, 6))
colors = {"Early (Vegetative)": "green", "Middle (Reproductive)": "orange", "Late (Maturity/Senescence)": "brown"}
for stage, color in colors.items():
    subset = df[df["growth_stage"] == stage]
    ax.scatter(subset["pca1"], subset["pca2"], s=4, alpha=0.3, color=color, label=stage)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("Growth stage clusters (normalized NDVI, LSWI, VV, VH)")
ax.legend(markerscale=4)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "growth_stage_clusters_scatter_normalized.png"), dpi=150)
plt.show()

print("PCA scatter plot closed. Building temporal profile plots...")

# =========================== STEP 7: VALIDATE - TEMPORAL PROFILES (RAW VALUES) =
# Important: plot the RAW (not normalized) NDVI/LSWI/VV/VH here, grouped by
# the new cluster labels - this is the real test of whether normalization
# fixed the problem. We want to see rise -> peak -> decline now, not the flat
# parallel-brightness-band pattern from before.
agg_dict2 = {f"mean_{c.lower()}": (c, "mean") for c in CLUSTER_FEATURES}
monthly_profile = df.groupby(["month", "growth_stage"]).agg(**agg_dict2).reset_index()

fig, axes = plt.subplots(1, len(CLUSTER_FEATURES), figsize=(22, 5))

for ax, feat in zip(axes, CLUSTER_FEATURES):
    col = f"mean_{feat.lower()}"
    for stage, color in colors.items():
        subset = monthly_profile[monthly_profile["growth_stage"] == stage].sort_values("month")
        ax.plot(subset["month"], subset[col], marker="o", color=color, label=stage)
    ax.set_title(f"{feat} temporal profile by growth stage (normalized clustering)")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Mean {feat} (raw)")
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "growth_stage_temporal_profiles_normalized.png"), dpi=150)
plt.show()

print("Temporal profile plots closed. Building stage composition plot...")

# =========================== STEP 8: SANITY CHECK - STAGE COMPOSITION PER MONTH
# Another useful check: for each calendar month, what fraction of points are
# labeled early/middle/late? Early June should be dominated by "early",
# September should be dominated by "middle", November by "late", etc.
composition = pd.crosstab(df["month"], df["growth_stage"], normalize="index")
print("\nFraction of points in each growth stage, by month:")
print(composition.round(3))

composition.plot(kind="bar", stacked=True, color=[colors[c] for c in composition.columns], figsize=(9, 5))
plt.title("Growth stage composition by month")
plt.xlabel("Month")
plt.ylabel("Fraction of points")
plt.legend(title="Growth stage", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "growth_stage_composition_by_month.png"), dpi=150)
plt.show()

print("\nDone. Check the PCA scatter, temporal profile, and month-composition plots.")
print(f"Cluster validity: silhouette={sil_score:.4f}, davies_bouldin={dbi_score:.4f}, "
      f"calinski_harabasz={chi_score:.4f}")
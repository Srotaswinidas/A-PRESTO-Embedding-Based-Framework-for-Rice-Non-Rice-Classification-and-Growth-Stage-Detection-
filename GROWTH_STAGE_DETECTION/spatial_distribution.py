"""
GRID OF SAMPLE POINTS COLORED BY GROWTH STAGE (RAW VERSION)
------------------------------------------------------------------------------
Plots sample points on a lat/lon scatter grid, one panel per (month, year).
Rows = months (July-November), columns = years (2019-2023).
The points are colored by their growth stage (Early, Middle, Late) as determined
by the clustering on normalized values.
The following code is implemented for Mandya can be used for other districts as well.

"""

import pandas as pd
import matplotlib.pyplot as plt

# =========================== CONFIG ==========================================
GROWTH_STAGE_CSV = "growth_stage_outputs/mandya_rice_growth_stages_normalized.csv"  # <-- CHANGE if needed

YEARS_TO_PLOT = [2019, 2020, 2021, 2022, 2023]
MONTHS_TO_PLOT = [7, 8, 9, 10, 11]   # July - November
MONTH_NAMES = {7: "July", 8: "August", 9: "September", 10: "October", 11: "November"}

STAGE_COLORS = {
    "Early (Vegetative)": "green",
    "Middle (Reproductive)": "orange",
    "Late (Maturity/Senescence)": "brown",
}

# =========================== PLOT GRID ========================================
df = pd.read_csv(GROWTH_STAGE_CSV)

n_rows = len(MONTHS_TO_PLOT)
n_cols = len(YEARS_TO_PLOT)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(3.2 * n_cols, 3.2 * n_rows),
    sharex=True, sharey=True,
)

for row, month in enumerate(MONTHS_TO_PLOT):
    for col, year in enumerate(YEARS_TO_PLOT):
        ax = axes[row, col]
        subset = df[(df["year"] == year) & (df["month"] == month)]

        ax.set_xticks([])
        ax.set_yticks([])

        if subset.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_aspect("equal")
            continue

        for stage, color in STAGE_COLORS.items():
            stage_subset = subset[subset["growth_stage"] == stage]
            ax.scatter(stage_subset["longitude"], stage_subset["latitude"],
                       s=1.5, alpha=0.5, color=color, label=stage,
                       edgecolors="none")

        ax.text(0.03, 0.95, f"n={len(subset)}", transform=ax.transAxes,
                 fontsize=7, color="dimgray", va="top", ha="left")
        ax.set_aspect("equal")

# Column headers (years)
for col, year in enumerate(YEARS_TO_PLOT):
    axes[0, col].set_title(str(year), fontsize=12, fontweight="bold", pad=8)

# Row headers (months)
for row, month in enumerate(MONTHS_TO_PLOT):
    axes[row, 0].set_ylabel(MONTH_NAMES[month], fontsize=12, fontweight="bold",
                              rotation=90, labelpad=10)

# Outer axes ticks
for col in range(n_cols):
    axes[-1, col].set_xlabel("Longitude", fontsize=9)
    axes[-1, col].tick_params(axis="x", labelsize=7)
    axes[-1, col].set_xticks(axes[-1, col].get_xlim())
for row in range(n_rows):
    axes[row, 0].tick_params(axis="y", labelsize=7)
    axes[row, 0].set_yticks(axes[row, 0].get_ylim())

fig.subplots_adjust(wspace=0.05, hspace=0.12, top=0.90, bottom=0.08)

# Shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
if not handles:
    for row in range(n_rows):
        for col in range(n_cols):
            handles, labels = axes[row, col].get_legend_handles_labels()
            if handles:
                break
        if handles:
            break

fig.legend(handles, labels, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, 0.0), markerscale=6, fontsize=10, frameon=False)

fig.suptitle("Mandya rice growth stages (raw) — months × years, 2019-2023",
             fontsize=15, y=0.98)
fig.savefig("growth_stage_grid_raw.png", dpi=200, bbox_inches="tight")
plt.show()

print("Point counts per (month, year):")
for month in MONTHS_TO_PLOT:
    for year in YEARS_TO_PLOT:
        subset = df[(df["year"] == year) & (df["month"] == month)]
        print(f"\n{MONTH_NAMES[month]} {year}: {len(subset)} points")
        print(subset["growth_stage"].value_counts())

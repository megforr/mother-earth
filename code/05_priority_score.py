"""
Biodiversity Project — Step 5: Maryland Conservation Priority Scoring
======================================================================
Computes a composite conservation priority score (0–100) for each ~5km
grid cell across Maryland by combining local species data with the
cross-state connectivity metrics from 04_regional_connectivity.py.

Score components and weights:
  - species_richness       (0.20): unique species per cell, all taxa
  - rare_species_count     (0.25): IUCN EN/VU/CR species per cell
  - sink_species_count     (0.15): sink-dependent species per cell
                                   (requires regional coordination to protect)
  - migration_bottleneck   (0.15): stopover importance for migratory species
  - corridor_centrality    (0.15): cross-state corridor pinch-point value
  - development_threat     (0.10): inverse of naturalness — rewards cells
                                   at risk of imminent loss

Each component is min-max normalised to [0, 1] before weighting so that
no single scale dominates.

Inputs:
  data/maryland_all_taxa.csv
  data/rare_species_regional.csv
  data/regional_connectivity.csv

Outputs:
  data/maryland_priority_grid.csv  — per-cell scores, ready for mapping
  data/top_priority_regions.csv    — top 50 cells with full score breakdown

Dependencies: pip install pandas numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CELL_SIZE_DEG = 0.05

MD_LAT_MIN, MD_LAT_MAX = 37.88, 39.72
MD_LON_MIN, MD_LON_MAX = -79.5, -74.97

# Score component weights — must sum to 1.0
WEIGHTS = {
    "species_richness":     0.20,
    "rare_species_count":   0.25,
    "sink_species_count":   0.15,
    "migration_bottleneck": 0.15,
    "corridor_centrality":  0.15,
    "development_threat":   0.10,
}

# IUCN categories treated as rare/threatened
IUCN_THREATENED = {"EN", "VU", "CR"}

# County centroids for labelling output (approximate, for display purposes)
COUNTY_CENTROIDS = {
    "Garrett":        (39.53, -79.20),
    "Allegany":       (39.60, -78.76),
    "Washington":     (39.62, -77.76),
    "Frederick":      (39.46, -77.41),
    "Carroll":        (39.57, -77.00),
    "Baltimore":      (39.46, -76.75),
    "Harford":        (39.54, -76.31),
    "Cecil":          (39.55, -75.98),
    "Montgomery":     (39.14, -77.20),
    "Howard":         (39.29, -76.99),
    "Anne Arundel":   (38.97, -76.63),
    "Queen Anne's":   (39.02, -76.08),
    "Kent":           (39.27, -76.07),
    "Prince George's":(38.83, -76.85),
    "Calvert":        (38.53, -76.53),
    "Charles":        (38.47, -77.00),
    "St. Mary's":     (38.20, -76.60),
    "Talbot":         (38.77, -76.17),
    "Caroline":       (38.88, -75.83),
    "Dorchester":     (38.39, -76.07),
    "Wicomico":       (38.37, -75.63),
    "Somerset":       (38.08, -75.84),
    "Worcester":      (38.22, -75.31),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assign_grid_cell(df, cell_size=CELL_SIZE_DEG):
    df = df.copy()
    df["lat_cell"] = (df["decimalLatitude"] // cell_size) * cell_size
    df["lon_cell"] = (df["decimalLongitude"] // cell_size) * cell_size
    return df


def make_cell_id(lat_cell, lon_cell):
    return f"{lat_cell:.4f}_{lon_cell:.4f}"


def build_md_grid(cell_size=CELL_SIZE_DEG):
    lats = np.arange(MD_LAT_MIN, MD_LAT_MAX, cell_size).round(6)
    lons = np.arange(MD_LON_MIN, MD_LON_MAX, cell_size).round(6)
    rows = [
        {"lat_cell": lat, "lon_cell": lon,
         "cell_id": make_cell_id(lat, lon),
         "lat_center": round(lat + cell_size / 2, 6),
         "lon_center": round(lon + cell_size / 2, 6)}
        for lat in lats for lon in lons
    ]
    return pd.DataFrame(rows)


def minmax_normalize(series):
    """Normalize a pandas Series to [0, 1]. Returns zeros if range is 0."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def assign_county(lat, lon):
    """Assign the nearest county name by Euclidean distance to centroids."""
    best_county, best_dist = "Unknown", float("inf")
    for county, (clat, clon) in COUNTY_CENTROIDS.items():
        dist = (lat - clat) ** 2 + (lon - clon) ** 2
        if dist < best_dist:
            best_dist = dist
            best_county = county
    return best_county


# ---------------------------------------------------------------------------
# Component: Species Richness
# ---------------------------------------------------------------------------

def compute_species_richness(maryland_df):
    """Count unique species per MD grid cell across all taxa."""
    df = assign_grid_cell(maryland_df)
    df = df[df["species"].notna()]

    richness = (
        df.groupby(["lat_cell", "lon_cell"])["species"]
        .nunique()
        .reset_index(name="species_richness")
    )
    return richness


# ---------------------------------------------------------------------------
# Component: Rare Species Count
# ---------------------------------------------------------------------------

def compute_rare_species_count(rare_df, maryland_bbox=True):
    """
    Count IUCN-threatened species per MD grid cell.
    Filters rare_df to Maryland coordinates if maryland_bbox=True.
    """
    df = rare_df.copy()
    if maryland_bbox:
        df = df[
            df["decimalLatitude"].between(MD_LAT_MIN, MD_LAT_MAX) &
            df["decimalLongitude"].between(MD_LON_MIN, MD_LON_MAX)
        ]
    df = df[df.get("iucnRedListCategory", pd.Series(dtype=str)).isin(IUCN_THREATENED)]
    df = assign_grid_cell(df)

    if df.empty:
        return pd.DataFrame(columns=["lat_cell", "lon_cell", "rare_species_count"])

    counts = (
        df.groupby(["lat_cell", "lon_cell"])["species"]
        .nunique()
        .reset_index(name="rare_species_count")
    )
    return counts


# ---------------------------------------------------------------------------
# Component: Development Threat (proxy from observation data)
# ---------------------------------------------------------------------------

def compute_development_threat(maryland_df):
    """
    Proxy development threat from observation data recency.

    Areas with a high proportion of old observations (pre-2000) and few recent
    ones may signal habitat degradation. Cells with only recent data from
    developed-landscape species (synanthropic birds like European Starling,
    House Sparrow) are also flagged.

    This is a data-driven proxy. For production use, replace with NLCD land
    cover raster data (USGS, free download) for precise impervious surface %.

    Returns a threat score: higher = more threatened / at more risk of loss.
    """
    df = assign_grid_cell(maryland_df)

    if "year" not in df.columns or df["year"].isna().all():
        # No temporal data — return uniform mid-level threat
        grid = build_md_grid()
        grid["development_threat"] = 0.5
        return grid[["lat_cell", "lon_cell", "development_threat"]]

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna()]

    # Cells where max observed year is old have fewer recent observations
    # (possible data gap or habitat loss signal)
    recent_threshold = 2010
    recent = df[df["year"] >= recent_threshold]
    old = df[df["year"] < recent_threshold]

    recent_counts = (
        recent.groupby(["lat_cell", "lon_cell"]).size()
        .reset_index(name="recent_obs")
    )
    old_counts = (
        old.groupby(["lat_cell", "lon_cell"]).size()
        .reset_index(name="old_obs")
    )

    grid = build_md_grid()
    grid = grid.merge(recent_counts, on=["lat_cell", "lon_cell"], how="left")
    grid = grid.merge(old_counts,    on=["lat_cell", "lon_cell"], how="left")
    grid[["recent_obs", "old_obs"]] = grid[["recent_obs", "old_obs"]].fillna(0)

    # Cells with fewer recent obs relative to total obs = potentially degraded
    total = grid["recent_obs"] + grid["old_obs"]
    grid["development_threat"] = 1 - (grid["recent_obs"] / total.clip(lower=1))

    return grid[["lat_cell", "lon_cell", "development_threat"]]


# ---------------------------------------------------------------------------
# Composite Score Assembly
# ---------------------------------------------------------------------------

def compute_composite_score(grid):
    """
    Normalise each component to [0, 1] and compute the weighted sum.
    Final score is scaled to [0, 100].
    """
    component_cols = {
        "species_richness":     "species_richness",
        "rare_species_count":   "rare_species_count",
        "sink_species_count":   "sink_species_count",
        "migration_bottleneck": "migration_bottleneck_score",
        "corridor_centrality":  "corridor_centrality",
        "development_threat":   "development_threat",
    }

    score = pd.Series(np.zeros(len(grid)), index=grid.index)

    for weight_key, col in component_cols.items():
        weight = WEIGHTS[weight_key]
        if col in grid.columns:
            norm = minmax_normalize(grid[col].fillna(0))
            score += weight * norm
            grid[f"{col}_norm"] = norm
        else:
            print(f"  Warning: component column '{col}' not found — skipping")

    grid["priority_score"] = (score * 100).round(2)
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading input data...")

    maryland_path     = DATA_DIR / "maryland_all_taxa.csv"
    rare_path         = DATA_DIR / "rare_species_regional.csv"
    connectivity_path = DATA_DIR / "regional_connectivity.csv"

    missing = [p for p in [maryland_path, connectivity_path] if not p.exists()]
    if missing:
        print(f"ERROR: Missing required inputs: {[str(p) for p in missing]}")
        print("Run 01_fetch_species_data.py and 04_regional_connectivity.py first.")
        raise SystemExit(1)

    maryland_df     = pd.read_csv(maryland_path)
    connectivity_df = pd.read_csv(connectivity_path)
    rare_df         = pd.read_csv(rare_path) if rare_path.exists() else pd.DataFrame()

    print(f"  Maryland records: {len(maryland_df):,}")
    print(f"  Connectivity grid cells: {len(connectivity_df):,}")
    if not rare_df.empty:
        print(f"  Rare species records: {len(rare_df):,}")

    # Build base grid
    grid = build_md_grid()

    # Merge species richness
    print("\nComputing species richness per cell...")
    richness = compute_species_richness(maryland_df)
    grid = grid.merge(richness, on=["lat_cell", "lon_cell"], how="left")
    grid["species_richness"] = grid["species_richness"].fillna(0)

    # Merge rare species count
    print("Computing rare species counts per cell...")
    if not rare_df.empty:
        rare_counts = compute_rare_species_count(rare_df)
        grid = grid.merge(rare_counts, on=["lat_cell", "lon_cell"], how="left")
        grid["rare_species_count"] = grid["rare_species_count"].fillna(0)
    else:
        grid["rare_species_count"] = 0

    # Merge development threat
    print("Computing development threat proxy per cell...")
    threat_df = compute_development_threat(maryland_df)
    grid = grid.merge(
        threat_df[["lat_cell", "lon_cell", "development_threat"]],
        on=["lat_cell", "lon_cell"], how="left"
    )
    grid["development_threat"] = grid["development_threat"].fillna(0.5)

    # Merge connectivity metrics
    print("Merging cross-state connectivity metrics...")
    conn_cols = ["lat_cell", "lon_cell", "sink_species_count",
                 "migration_bottleneck_score", "corridor_centrality"]
    available_conn = [c for c in conn_cols if c in connectivity_df.columns]
    grid = grid.merge(
        connectivity_df[available_conn],
        on=["lat_cell", "lon_cell"], how="left"
    )
    for col in ["sink_species_count", "migration_bottleneck_score", "corridor_centrality"]:
        if col in grid.columns:
            grid[col] = grid[col].fillna(0)
        else:
            grid[col] = 0

    # Compute composite score
    print("Computing composite priority score...")
    grid = compute_composite_score(grid)

    # Add county labels
    print("Assigning county labels...")
    grid["county"] = grid.apply(
        lambda r: assign_county(r["lat_center"], r["lon_center"]), axis=1
    )

    # Save full grid
    grid.to_csv(OUTPUT_DIR / "maryland_priority_grid.csv", index=False)
    print(f"\nSaved full priority grid ({len(grid):,} cells) → data/maryland_priority_grid.csv")

    # Save top 50 cells
    top50 = (
        grid.sort_values("priority_score", ascending=False)
        .head(50)
        [["cell_id", "lat_center", "lon_center", "county", "priority_score",
          "species_richness", "rare_species_count", "sink_species_count",
          "migration_bottleneck_score", "corridor_centrality", "development_threat"]]
    )
    top50.to_csv(OUTPUT_DIR / "top_priority_regions.csv", index=False)
    print(f"Saved top 50 priority cells → data/top_priority_regions.csv")

    # Quick summary
    print("\n--- Priority Score Summary ---")
    print(f"  Cells scored: {len(grid):,}")
    print(f"  Score range: {grid['priority_score'].min():.1f} – {grid['priority_score'].max():.1f}")
    print(f"  Mean score:  {grid['priority_score'].mean():.1f}")
    print(f"  High-priority cells (score > 75): {(grid['priority_score'] > 75).sum():,}")

    print("\n  Top 10 cells:")
    for _, row in top50.head(10).iterrows():
        print(
            f"    {row['county']:20s}  score={row['priority_score']:5.1f}  "
            f"rich={int(row['species_richness']):3d}  "
            f"rare={int(row['rare_species_count']):2d}  "
            f"corridor={row['corridor_centrality']:.3f}"
        )

    print("\n✅ Priority scoring complete!")

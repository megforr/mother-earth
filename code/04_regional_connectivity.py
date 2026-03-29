"""
Biodiversity Project — Step 4: Regional Cross-State Connectivity Analysis
=========================================================================
Computes three connectivity metrics for each Maryland grid cell that capture
how a given cell's protection value depends on the broader 5-state ecosystem:

  1. Source-Sink Flags
     For each species observed in Maryland, compare its observation density
     in MD vs. adjacent states. Species with low MD density relative to their
     regional density are flagged as potentially sink-dependent — their MD
     populations may not be self-sustaining without immigration from VA/PA/WV.
     Per-cell output: count of sink-dependent species observed in the cell.

  2. Migration Bottleneck Score
     Identifies cells where migrant species concentrate during spring/fall
     passage but rarely breed in summer. These stopover sites have conservation
     value invisible to static species richness counts.

  3. Cross-State Corridor Centrality
     Builds a habitat graph across all 5 states (nodes = high-richness grid
     cells, edges = spatial adjacency). For each MD cell, computes its
     betweenness centrality — cells that sit on the only viable path between
     VA/WV habitat and PA/DE habitat score highest. These are the corridor
     pinch-points most critical to wide-ranging species.

Inputs (produced by 01_fetch_species_data.py):
  data/regional_all_taxa.csv
  data/maryland_all_taxa.csv

Outputs:
  data/regional_connectivity.csv  — per-cell connectivity metrics for all
                                    MD grid cells, ready to join into the
                                    priority score in 05_priority_score.py

Dependencies: pip install pandas numpy networkx
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Grid cell size in degrees (~5 km at mid-latitudes)
CELL_SIZE_DEG = 0.05

# Maryland bounding box
MD_LAT_MIN, MD_LAT_MAX = 37.88, 39.72
MD_LON_MIN, MD_LON_MAX = -79.5, -74.97

# Spring and fall migration windows (months)
SPRING_MONTHS = {3, 4, 5}
FALL_MONTHS   = {8, 9, 10, 11}
SUMMER_MONTHS = {6, 7}

# Source-sink threshold: if a species' MD record fraction is less than this
# fraction of the regional average per-state fraction, flag it as sink-dependent.
# e.g., 0.5 means MD has less than half the representation expected if evenly spread.
SOURCE_SINK_THRESHOLD = 0.5

# Richness percentile cutoff for corridor graph nodes
CORRIDOR_NODE_PERCENTILE = 75  # top quartile of grid cells by species richness


# ---------------------------------------------------------------------------
# Helper: assign each record to a grid cell
# ---------------------------------------------------------------------------

def assign_grid_cell(df, cell_size=CELL_SIZE_DEG):
    """Add lat_cell and lon_cell columns (lower-left corner of each grid cell)."""
    df = df.copy()
    df["lat_cell"] = (df["decimalLatitude"] // cell_size) * cell_size
    df["lon_cell"] = (df["decimalLongitude"] // cell_size) * cell_size
    return df


def make_cell_id(lat_cell, lon_cell):
    return f"{lat_cell:.4f}_{lon_cell:.4f}"


# ---------------------------------------------------------------------------
# 1. Source-Sink Analysis
# ---------------------------------------------------------------------------

def compute_source_sink_flags(regional_df, maryland_df):
    """
    Flag species as sink-dependent in Maryland based on relative observation
    density compared to the wider 5-state region.

    A species is flagged as sink-dependent if its Maryland observation fraction
    is less than SOURCE_SINK_THRESHOLD × (1 / number_of_states_it_appears_in).

    Args:
        regional_df: DataFrame with stateProvince and species columns (5-state)
        maryland_df: DataFrame for Maryland records

    Returns:
        sink_species: set of species names flagged as sink-dependent
        species_flags: DataFrame with per-species sink classification
    """
    # Normalise stateProvince values to short codes
    state_map = {
        "Maryland": "MD", "Virginia": "VA", "Pennsylvania": "PA",
        "West Virginia": "WV", "Delaware": "DE",
        "MD": "MD", "VA": "VA", "PA": "PA", "WV": "WV", "DE": "DE",
    }
    reg = regional_df.copy()
    reg["state_code"] = reg["stateProvince"].map(state_map)
    reg = reg[reg["state_code"].notna() & reg["species"].notna()]

    if reg.empty:
        print("  Source-sink: no usable regional records — skipping")
        return set(), pd.DataFrame()

    # Total records per species
    species_total = reg.groupby("species").size().rename("regional_count")

    # Records per species per state
    species_state = reg.groupby(["species", "state_code"]).size().reset_index(name="state_count")

    # Maryland counts
    md_counts = (
        species_state[species_state["state_code"] == "MD"]
        .set_index("species")["state_count"]
        .rename("md_count")
    )

    # Number of states each species appears in
    states_present = (
        species_state.groupby("species")["state_code"]
        .nunique()
        .rename("n_states")
    )

    summary = pd.concat([species_total, md_counts, states_present], axis=1).fillna(0)
    summary["md_fraction"] = summary["md_count"] / summary["regional_count"]
    summary["expected_fraction"] = 1.0 / summary["n_states"].clip(lower=1)
    summary["sink_ratio"] = summary["md_fraction"] / summary["expected_fraction"].clip(lower=1e-6)

    # Flag species where MD representation is well below expectation
    summary["is_sink_dependent"] = summary["sink_ratio"] < SOURCE_SINK_THRESHOLD

    sink_species = set(summary[summary["is_sink_dependent"]].index)
    print(f"  Source-sink: {len(sink_species)} sink-dependent species identified")

    return sink_species, summary.reset_index()


def compute_sink_count_per_cell(maryland_df, sink_species):
    """
    For each Maryland grid cell, count how many sink-dependent species were observed.

    Returns:
        DataFrame with columns [lat_cell, lon_cell, cell_id, sink_species_count]
    """
    md = assign_grid_cell(maryland_df)
    md = md[md["species"].isin(sink_species)]

    if md.empty:
        # Return empty grid with zeros
        grid = build_md_grid()
        grid["sink_species_count"] = 0
        return grid

    counts = (
        md.groupby(["lat_cell", "lon_cell"])["species"]
        .nunique()
        .reset_index(name="sink_species_count")
    )
    counts["cell_id"] = counts.apply(
        lambda r: make_cell_id(r["lat_cell"], r["lon_cell"]), axis=1
    )
    return counts


# ---------------------------------------------------------------------------
# 2. Migration Bottleneck Score
# ---------------------------------------------------------------------------

def compute_migration_score(regional_df):
    """
    Score each Maryland grid cell by its importance as a migration stopover.

    Method:
      - Count unique species observed per cell during spring/fall migration months
      - Count unique species observed per cell during summer (breeding season)
      - Migration score = migrant species count / (summer species count + 1)
      - Cells with many migrants but few summer residents are stopover sites

    Returns:
        DataFrame with columns [lat_cell, lon_cell, cell_id, migration_bottleneck_score]
    """
    df = assign_grid_cell(regional_df)

    # Filter to Maryland cells only
    df = df[
        df["lat_cell"].between(MD_LAT_MIN, MD_LAT_MAX) &
        df["lon_cell"].between(MD_LON_MIN, MD_LON_MAX)
    ]

    if "month" not in df.columns or df["month"].isna().all():
        print("  Migration score: no month data available — assigning zeros")
        grid = build_md_grid()
        grid["migration_bottleneck_score"] = 0.0
        return grid

    df["month"] = pd.to_numeric(df["month"], errors="coerce")

    migrants = df[df["month"].isin(SPRING_MONTHS | FALL_MONTHS)]
    summer   = df[df["month"].isin(SUMMER_MONTHS)]

    migrant_counts = (
        migrants.groupby(["lat_cell", "lon_cell"])["species"]
        .nunique()
        .reset_index(name="migrant_species")
    )
    summer_counts = (
        summer.groupby(["lat_cell", "lon_cell"])["species"]
        .nunique()
        .reset_index(name="summer_species")
    )

    grid = build_md_grid()
    grid = grid.merge(migrant_counts, on=["lat_cell", "lon_cell"], how="left")
    grid = grid.merge(summer_counts, on=["lat_cell", "lon_cell"], how="left")
    grid[["migrant_species", "summer_species"]] = grid[["migrant_species", "summer_species"]].fillna(0)

    grid["migration_bottleneck_score"] = (
        grid["migrant_species"] / (grid["summer_species"] + 1)
    )

    print(
        f"  Migration score: {(grid['migration_bottleneck_score'] > 1).sum()} "
        f"cells flagged as migration-dominant"
    )
    return grid[["lat_cell", "lon_cell", "cell_id", "migration_bottleneck_score"]]


# ---------------------------------------------------------------------------
# 3. Cross-State Corridor Centrality
# ---------------------------------------------------------------------------

def compute_corridor_centrality(regional_df):
    """
    Build a habitat graph across all 5 states and compute the betweenness
    centrality of each Maryland cell.

    High-centrality MD cells sit on the critical path between southern habitat
    (VA/WV) and northern habitat (PA/DE) — protecting them preserves the
    connectivity that wide-ranging species depend on.

    Method:
      1. Grid all 5-state records at CELL_SIZE_DEG resolution
      2. Identify nodes = cells in the top CORRIDOR_NODE_PERCENTILE by richness
      3. Add edges between nodes within 2 cell-widths of each other (adjacency)
      4. Compute betweenness centrality for all nodes
      5. Return centrality scores for MD cells only

    Returns:
        DataFrame with columns [lat_cell, lon_cell, cell_id, corridor_centrality]
    """
    try:
        import networkx as nx
    except ImportError:
        print("  networkx not installed (pip install networkx) — corridor centrality set to 0")
        grid = build_md_grid()
        grid["corridor_centrality"] = 0.0
        return grid[["lat_cell", "lon_cell", "cell_id", "corridor_centrality"]]

    df = assign_grid_cell(regional_df)
    df = df[df["species"].notna()]

    # Species richness per cell across all 5 states
    richness = (
        df.groupby(["lat_cell", "lon_cell"])["species"]
        .nunique()
        .reset_index(name="richness")
    )

    # Keep only top-quartile cells as graph nodes
    threshold = richness["richness"].quantile(CORRIDOR_NODE_PERCENTILE / 100)
    nodes = richness[richness["richness"] >= threshold].copy()
    nodes["cell_id"] = nodes.apply(
        lambda r: make_cell_id(r["lat_cell"], r["lon_cell"]), axis=1
    )

    print(f"  Corridor graph: {len(nodes)} nodes (richness >= {threshold:.0f} species)")

    # Build graph with edges between spatially adjacent nodes
    G = nx.Graph()
    G.add_nodes_from(nodes["cell_id"])

    node_coords = dict(zip(
        nodes["cell_id"],
        zip(nodes["lat_cell"], nodes["lon_cell"])
    ))

    # Spatial index: map (lat_i, lon_i) → cell_id for O(1) adjacency lookup
    coord_to_id = {v: k for k, v in node_coords.items()}
    step = CELL_SIZE_DEG

    for cell_id, (lat, lon) in node_coords.items():
        # Check 8-neighbors + 2-cell-radius neighbours (allows 1 cell gap)
        for dlat in [-2 * step, -step, 0, step, 2 * step]:
            for dlon in [-2 * step, -step, 0, step, 2 * step]:
                if dlat == 0 and dlon == 0:
                    continue
                neighbour_lat = round(lat + dlat, 6)
                neighbour_lon = round(lon + dlon, 6)
                neighbour_id = coord_to_id.get((neighbour_lat, neighbour_lon))
                if neighbour_id and not G.has_edge(cell_id, neighbour_id):
                    G.add_edge(cell_id, neighbour_id)

    print(f"  Corridor graph: {G.number_of_edges()} edges")

    # Betweenness centrality (normalized)
    centrality = nx.betweenness_centrality(G, normalized=True, endpoints=False)

    # Extract MD cells only
    md_nodes = nodes[
        nodes["lat_cell"].between(MD_LAT_MIN, MD_LAT_MAX) &
        nodes["lon_cell"].between(MD_LON_MIN, MD_LON_MAX)
    ].copy()
    md_nodes["corridor_centrality"] = md_nodes["cell_id"].map(centrality).fillna(0)

    # Full MD grid with zeros for cells not in the graph
    grid = build_md_grid()
    grid = grid.merge(
        md_nodes[["lat_cell", "lon_cell", "corridor_centrality"]],
        on=["lat_cell", "lon_cell"],
        how="left"
    )
    grid["corridor_centrality"] = grid["corridor_centrality"].fillna(0)

    n_high = (grid["corridor_centrality"] > 0).sum()
    print(f"  Corridor centrality: {n_high} MD cells with nonzero centrality")
    return grid[["lat_cell", "lon_cell", "cell_id", "corridor_centrality"]]


# ---------------------------------------------------------------------------
# Grid builder helpers
# ---------------------------------------------------------------------------

def build_md_grid(cell_size=CELL_SIZE_DEG):
    """Build a complete grid of all cells covering Maryland."""
    lats = np.arange(MD_LAT_MIN, MD_LAT_MAX, cell_size).round(6)
    lons = np.arange(MD_LON_MIN, MD_LON_MAX, cell_size).round(6)
    rows = [
        {"lat_cell": lat, "lon_cell": lon, "cell_id": make_cell_id(lat, lon)}
        for lat in lats for lon in lons
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    regional_path = DATA_DIR / "regional_all_taxa.csv"
    maryland_path = DATA_DIR / "maryland_all_taxa.csv"

    if not regional_path.exists() or not maryland_path.exists():
        print("ERROR: Run 01_fetch_species_data.py first to generate input CSVs.")
        raise SystemExit(1)

    print("Loading datasets...")
    regional_df = pd.read_csv(regional_path)
    maryland_df = pd.read_csv(maryland_path)
    print(f"  Regional: {len(regional_df):,} records")
    print(f"  Maryland: {len(maryland_df):,} records")

    # 1. Source-sink analysis
    print("\n--- Source-Sink Analysis ---")
    sink_species, species_flags = compute_source_sink_flags(regional_df, maryland_df)
    sink_counts = compute_sink_count_per_cell(maryland_df, sink_species)

    if not species_flags.empty:
        species_flags.to_csv(OUTPUT_DIR / "species_sink_flags.csv", index=False)
        print(f"  Saved species flags → data/species_sink_flags.csv")

    # 2. Migration bottleneck
    print("\n--- Migration Bottleneck Analysis ---")
    migration_df = compute_migration_score(regional_df)

    # 3. Corridor centrality
    print("\n--- Cross-State Corridor Centrality ---")
    centrality_df = compute_corridor_centrality(regional_df)

    # 4. Merge all metrics into one grid
    print("\n--- Assembling Connectivity Grid ---")
    grid = build_md_grid()

    for metric_df, merge_cols in [
        (sink_counts,   ["sink_species_count"]),
        (migration_df,  ["migration_bottleneck_score"]),
        (centrality_df, ["corridor_centrality"]),
    ]:
        if not metric_df.empty and all(c in metric_df.columns for c in merge_cols):
            grid = grid.merge(
                metric_df[["lat_cell", "lon_cell"] + merge_cols],
                on=["lat_cell", "lon_cell"],
                how="left"
            )

    fill_cols = ["sink_species_count", "migration_bottleneck_score", "corridor_centrality"]
    for col in fill_cols:
        if col in grid.columns:
            grid[col] = grid[col].fillna(0)
        else:
            grid[col] = 0

    grid.to_csv(OUTPUT_DIR / "regional_connectivity.csv", index=False)
    print(f"\nSaved connectivity grid ({len(grid):,} MD cells) → data/regional_connectivity.csv")
    print("\n✅ Regional connectivity analysis complete!")

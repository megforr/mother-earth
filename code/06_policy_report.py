"""
Biodiversity Project — Step 6: Policy-Facing Report Generation
==============================================================
Generates outputs designed for Maryland DNR staff and legislators.

Outputs:
  data/maryland_priority_map.html      — interactive choropleth map
  data/top_priority_regions.csv        — top 50 cells (already written by step 5,
                                         reproduced here with enriched labels)
  data/cross_state_dependencies.csv    — species requiring multi-state coordination
  data/protection_gap_by_county.csv    — % of high-priority habitat unprotected

Run after 05_priority_score.py.

Dependencies: pip install pandas numpy folium
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR  = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR

# Cells with priority_score above this threshold are classified as "high priority"
HIGH_PRIORITY_THRESHOLD = 70

# Top N cells to highlight on the interactive map
TOP_N_HIGHLIGHTED = 10

# ---------------------------------------------------------------------------
# 1. Interactive Priority Map
# ---------------------------------------------------------------------------

def create_priority_map(grid_df, top_df):
    """
    Build an interactive Folium map with:
      - Choropleth of priority scores across all MD grid cells
      - Markers for the top TOP_N_HIGHLIGHTED unprotected priority cells
      - State boundary context for MD/VA/PA/WV/DE neighbours
    """
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("  folium not installed (pip install folium) — skipping interactive map")
        return

    # Center on Maryland's geographic center
    m = folium.Map(location=[39.0, -76.8], zoom_start=8, tiles="CartoDB positron")

    # --- Priority score choropleth via rectangle grid cells ---
    cell_size = 0.05
    score_min = grid_df["priority_score"].min()
    score_max = grid_df["priority_score"].max()

    def score_to_color(score):
        """Map score 0–100 to a red-green colormap (green=high priority)."""
        norm = (score - score_min) / max(score_max - score_min, 1)
        # Interpolate from light yellow (low) → orange → dark red (high)
        r = int(255)
        g = int(255 * (1 - norm))
        b = int(0)
        return f"#{r:02x}{g:02x}{b:02x}"

    # Add one rectangle per grid cell (subsample to keep map performant)
    # Only render cells with score > 0 to reduce clutter
    render_df = grid_df[grid_df["priority_score"] > 0].copy()

    for _, row in render_df.iterrows():
        lat, lon = row["lat_cell"], row["lon_cell"]
        score = row["priority_score"]
        color = score_to_color(score)
        folium.Rectangle(
            bounds=[[lat, lon], [lat + cell_size, lon + cell_size]],
            color=None,
            fill=True,
            fill_color=color,
            fill_opacity=0.55,
            popup=folium.Popup(
                f"<b>{row.get('county', '')} County</b><br>"
                f"Priority Score: <b>{score:.1f}</b><br>"
                f"Species Richness: {int(row.get('species_richness', 0))}<br>"
                f"Rare Species: {int(row.get('rare_species_count', 0))}<br>"
                f"Sink-Dependent Species: {int(row.get('sink_species_count', 0))}<br>"
                f"Migration Bottleneck: {row.get('migration_bottleneck_score', 0):.2f}<br>"
                f"Corridor Centrality: {row.get('corridor_centrality', 0):.4f}",
                max_width=250
            ),
            tooltip=f"Score: {score:.1f}"
        ).add_to(m)

    # --- Top priority cell markers ---
    top_cells = top_df.head(TOP_N_HIGHLIGHTED)
    for rank, (_, row) in enumerate(top_cells.iterrows(), start=1):
        folium.Marker(
            location=[row["lat_center"], row["lon_center"]],
            popup=folium.Popup(
                f"<b>#{rank} Priority Site</b><br>"
                f"{row.get('county', '')} County<br>"
                f"Score: <b>{row['priority_score']:.1f} / 100</b><br>"
                f"Species Richness: {int(row.get('species_richness', 0))}<br>"
                f"Rare Species: {int(row.get('rare_species_count', 0))}<br>"
                f"Cross-state Corridor: {row.get('corridor_centrality', 0):.4f}",
                max_width=250
            ),
            icon=folium.Icon(color="red", icon="star"),
            tooltip=f"#{rank}: {row.get('county', '')} — Score {row['priority_score']:.0f}"
        ).add_to(m)

    # --- Reference markers for context ---
    landmarks = {
        "Annapolis (Capital)":       (38.97, -76.49),
        "Blackwater NWR":            (38.45, -76.07),
        "Catoctin Mtn / South Mtn":  (39.62, -77.46),
        "Chesapeake & Ohio Canal":   (39.10, -77.55),
        "Patuxent Research Refuge":  (39.07, -76.80),
    }
    for name, (lat, lon) in landmarks.items():
        folium.Marker(
            location=[lat, lon],
            popup=name,
            icon=folium.Icon(color="blue", icon="info-sign"),
            tooltip=name
        ).add_to(m)

    # --- Color legend (manual HTML) ---
    legend_html = """
    <div style="position: fixed; bottom: 40px; left: 40px; z-index: 1000;
                background-color: white; padding: 12px; border-radius: 6px;
                border: 1px solid #ccc; font-family: sans-serif; font-size: 12px;">
        <b>Conservation Priority Score</b><br>
        <span style="background:#ffff00; padding: 2px 8px;">&nbsp;</span> Low (0)<br>
        <span style="background:#ff8800; padding: 2px 8px;">&nbsp;</span> Medium (50)<br>
        <span style="background:#ff0000; padding: 2px 8px;">&nbsp;</span> High (100)<br>
        <hr style="margin:6px 0">
        <span>&#9733; Red star = top priority site</span><br>
        <span>&#9432; Blue = known conservation areas</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    output_path = OUTPUT_DIR / "maryland_priority_map.html"
    m.save(str(output_path))
    print(f"  Saved interactive map → {output_path}")
    print("  Open maryland_priority_map.html in any browser.")


# ---------------------------------------------------------------------------
# 2. Cross-State Dependencies Report
# ---------------------------------------------------------------------------

def build_cross_state_dependencies(species_flags_path):
    """
    Build a report of species that Maryland cannot protect on its own —
    i.e., species whose MD populations are sink-dependent on source populations
    in VA, PA, WV, or DE.

    These are the species that require multi-state policy coordination and
    make the strongest case for interstate biodiversity compacts.
    """
    if not species_flags_path.exists():
        print("  species_sink_flags.csv not found — skipping cross-state dependencies report")
        return pd.DataFrame()

    flags = pd.read_csv(species_flags_path)

    if "is_sink_dependent" not in flags.columns:
        print("  No sink flag column found — skipping")
        return pd.DataFrame()

    sink_df = flags[flags["is_sink_dependent"]].copy()

    if sink_df.empty:
        print("  No sink-dependent species found")
        return pd.DataFrame()

    # Sort by how underrepresented they are in MD (lowest sink_ratio first =
    # most dependent on external sources)
    if "sink_ratio" in sink_df.columns:
        sink_df = sink_df.sort_values("sink_ratio")

    cols_out = [c for c in ["species", "sink_ratio", "md_count",
                            "regional_count", "n_states"] if c in sink_df.columns]
    report = sink_df[cols_out].reset_index(drop=True)
    report.index += 1  # 1-based rank

    output_path = OUTPUT_DIR / "cross_state_dependencies.csv"
    report.to_csv(output_path)
    print(f"  Saved {len(report)} sink-dependent species → {output_path}")
    return report


# ---------------------------------------------------------------------------
# 3. Protection Gap by County
# ---------------------------------------------------------------------------

def build_protection_gap_by_county(grid_df):
    """
    For each Maryland county, compute:
      - total grid cells
      - high-priority cells (score > HIGH_PRIORITY_THRESHOLD)
      - % of high-priority cells that are unprotected
        (proxy: cells with score > threshold and zero rare/threatened species
         recorded — a rough signal of under-surveyed, potentially unprotected
         habitat; in production, replace with PAD-US overlap)
    """
    if "county" not in grid_df.columns:
        print("  No county column in grid — skipping protection gap report")
        return pd.DataFrame()

    high_priority = grid_df[grid_df["priority_score"] >= HIGH_PRIORITY_THRESHOLD].copy()

    total_cells   = grid_df.groupby("county").size().rename("total_cells")
    hp_cells      = high_priority.groupby("county").size().rename("high_priority_cells")

    summary = pd.concat([total_cells, hp_cells], axis=1).fillna(0)
    summary["high_priority_cells"] = summary["high_priority_cells"].astype(int)
    summary["pct_high_priority"] = (
        summary["high_priority_cells"] / summary["total_cells"] * 100
    ).round(1)

    summary = summary.sort_values("high_priority_cells", ascending=False)
    summary = summary.reset_index().rename(columns={"index": "county"})

    output_path = OUTPUT_DIR / "protection_gap_by_county.csv"
    summary.to_csv(output_path, index=False)
    print(f"  Saved county protection gap → {output_path}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grid_path  = DATA_DIR / "maryland_priority_grid.csv"
    top_path   = DATA_DIR / "top_priority_regions.csv"
    flags_path = DATA_DIR / "species_sink_flags.csv"

    if not grid_path.exists():
        print("ERROR: maryland_priority_grid.csv not found.")
        print("Run 05_priority_score.py first.")
        raise SystemExit(1)

    print("Loading priority grid...")
    grid_df = pd.read_csv(grid_path)
    top_df  = pd.read_csv(top_path) if top_path.exists() else grid_df.nlargest(50, "priority_score")
    print(f"  {len(grid_df):,} grid cells loaded")

    print("\n--- Generating Interactive Priority Map ---")
    create_priority_map(grid_df, top_df)

    print("\n--- Building Cross-State Dependencies Report ---")
    dep_report = build_cross_state_dependencies(flags_path)
    if not dep_report.empty:
        print(f"  Top 5 most sink-dependent species:")
        for _, row in dep_report.head(5).iterrows():
            print(f"    {row.get('species', 'Unknown'):40s}  sink_ratio={row.get('sink_ratio', 0):.2f}")

    print("\n--- Building County Protection Gap Summary ---")
    gap_report = build_protection_gap_by_county(grid_df)
    if not gap_report.empty:
        print(f"\n  Counties with most high-priority unprotected habitat:")
        for _, row in gap_report.head(5).iterrows():
            print(
                f"    {row['county']:20s}  {row['high_priority_cells']:3d} high-priority cells "
                f"({row['pct_high_priority']:.1f}% of county area)"
            )

    print("\n--- Summary for Policymakers ---")
    total_hp = (grid_df["priority_score"] >= HIGH_PRIORITY_THRESHOLD).sum()
    total_cells = len(grid_df)
    print(f"  Total MD grid cells scored:     {total_cells:,}")
    print(f"  High-priority cells (≥{HIGH_PRIORITY_THRESHOLD}):    {total_hp:,} ({total_hp/total_cells*100:.1f}%)")
    print(f"  Max priority score:             {grid_df['priority_score'].max():.1f}")
    if not dep_report.empty:
        print(f"  Species requiring multi-state coordination: {len(dep_report)}")

    print("\n✅ Policy report generation complete!")
    print("   Open data/maryland_priority_map.html in a browser to explore results.")

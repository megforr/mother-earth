"""
Biodiversity Project — Step 2: Visualize Biodiversity Hotspots
==============================================================
Loads species occurrence data and creates a heatmap of biodiversity
hotspots across MD, VA, and PA.

Run 01_fetch_species_data.py first to generate the CSV files.

Dependencies: pip install pandas geopandas matplotlib folium seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "data"


def load_all_occurrences():
    """Load and combine all species occurrence CSVs."""
    dfs = []
    for csv_file in DATA_DIR.glob("*.csv"):
        if "richness_grid" not in csv_file.name:
            try:
                df = pd.read_csv(csv_file)
                df["source_file"] = csv_file.stem
                dfs.append(df)
                print(f"  Loaded {len(df):,} records from {csv_file.name}")
            except Exception as e:
                print(f"  Error loading {csv_file.name}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def plot_species_heatmap(df, title="Biodiversity Occurrences — MD, VA, PA"):
    """
    Create a 2D heatmap of species occurrence density.
    """
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])
    df = df[
        (df["decimalLatitude"].between(37.0, 42.5)) &
        (df["decimalLongitude"].between(-80.5, -74.5))
    ]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Hexbin heatmap
    hb = ax.hexbin(
        df["decimalLongitude"],
        df["decimalLatitude"],
        gridsize=50,
        cmap="YlOrRd",
        mincnt=1,
        bins="log"
    )
    cb = fig.colorbar(hb, ax=ax, label="Log10(Occurrence count)")

    # Annotate key cities
    cities = {
        "Annapolis, MD": (-76.49, 38.97),
        "Baltimore, MD": (-76.61, 39.29),
        "Washington, DC": (-77.04, 38.91),
        "Richmond, VA": (-77.46, 37.54),
        "Philadelphia, PA": (-75.16, 39.95),
        "Pittsburgh, PA": (-80.00, 40.44),
    }
    for city, (lon, lat) in cities.items():
        ax.plot(lon, lat, "b^", markersize=6, zorder=5)
        ax.annotate(city, (lon, lat), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="navy")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(-80.5, -74.5)
    ax.set_ylim(37.0, 42.5)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "biodiversity_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved heatmap → {output_path}")
    plt.show()
    return fig


def plot_species_richness_by_state(df):
    """Bar chart of unique species per state."""
    if "stateProvince" not in df.columns or "species" not in df.columns:
        print("  Missing stateProvince or species columns — skipping state chart")
        return

    state_map = {
        "Maryland": "Maryland",
        "Virginia": "Virginia",
        "Pennsylvania": "Pennsylvania",
        "MD": "Maryland", "VA": "Virginia", "PA": "Pennsylvania"
    }
    df["state_clean"] = df["stateProvince"].map(state_map)
    target_states = df[df["state_clean"].isin(["Maryland", "Virginia", "Pennsylvania"])]

    richness = (
        target_states.groupby("state_clean")["species"]
        .nunique()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2d6a4f", "#40916c", "#52b788"]
    bars = ax.bar(richness.index, richness.values, color=colors)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=11)
    ax.set_ylabel("Unique Species (from GBIF sample)")
    ax.set_title("Species Richness by State (Sample)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, richness.max() * 1.15)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "species_richness_by_state.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved state chart → {output_path}")
    plt.show()


def create_interactive_map(df):
    """
    Create an interactive Folium map of occurrences.
    Saves as an HTML file you can open in any browser.
    """
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("  folium not installed. Run: pip install folium")
        return

    df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])
    df = df[
        (df["decimalLatitude"].between(37.0, 42.5)) &
        (df["decimalLongitude"].between(-80.5, -74.5))
    ]

    # Center on Annapolis
    m = folium.Map(location=[38.97, -76.49], zoom_start=7, tiles="CartoDB positron")

    # Heatmap layer
    heat_data = df[["decimalLatitude", "decimalLongitude"]].values.tolist()
    HeatMap(heat_data, radius=8, blur=10, max_zoom=13).add_to(m)

    # Mark Annapolis
    folium.Marker(
        [38.97, -76.49],
        popup="Annapolis, MD (Your Base)",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)

    output_path = OUTPUT_DIR / "biodiversity_interactive_map.html"
    m.save(str(output_path))
    print(f"  Saved interactive map → {output_path}")
    print("  Open biodiversity_interactive_map.html in your browser!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading occurrence data...")
    df = load_all_occurrences()

    if df.empty:
        print("No data found. Run 01_fetch_species_data.py first.")
    else:
        print(f"\nTotal records loaded: {len(df):,}")
        print(f"Unique species: {df['species'].nunique() if 'species' in df.columns else 'N/A'}")

        print("\n=== Generating Heatmap ===")
        plot_species_heatmap(df)

        print("\n=== Generating State Richness Chart ===")
        plot_species_richness_by_state(df)

        print("\n=== Generating Interactive Map ===")
        create_interactive_map(df)

        print("\n✅ Visualization complete! Check the /data folder.")

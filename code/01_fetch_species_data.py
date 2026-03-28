"""
Biodiversity Project — Step 1: Fetch Regional Species Occurrence Data
======================================================================
Queries the GBIF API for species observations across Maryland, Virginia,
and Pennsylvania. Saves results as CSV for further analysis.

Dependencies: pip install requests pandas geopandas matplotlib
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Bounding box covering MD, VA, and PA (decimal degrees)
# [decimalLongitude_min, decimalLatitude_min, decimalLongitude_max, decimalLatitude_max]
BOUNDING_BOX = {
    "decimalLongitude_min": -80.5,
    "decimalLatitude_min":  37.0,
    "decimalLongitude_max": -74.5,
    "decimalLatitude_max":  42.5,
}

# GBIF API base URL
GBIF_BASE = "https://api.gbif.org/v1"

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Fetch Functions
# ---------------------------------------------------------------------------

def fetch_gbif_occurrences(taxon_key=None, kingdom=None, limit=300, max_records=5000):
    """
    Fetch species occurrence records from GBIF within the Mid-Atlantic bounding box.

    Args:
        taxon_key: Optional GBIF taxon key (e.g., 212 = Aves/Birds)
        kingdom:   Optional kingdom filter (e.g., 'Plantae', 'Animalia')
        limit:     Records per page (max 300)
        max_records: Maximum total records to fetch

    Returns:
        pd.DataFrame of occurrence records
    """
    all_records = []
    offset = 0
    total = None

    params = {
        "decimalLatitude":  f"{BOUNDING_BOX['decimalLatitude_min']},{BOUNDING_BOX['decimalLatitude_max']}",
        "decimalLongitude": f"{BOUNDING_BOX['decimalLongitude_min']},{BOUNDING_BOX['decimalLongitude_max']}",
        "hasCoordinate":    "true",
        "hasGeospatialIssue": "false",
        "limit":            limit,
        "offset":           offset,
    }
    if taxon_key:
        params["taxonKey"] = taxon_key
    if kingdom:
        params["kingdom"] = kingdom

    print(f"Fetching GBIF occurrences (kingdom={kingdom}, taxon_key={taxon_key})...")

    while True:
        params["offset"] = offset
        response = requests.get(f"{GBIF_BASE}/occurrence/search", params=params)
        response.raise_for_status()
        data = response.json()

        if total is None:
            total = data.get("count", 0)
            print(f"  Total available records: {total:,}")

        results = data.get("results", [])
        if not results:
            break

        all_records.extend(results)
        print(f"  Fetched {len(all_records):,} / {min(max_records, total):,}")

        if len(all_records) >= max_records or data.get("endOfRecords", True):
            break

        offset += limit
        time.sleep(0.2)  # Be polite to the API

    return pd.DataFrame(all_records)


def get_species_checklist_for_state(state_name):
    """
    Fetch a checklist of species observed in a specific US state via GBIF.
    Uses the GBIF Occurrence facet API to get unique species per state.

    Args:
        state_name: e.g., 'Maryland', 'Virginia', 'Pennsylvania'

    Returns:
        List of dicts with species info
    """
    # Map state names to GBIF stateProvince values
    params = {
        "stateProvince": state_name,
        "country": "US",
        "hasCoordinate": "true",
        "facet": "speciesKey",
        "facetLimit": 1000,
        "limit": 0,  # We only want facets, not records
    }
    response = requests.get(f"{GBIF_BASE}/occurrence/search", params=params)
    response.raise_for_status()
    data = response.json()

    facets = data.get("facets", [])
    species_keys = []
    for facet in facets:
        if facet.get("field") == "SPECIES_KEY":
            species_keys = [item["name"] for item in facet.get("counts", [])]
            break

    print(f"  {state_name}: {len(species_keys)} unique species found in GBIF")
    return species_keys


def fetch_species_richness_grid(cell_size_deg=0.5):
    """
    Create a grid of species richness counts across the region.
    Divides the bounding box into cells and counts unique species per cell.

    Args:
        cell_size_deg: Grid cell size in degrees (0.5 ≈ ~50km)

    Returns:
        pd.DataFrame with columns [lat, lon, species_count]
    """
    import numpy as np

    lats = list(range(
        int(BOUNDING_BOX["decimalLatitude_min"] * 2),
        int(BOUNDING_BOX["decimalLatitude_max"] * 2)
    ))
    lons = list(range(
        int(BOUNDING_BOX["decimalLongitude_min"] * 2),
        int(BOUNDING_BOX["decimalLongitude_max"] * 2)
    ))

    results = []
    total_cells = len(lats) * len(lons)
    cell_num = 0

    for lat_i in lats:
        for lon_i in lons:
            lat = lat_i / 2.0
            lon = lon_i / 2.0
            cell_num += 1

            params = {
                "decimalLatitude":  f"{lat},{lat + cell_size_deg}",
                "decimalLongitude": f"{lon},{lon + cell_size_deg}",
                "hasCoordinate":    "true",
                "hasGeospatialIssue": "false",
                "facet":            "speciesKey",
                "facetLimit":       1,
                "limit":            0,
            }
            try:
                response = requests.get(f"{GBIF_BASE}/occurrence/search", params=params)
                response.raise_for_status()
                data = response.json()
                count = data.get("count", 0)
                results.append({
                    "lat_center": lat + cell_size_deg / 2,
                    "lon_center": lon + cell_size_deg / 2,
                    "lat_min": lat,
                    "lon_min": lon,
                    "occurrence_count": count
                })
                if cell_num % 10 == 0:
                    print(f"  Grid cell {cell_num}/{total_cells}: ({lat:.1f}, {lon:.1f}) → {count:,} occurrences")
                time.sleep(0.1)
            except Exception as e:
                print(f"  Error at ({lat}, {lon}): {e}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- 1. Fetch bird occurrences (Aves = taxon key 212) ---
    print("\n=== Fetching Bird Occurrences ===")
    birds_df = fetch_gbif_occurrences(taxon_key=212, max_records=2000)
    if not birds_df.empty:
        cols_keep = [c for c in [
            "species", "scientificName", "decimalLatitude", "decimalLongitude",
            "stateProvince", "eventDate", "datasetName", "taxonRank",
            "kingdom", "phylum", "class", "order", "family", "genus"
        ] if c in birds_df.columns]
        birds_df[cols_keep].to_csv(OUTPUT_DIR / "birds_md_va_pa.csv", index=False)
        print(f"  Saved {len(birds_df):,} bird records → data/birds_md_va_pa.csv")

    # --- 2. Fetch plant occurrences ---
    print("\n=== Fetching Plant Occurrences ===")
    plants_df = fetch_gbif_occurrences(kingdom="Plantae", max_records=2000)
    if not plants_df.empty:
        cols_keep = [c for c in [
            "species", "scientificName", "decimalLatitude", "decimalLongitude",
            "stateProvince", "eventDate", "datasetName", "taxonRank",
            "kingdom", "phylum", "class", "order", "family", "genus"
        ] if c in plants_df.columns]
        plants_df[cols_keep].to_csv(OUTPUT_DIR / "plants_md_va_pa.csv", index=False)
        print(f"  Saved {len(plants_df):,} plant records → data/plants_md_va_pa.csv")

    # --- 3. State-level species summary ---
    print("\n=== State Species Counts ===")
    for state in ["Maryland", "Virginia", "Pennsylvania"]:
        keys = get_species_checklist_for_state(state)
        time.sleep(0.3)

    # --- 4. Build species richness grid (comment out if slow) ---
    # print("\n=== Building Species Richness Grid ===")
    # grid_df = fetch_species_richness_grid(cell_size_deg=0.5)
    # grid_df.to_csv(OUTPUT_DIR / "species_richness_grid.csv", index=False)
    # print(f"  Saved grid → data/species_richness_grid.csv")

    print("\n✅ Data collection complete! Check the /data folder.")

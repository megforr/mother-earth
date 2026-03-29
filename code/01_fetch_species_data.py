"""
Biodiversity Project — Step 1: Fetch Regional Species Occurrence Data
======================================================================
Queries the GBIF API for species observations across the 5-state Mid-Atlantic
region (Maryland, Virginia, Pennsylvania, West Virginia, Delaware).

MD/VA/PA were the original scope; WV and DE are added because Maryland shares
critical ecological corridors with both:
  - WV/WVa: Appalachian Mountain spine through MD's western panhandle
  - DE: Delmarva Peninsula coastal plain (a global biodiversity hotspot)

Outputs:
  data/regional_all_taxa.csv   — all 5 states, all taxa
  data/maryland_all_taxa.csv   — Maryland subset (used for priority scoring)

Dependencies: pip install requests pandas
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Bounding box covering MD, VA, PA, WV, and DE (decimal degrees)
# Expanded west to -81.0 to capture WV panhandle; north to 42.5 for PA
BOUNDING_BOX = {
    "decimalLongitude_min": -81.0,
    "decimalLatitude_min":  37.0,
    "decimalLongitude_max": -74.5,
    "decimalLatitude_max":  42.5,
}

# Maryland-only bounding box for the MD subset
MARYLAND_BBOX = {
    "decimalLongitude_min": -79.5,
    "decimalLatitude_min":  37.88,
    "decimalLongitude_max": -74.97,
    "decimalLatitude_max":  39.72,
}

# GBIF taxon keys for target taxonomic groups
# Each (taxon_key, label) pair maps to a GBIF higher taxon
TAXA = [
    (212,    "Aves"),          # Birds
    (194,    "Reptilia"),      # Reptiles
    (131,    "Amphibia"),      # Amphibians
    (216,    "Insecta"),       # Insects (pollinators, etc.)
    (204,    "Actinopterygii"),# Freshwater fish
    (None,   "Plantae"),       # Plants (filtered by kingdom)
]

# GBIF API base URL
GBIF_BASE = "https://api.gbif.org/v1"

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# IUCN threatened categories to flag as rare/at-risk
IUCN_THREATENED = ["EN", "VU", "CR"]  # Endangered, Vulnerable, Critically Endangered


# ---------------------------------------------------------------------------
# Fetch Functions
# ---------------------------------------------------------------------------

def fetch_gbif_occurrences(taxon_key=None, kingdom=None, bbox=None, limit=300, max_records=5000):
    """
    Fetch species occurrence records from GBIF within a bounding box.

    Args:
        taxon_key:   Optional GBIF taxon key (e.g., 212 = Aves/Birds)
        kingdom:     Optional kingdom filter (e.g., 'Plantae', 'Animalia')
        bbox:        Dict with decimalLatitude_min/max, decimalLongitude_min/max.
                     Defaults to the full 5-state BOUNDING_BOX.
        limit:       Records per page (max 300)
        max_records: Maximum total records to fetch

    Returns:
        pd.DataFrame of occurrence records
    """
    if bbox is None:
        bbox = BOUNDING_BOX

    all_records = []
    offset = 0
    total = None

    params = {
        "decimalLatitude":    f"{bbox['decimalLatitude_min']},{bbox['decimalLatitude_max']}",
        "decimalLongitude":   f"{bbox['decimalLongitude_min']},{bbox['decimalLongitude_max']}",
        "hasCoordinate":      "true",
        "hasGeospatialIssue": "false",
        "limit":              limit,
        "offset":             offset,
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


def fetch_regional_occurrences(max_records_per_taxon=2000):
    """
    Fetch occurrences for all target taxa across the full 5-state region
    (MD, VA, PA, WV, DE).

    Returns:
        pd.DataFrame of combined occurrence records with a 'taxon_group' column
    """
    cols_keep = [
        "species", "scientificName", "decimalLatitude", "decimalLongitude",
        "stateProvince", "eventDate", "month", "year", "datasetName",
        "taxonRank", "kingdom", "phylum", "class", "order", "family", "genus",
        "iucnRedListCategory",
    ]

    all_dfs = []
    for taxon_key, label in TAXA:
        print(f"\n=== Fetching {label} ===")
        kwargs = {"max_records": max_records_per_taxon}
        if taxon_key:
            kwargs["taxon_key"] = taxon_key
        else:
            kwargs["kingdom"] = label  # e.g. "Plantae"

        df = fetch_gbif_occurrences(**kwargs)
        if not df.empty:
            df["taxon_group"] = label
            available = [c for c in cols_keep if c in df.columns] + ["taxon_group"]
            all_dfs.append(df[available])
            print(f"  → {len(df):,} {label} records")
        time.sleep(0.3)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined regional dataset: {len(combined):,} records")
    return combined


def fetch_maryland_occurrences(max_records_per_taxon=2000):
    """
    Fetch occurrences for all target taxa restricted to the Maryland bounding box.
    This is used as the fine-grained input for the priority scoring model.

    Returns:
        pd.DataFrame restricted to Maryland coordinates
    """
    cols_keep = [
        "species", "scientificName", "decimalLatitude", "decimalLongitude",
        "stateProvince", "eventDate", "month", "year", "datasetName",
        "taxonRank", "kingdom", "phylum", "class", "order", "family", "genus",
        "iucnRedListCategory",
    ]

    all_dfs = []
    for taxon_key, label in TAXA:
        print(f"\n=== Fetching MD {label} ===")
        kwargs = {"bbox": MARYLAND_BBOX, "max_records": max_records_per_taxon}
        if taxon_key:
            kwargs["taxon_key"] = taxon_key
        else:
            kwargs["kingdom"] = label

        df = fetch_gbif_occurrences(**kwargs)
        if not df.empty:
            df["taxon_group"] = label
            available = [c for c in cols_keep if c in df.columns] + ["taxon_group"]
            all_dfs.append(df[available])
            print(f"  → {len(df):,} MD {label} records")
        time.sleep(0.3)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nMaryland dataset: {len(combined):,} records")
    return combined


def fetch_rare_species(bbox=None, max_records=3000):
    """
    Fetch occurrence records for IUCN-threatened species (EN/VU/CR) in the region.
    These records get extra weight in the priority scoring model.

    Args:
        bbox: Bounding box dict; defaults to full 5-state region
        max_records: Maximum records to fetch

    Returns:
        pd.DataFrame filtered to threatened species
    """
    if bbox is None:
        bbox = BOUNDING_BOX

    print("\n=== Fetching IUCN-Threatened Species ===")
    all_records = []

    for category in IUCN_THREATENED:
        params = {
            "decimalLatitude":    f"{bbox['decimalLatitude_min']},{bbox['decimalLatitude_max']}",
            "decimalLongitude":   f"{bbox['decimalLongitude_min']},{bbox['decimalLongitude_max']}",
            "hasCoordinate":      "true",
            "hasGeospatialIssue": "false",
            "iucnRedListCategory": category,
            "limit":              300,
            "offset":             0,
        }
        fetched = 0
        while fetched < max_records // len(IUCN_THREATENED):
            response = requests.get(f"{GBIF_BASE}/occurrence/search", params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results:
                break
            all_records.extend(results)
            fetched += len(results)
            if data.get("endOfRecords", True):
                break
            params["offset"] += 300
            time.sleep(0.2)
        print(f"  IUCN {category}: {fetched} records")
        time.sleep(0.3)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    cols_keep = [
        "species", "scientificName", "decimalLatitude", "decimalLongitude",
        "stateProvince", "eventDate", "month", "year",
        "kingdom", "class", "family", "iucnRedListCategory",
    ]
    available = [c for c in cols_keep if c in df.columns]
    return df[available].drop_duplicates()


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

    # --- 1. Full 5-state regional dataset (all taxa) ---
    print("\n=== Fetching Full Regional Dataset (MD, VA, PA, WV, DE) ===")
    regional_df = fetch_regional_occurrences(max_records_per_taxon=2000)
    if not regional_df.empty:
        regional_df.to_csv(OUTPUT_DIR / "regional_all_taxa.csv", index=False)
        print(f"  Saved {len(regional_df):,} records → data/regional_all_taxa.csv")

    # --- 2. Maryland-only dataset (all taxa, finer bounding box) ---
    print("\n=== Fetching Maryland-Only Dataset ===")
    maryland_df = fetch_maryland_occurrences(max_records_per_taxon=2000)
    if not maryland_df.empty:
        maryland_df.to_csv(OUTPUT_DIR / "maryland_all_taxa.csv", index=False)
        print(f"  Saved {len(maryland_df):,} records → data/maryland_all_taxa.csv")

    # --- 3. IUCN-threatened species across the full region ---
    print("\n=== Fetching Rare/Threatened Species ===")
    rare_df = fetch_rare_species()
    if not rare_df.empty:
        rare_df.to_csv(OUTPUT_DIR / "rare_species_regional.csv", index=False)
        print(f"  Saved {len(rare_df):,} threatened species records → data/rare_species_regional.csv")

    # --- 4. State-level species summary ---
    print("\n=== State Species Counts ===")
    for state in ["Maryland", "Virginia", "Pennsylvania", "West Virginia", "Delaware"]:
        keys = get_species_checklist_for_state(state)
        time.sleep(0.3)

    # --- 5. Build species richness grid (comment out if slow) ---
    # print("\n=== Building Species Richness Grid ===")
    # grid_df = fetch_species_richness_grid(cell_size_deg=0.5)
    # grid_df.to_csv(OUTPUT_DIR / "species_richness_grid.csv", index=False)
    # print(f"  Saved grid → data/species_richness_grid.csv")

    print("\n✅ Data collection complete! Check the /data folder.")

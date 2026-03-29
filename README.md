# Biodiversity & Conservation Project
**Base:** Annapolis, MD | **Region:** Maryland · Virginia · Pennsylvania · West Virginia · Delaware

## Goal
Use open-source AI models (Google DeepMind) and open biodiversity data to identify conservation
priority areas across the 5-state Mid-Atlantic region, then produce policy-facing outputs to help
Maryland decision-makers protect the regions that maximize biodiversity outcomes.

A key design principle: **Maryland protection decisions are evaluated in their regional context.**
Many MD populations are ecological sinks sustained by source populations in adjacent states, and
migratory species depend on corridor integrity across the full Mid-Atlantic. The pipeline accounts
for these cross-state dependencies before ranking sites for protection.

## Folder Structure
```
mother-earth/
├── README.md
├── research/
│   └── project-research-overview.md   ← Research background, hotspots, DeepMind models
├── code/
│   ├── 01_fetch_species_data.py       ← Fetch GBIF data: 5 states, 6 taxa, IUCN-threatened
│   ├── 02_hotspot_map.py              ← Heatmaps, interactive maps, priority choropleth
│   ├── 03_perch_setup.py              ← Perch 2.0 bioacoustics model (field validation)
│   ├── 04_regional_connectivity.py   ← Cross-state connectivity analysis
│   ├── 05_priority_score.py          ← Composite conservation priority scoring
│   └── 06_policy_report.py           ← Policy-facing outputs (map, reports, gap analysis)
├── eda/
│   └── 01_exploratory_analysis.ipynb  ← Exploratory data analysis notebook
└── data/                              ← Generated files (gitignored)
    └── audio/                         ← Place AudioMoth .wav files here
```

## Pipeline Overview

The pipeline runs in six steps:

```
01 → fetch data        →  regional_all_taxa.csv
                           maryland_all_taxa.csv
                           rare_species_regional.csv

04 → connectivity      →  regional_connectivity.csv
                           species_sink_flags.csv

05 → priority score    →  maryland_priority_grid.csv
                           top_priority_regions.csv

06 → policy report     →  maryland_priority_map.html   ← open in browser
                           cross_state_dependencies.csv
                           protection_gap_by_county.csv

02 → visualize         →  maryland_priority_choropleth.png
                           biodiversity_heatmap.png

03 → field validation  →  perch_detections.csv   (optional, requires AudioMoth recordings)
```

## Quick Start

### 1. Install dependencies
```bash
pip install requests pandas numpy networkx folium matplotlib seaborn
# For bioacoustics (step 03, optional):
pip install librosa tensorflow kagglehub
```

### 2. Fetch species data across the 5-state region
```bash
python code/01_fetch_species_data.py
```
Queries GBIF for birds, reptiles, amphibians, insects, freshwater fish, and plants across
MD, VA, PA, WV, and DE. Also fetches IUCN-threatened species records.

### 3. Compute cross-state connectivity metrics
```bash
python code/04_regional_connectivity.py
```
Runs three analyses:
- **Source-sink flagging** — identifies species whose Maryland populations depend on source
  populations in adjacent states
- **Migration bottleneck scoring** — identifies stopover sites with high migrant concentrations
- **Corridor centrality** — finds MD grid cells that are pinch-points in the 5-state habitat graph

### 4. Score Maryland grid cells
```bash
python code/05_priority_score.py
```
Produces a composite conservation priority score (0–100) per ~5km grid cell using:

| Component | Weight | What it captures |
|---|---|---|
| Rare/threatened species | 25% | IUCN EN/VU/CR records per cell |
| Species richness | 20% | Unique species across all taxa |
| Sink-dependent species | 15% | Species needing multi-state coordination |
| Migration bottleneck | 15% | Stopover importance for migrants |
| Corridor centrality | 15% | Cross-state corridor pinch-point value |
| Development threat | 10% | Risk of habitat loss |

### 5. Generate policy outputs
```bash
python code/06_policy_report.py
```
Produces:
- **`data/maryland_priority_map.html`** — Interactive map; open in any browser
- **`data/cross_state_dependencies.csv`** — Species requiring interstate coordination
- **`data/protection_gap_by_county.csv`** — High-priority unprotected habitat by county

### 6. Generate static visualizations
```bash
python code/02_hotspot_map.py
```
Produces static PNG heatmaps and the priority choropleth (requires step 5 output).

### 7. Field validation with bioacoustics (optional)
```bash
python code/03_perch_setup.py
```
Runs Google DeepMind's Perch 2.0 model on AudioMoth `.wav` recordings placed in `data/audio/`.
Use this to ground-truth top-ranked cells and detect species missed in citizen-science databases.

## Data Sources

| Source | Type | Taxa | Use |
|---|---|---|---|
| GBIF | API | All | Primary occurrence records |
| iNaturalist | via GBIF | All | Citizen science observations |
| eBird (Cornell) | Bulk download | Birds | Migration & stopover data |
| USGS PAD-US | Download | — | Existing protected areas (connectivity) |
| USGS NLCD | Download | — | Land cover / development threat |
| MD DNR Natural Heritage | Download | Rare species | Ground-truth rare species layer |

## External Data Downloads (one-time, recommended)

For production-quality scoring, download these free government datasets and place them in `data/`:

- **PAD-US (5 states)** — USGS Protected Areas Database: https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-download
- **NLCD Land Cover** — USGS: https://www.mrlc.gov/data
- **MD County Boundaries** — US Census TIGER: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

## Key Resources
- **Research overview:** `research/project-research-overview.md`
- **Perch model (DeepMind):** https://github.com/google-research/perch
- **GBIF data:** https://www.gbif.org/
- **Chesapeake Conservancy:** https://www.chesapeakeconservancy.org/

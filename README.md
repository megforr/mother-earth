# Biodiversity & Conservation Project
**Base:** Annapolis, MD | **Region:** Maryland · Virginia · Pennsylvania

## Goal
Use open-source AI models (Google DeepMind) and open biodiversity data to identify
hotspots across the Mid-Atlantic, then guide conservation actions to maximize biodiversity.

## Folder Structure
```
biodiversity-conservation-project/
├── README.md                        ← You are here
├── research/
│   └── project-research-overview.md ← Full research summary, hotspots, DeepMind models
├── code/
│   ├── 01_fetch_species_data.py     ← Query GBIF API for species occurrences
│   ├── 02_hotspot_map.py            ← Generate heatmaps and interactive maps
│   └── 03_perch_setup.py           ← Set up Perch 2.0 bioacoustics model
└── data/                            ← Generated data files go here
    └── audio/                       ← Place AudioMoth .wav files here
```

## Quick Start

### 1. Install dependencies
```bash
pip install requests pandas geopandas matplotlib folium seaborn librosa tensorflow kagglehub
```

### 2. Fetch regional species data
```bash
python code/01_fetch_species_data.py
```

### 3. Visualize hotspots
```bash
python code/02_hotspot_map.py
```

### 4. Set up bioacoustic monitoring
```bash
python code/03_perch_setup.py
```

## Key Resources
- **Research overview:** `research/project-research-overview.md`
- **Perch model (DeepMind):** https://github.com/google-research/perch
- **GBIF data:** https://www.gbif.org/
- **Chesapeake Conservancy:** https://www.chesapeakeconservancy.org/

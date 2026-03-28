# Biodiversity & Conservation Project — Research Overview
**Location Focus:** Annapolis, MD · Maryland · Virginia · Pennsylvania
**Goal:** Identify biodiversity hotspots and maximize biodiversity using AI/ML tools
**Date:** March 2026

---

## 1. Project Vision

This project aims to use open-source AI models — particularly from Google DeepMind — to map and analyze biodiversity hotspots across the Mid-Atlantic region (MD, VA, PA), with a local focus on the Chesapeake Bay watershed. The ultimate goal is to use those insights to guide conservation actions that maximize biodiversity outcomes.

---

## 2. Google DeepMind Open Source Models for Biodiversity

### 2a. Perch 2.0 — Bioacoustics & Species Identification
- **What it is:** A foundational AI model for classifying animal vocalizations (birds, mammals, amphibians, and more).
- **Coverage:** Trained on nearly **15,000 species**, making it one of the broadest bioacoustic models available.
- **Training data:** Public sources including Xeno-Canto (bird sounds) and iNaturalist audio.
- **Key capability:** Allows field ecologists to fine-tune the model on new species and habitats with minimal local data (transfer learning).
- **Open source:** Available on [GitHub (google-research/perch)](https://github.com/google-research/perch) and [Kaggle Models](https://www.kaggle.com/models/google/bird-vocalization-classifier); 250,000+ downloads since 2023.
- **Local relevance:** Ideal for monitoring bird species, frogs, and other vocal wildlife across the Chesapeake watershed, Appalachian forests, and Atlantic coastal habitats.
- **Example use case:** Deploy passive audio recorders at candidate conservation sites around Annapolis and run Perch to identify which species are present — including rare or endangered ones.

### 2b. Graph Neural Network (GNN) Species Range Maps
- **What it is:** A GNN model that combines:
  - Open field observation databases (iNaturalist, GBIF, eBird)
  - Satellite image embeddings from **AlphaEarth Foundations**
  - Species trait data (body mass, habitat preferences)
- **Output:** High-resolution range maps for thousands of species simultaneously.
- **Release:** 23 species range maps already released via the [UN Biodiversity Lab](https://unbiodiversitylab.org/) and Google Earth Engine.
- **Local relevance:** Can be used to model which species currently exist (or historically existed) in your target region and identify where habitat restoration would have maximum impact.

### 2c. AlphaEarth Foundations (Satellite Embeddings)
- **What it is:** A satellite imagery foundation model from Google DeepMind that produces rich embeddings of land surface conditions.
- **Use:** Powers the GNN species range model; can also be used independently for land cover classification, habitat quality assessment, and change detection.
- **Local relevance:** Map habitat types across MD/VA/PA at fine resolution; track habitat loss and fragmentation over time.

---

## 3. Biodiversity Hotspots in MD, VA, and PA

### 3a. North American Coastal Plain (NACP)
- **One of only two U.S. biodiversity hotspots** recognized globally by Conservation International.
- Covers much of the Chesapeake Bay watershed, including the counties around Annapolis.
- Contains **1,800 endemic plant species**, 51 endemic bird species, 114 endemic mammal species.
- **Your backyard is literally inside a global biodiversity hotspot.**

### 3b. Chesapeake Bay Watershed (Your Local Area)
- Supports **3,600+ species** of plants and animals, including 300+ fish species and 2,700 plant types.
- Key taxa: blue crab, oysters, striped bass, freshwater mussels, migratory waterfowl.
- **Threat:** ~100 acres of forest habitat lost per day to development.
- Conservation resources near Annapolis:
  - **Chesapeake Bay Foundation HQ (Annapolis):** 33 acres with native plant gardens, wetlands, reforestation zones, and rain gardens.
  - **Chesapeake Ecology Center:** 10 acres along College Creek with 24 Native Plant Demonstration Gardens.

### 3c. Appalachian Forests (Western MD + PA + VA)
- Western Maryland sits within the central Appalachian range — a **top conservation priority** linking large conservation lands in WV (south) and PA (north).
- Salamander biodiversity capital of the world — highest density of salamander species anywhere on Earth.
- Contains **Mid-Appalachian Shale Barrens**: 18 plant species found *nowhere else on Earth* outside this narrow corridor (SW Pennsylvania → Maryland → WV → W. Virginia).

### 3d. Central Appalachians — Virginia
- High plant biodiversity due to mixed mesophytic forests.
- **Threatened** by coal mining and development in parts of SW Virginia.
- Virginia Natural Heritage Program has developed GIS tools and a conservation sites layer for rare species habitats.

### 3e. Freshwater Mussel Hotspots — Virginia & Chesapeake Watershed
- Virginia is a global center of freshwater mussel biodiversity — one of the most imperiled groups in North America.
- Mussels are critical water filterers; protecting mussel habitat = protecting water quality for everything downstream.
- New mussel hotspot mapping projects underway via the Chesapeake Bay Foundation.

---

## 4. Open Data Sources

| Platform | Data Type | API Available | Notes |
|---|---|---|---|
| [GBIF](https://www.gbif.org/) | All taxa species occurrences | ✅ Yes | Best single source; aggregates iNaturalist, eBird, museum records |
| [iNaturalist](https://www.inaturalist.org/) | Citizen science observations (all taxa) | ✅ Yes | 60-100 req/min; research-grade records exported to GBIF |
| [eBird (Cornell Lab)](https://ebird.org/) | Bird observations | ✅ Yes | Best for birds; downloadable bulk data + API |
| [USGS BioData](https://biodata.usgs.gov/) | Aquatic species & habitat | ✅ Partial | Strong for streams and freshwater |
| [MD DNR Natural Heritage Program](https://dnr.maryland.gov/wildlife/Pages/Plants_Wildlife/naturalheritage.aspx) | MD rare/threatened species | ❌ Download only | GIS shapefiles downloadable |
| [VA Natural Heritage Program](http://www.naturalheritage.dcr.virginia.gov/) | VA rare/threatened species | ❌ Download only | Conservation sites layer available |
| [PA Natural Heritage Program](https://www.naturalheritage.state.pa.us/) | PA communities & rare species | ❌ Download only | Community reference data available |
| [Chesapeake Bay Program](https://www.chesapeakebay.net/what/data) | Bay-wide environmental data | ✅ Yes | Water quality, habitat, species monitoring |
| [Xeno-Canto](https://xeno-canto.org/) | Bird vocalizations (audio) | ✅ Yes | Training data for Perch |

---

## 5. Recommended Project Phases

### Phase 1 — Data Collection & Baseline Mapping (Months 1–2)
1. Query GBIF API for all research-grade species observations in MD, VA, PA (lat/lon bounding box).
2. Download eBird data for the region.
3. Download state Natural Heritage Program GIS layers for rare/threatened species.
4. Create a unified species occurrence geodatabase.

### Phase 2 — Hotspot Analysis (Months 2–3)
1. Use species richness mapping to identify areas of highest biodiversity.
2. Apply threat layer (land use change, development pressure) to identify **threatened hotspots**.
3. Use AlphaEarth/satellite imagery to assess habitat quality and fragmentation.
4. Run the DeepMind GNN species range model to predict distribution of underobserved species.

### Phase 3 — Acoustic Monitoring (Ongoing)
1. Deploy passive audio recorders (e.g., AudioMoth devices — ~$70 each) at target sites.
2. Run Perch 2.0 inference on audio recordings to build species inventories.
3. Compare acoustic diversity across sites to rank conservation priority.

### Phase 4 — Conservation Action Recommendations
1. Identify parcels for protection or restoration with highest biodiversity ROI.
2. Prioritize native plant restoration (especially for pollinators and migratory birds).
3. Engage with local land trusts, the Chesapeake Conservancy, and state Natural Heritage Programs.

---

## 6. Strategies to Maximize Biodiversity

### Native Plant Restoration
- Native plants support ~35x more insect species than non-native alternatives (Doug Tallamy research).
- Priority species for Annapolis area: oaks (Quercus spp.), native willows, buttonbush, wild bergamot, native grasses.
- Removing invasive species (English ivy, Japanese honeysuckle, multiflora rose, garlic mustard) is often the highest-ROI action.

### Habitat Connectivity
- Fragmentation is one of the top drivers of biodiversity loss in the Mid-Atlantic.
- Focus restoration on linking existing protected areas (greenways, riparian corridors).
- The Chesapeake Conservancy's **30x30 objective** aims to protect 30% of land/water by 2030 — align your work with this framework.

### Riparian Buffers
- Restoring forested buffers along streams dramatically increases biodiversity (aquatic + terrestrial).
- Critical for freshwater mussel and salamander habitat.

### Dark Sky / Light Pollution Reduction
- Artificial light at night disrupts migratory birds and insects — reducing light pollution is a surprisingly high-impact action.

### Citizen Science Integration
- Organize iNaturalist BioBlitz events locally to fill data gaps.
- Partner with local birding groups (Anne Arundel Bird Club) for eBird data collection.

---

## 7. Key Organizations to Connect With

| Organization | Focus | Location |
|---|---|---|
| [Chesapeake Conservancy](https://www.chesapeakeconservancy.org/) | Bay-wide land & water conservation | Annapolis, MD |
| [Chesapeake Bay Foundation](https://www.cbf.org/) | Restoration, advocacy, education | Annapolis, MD |
| [MD DNR Wildlife & Heritage Service](https://dnr.maryland.gov/wildlife/Pages/default.aspx) | State wildlife mgmt | MD |
| [The Nature Conservancy MD/DC](https://www.nature.org/en-us/about-us/where-we-work/united-states/maryland-dc/) | Appalachian + Bay habitats | MD/DC |
| [VA Natural Heritage Program](http://www.dcr.virginia.gov/natural-heritage/) | Rare species & habitat tracking | VA |
| [PA Natural Heritage Program](https://www.naturalheritage.state.pa.us/) | Communities & rare species | PA |
| [Alliance for the Chesapeake Bay](https://www.allianceforthebay.org/) | Watershed-wide restoration | Regional |

---

## 8. Starter Code Sketch

See `/code/` folder for Python notebooks to:
- Query GBIF API for regional species data
- Visualize species richness maps
- Set up Perch 2.0 inference pipeline

---

## Sources
- [Google DeepMind — Mapping, Modeling and Understanding Nature with AI](https://deepmind.google/blog/mapping-modeling-and-understanding-nature-with-ai/)
- [GitHub: google-research/perch](https://github.com/google-research/perch)
- [Perch 2.0 — HackerNoon Overview](https://hackernoon.com/perch-20-bioacoustics-model-for-species-identification)
- [Heterogenous GNN for Species Distribution Modeling (arXiv)](https://arxiv.org/html/2503.11900v1)
- [GBIF](https://www.gbif.org/)
- [Maryland DNR — Native Plants for Wildlife in the Chesapeake Bay Watershed](https://www.fws.gov/media/native-plants-wildlife-habitat-and-conservation-landscaping-chesapeake-bay-watershed)
- [The Nature Conservancy — Maryland & DC Appalachian Forests](https://www.nature.org/en-us/about-us/where-we-work/united-states/maryland-dc/appalachian-forests/)
- [Chesapeake Conservancy — 30x30 Objective](https://www.chesapeakeconservancy.org/about-us/30-by-30-objective)
- [Alliance for the Chesapeake Bay — Wildlife of Our Watershed](https://www.allianceforthebay.org/the-wildlife-of-our-watershed/)
- [Chesapeake Bay Foundation — Freshwater Mussel Hotspot Map](https://www.cbf.org/news-media/newsroom/2024/virginia/new-map-of-mussel-hotspots-created.html)
- [Virginia Biodiversity Assessment — LandScope America](http://www.landscope.org/virginia/map_layers/priorities/biodiversity_assessment/15617/)

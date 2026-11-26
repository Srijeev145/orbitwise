# OrbitWise: Flood to Payout

AI based flood system and collision model.

**Live site (GitHub Pages):** https://srijeev145.github.io/orbitwise/ · [Mission Control](https://srijeev145.github.io/orbitwise/mission-control/) · [Interactive demo](https://srijeev145.github.io/orbitwise/demo/)

Boeing hackathon project: one system that covers both tracks.
- **Track A (Read the Planet):** map floods with satellite radar and turn them into crop-insurance payouts.
- **Track B (Build the Orbit):** protect the radar satellites Track A depends on. It predicts collision risk 2 days early and screens this week's close approaches with debris.

## Run it (one command, ~3 minutes, no internet)

```bash
pip install -r requirements.txt
python orbitwise.py
```

Then open `results/report.md` (every number), `results/kerala_flood_and_payout.png` (the map figure) and `results/charts/` (bar charts for the slides).

To view the websites locally: `python -m http.server 8000` in this folder, then open http://localhost:8000 (double-clicking the HTML files will not load their data).

The system runs on **pieces of the real datasets bundled in `data_sample/`** (40 MB).

## The five stages in `orbitwise.py`

| Stage | Function | Real data | Produces |
|---|---|---|---|
| 1 · Flood AI | `stage1_flood_ai` | 650,000 expert-labelled radar pixels (Sen1Floods11) | Trained Gradient Boosting model; IoU vs fixed and Otsu thresholds |
| 2 · Flood map | `stage2_flood_map` | Kerala 2018 Sentinel-1 before/after + ESA WorldCover | New water km², flooded farmland km² |
| 3 · Payout | `stage3_payout` | Stage 2 map + payout rule | Triggered ~1 km cells, hectares, ₹ (once sum insured is set) |
| 4 · Collision risk | `stage4_collision_risk` | 11,942 real ESA Kelvins events | AI vs "do nothing", 2 days before closest approach |
| 5 · Debris screen | `stage5_debris_screen` | Current orbits: 11 satellites vs 14,334 objects (2,685 debris + 11,649 active satellites, CelesTrak) | Close approaches in the next 72 h |

Settings are in the `CONFIG` block at the top of `orbitwise.py`. **Set `sum_insured_per_ha`** (PMFBY scale of finance, ₹/ha) to get the ₹ payout.

## Folders

| Path | What it is |
|---|---|
| `orbitwise.py` | The whole system |
| `data_sample/` | Bundled pieces of real data (see below) |
| `results/` | Output of the last run |
| `index.html`, `demo/`, `mission-control/` | The websites served by GitHub Pages: landing page, interactive demo, 3D mission control (regenerate demo data with `python tools/build_web_data.py`) |
| `tools/` | Download the full datasets (`fetch_s1.py`, `fetch_sen1floods11.sh`, `fetch_kelvins.sh`) and rebuild the samples (`make_samples.py`) |
| `notebook/Phase2_GradientBoosting.ipynb` | Colab notebook for stage 4 on its own (Phase II presentation) |
| `docs/` | Literature review, economics and validation figures, reference flood maps, PMFBY rules |
| `archive/` | Earlier separate scripts, Earth Engine scripts, and the dropped solar/wind idea |

## Bundled data (`data_sample/`)

| File | Contents | Source |
|---|---|---|
| `flood_pixels.npz` | 650k labelled pixels × 8 features (train / validation / test / Bolivia) | Sen1Floods11 |
| `KL2018_t165_before.tif`, `KL2018_t165_after_0821.tif` | Kerala radar, VV+VH in dB×100, ~55 m | Sentinel-1 RTC via Microsoft Planetary Computer |
| `KL2018_worldcover.npz` | Land use on the radar grid | ESA WorldCover 2021 |
| `kelvins_events_train.csv`, `kelvins_events_test.csv` | One row per event: latest warning ≥ 2 days out (+ final risk) | ESA Kelvins Collision Avoidance Challenge |
| `tle_*.txt` | Orbits: Sentinel-1A–1D, Cartosat, 4 debris clouds, all active satellites | CelesTrak (refresh on pitch day) |

## Results from the last run

| | Result |
|---|---|
| Flood AI, water IoU | Test floods 0.58 vs Otsu 0.51 (+14%) · Indian 0.70 vs 0.72 · Bolivia (unseen) 0.61 vs 0.70 |
| Kerala, 21 Aug 2018 | 217 km² new water (baseline) / 68 km² (AI); 142 / 45 km² farmland |
| Payout trigger | 517 / 138 cells; 11,979 / 2,422 ha |
| Collision risk, 2 days early | AI catches 9 of 13 dangerous events vs 8 for "do nothing"; F2 0.35 for both |
| Debris screen | 208 approaches under 10 km in 72 h (26 involve Sentinel-1); closest for Sentinel-1: Sentinel-1C vs Iridium 33 debris, 1.4 km at 14.5 km/s |

## Sources for key figures
- Kerala Post Disaster Needs Assessment (UN, World Bank, EU): ₹26,720 crore damages + losses.
- Karnataka 2019: 3,22,448 ha (CM's request, 10 Aug 2019) to 6.9 lakh ha (later preliminary assessment).
- PIB, Ministry of Agriculture & Farmers Welfare, 19 Jul 2022: PMFBY claims generally paid within two months of crop-cutting experiments / harvest.
- IWMI / UN-SPIDER flood map, 14 Aug 2018: Alappuzha 241.94 km², Kottayam 164.82 km².

## Not built yet
- Flood-history (2016–2024) premium pricing
- Full-resolution (10 m) flood AI
- Karnataka 2019 at its flood peak (no matching satellite pass)

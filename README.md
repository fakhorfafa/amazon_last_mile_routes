# What makes a high-quality delivery route?

An analysis of **6,112 real Amazon delivery routes** from the
[2021 Amazon Last Mile Routing Research Challenge](https://registry.opendata.aws/amazon-last-mile-challenges/)
(Merchán et al., *Transportation Science* 58(1), 2024). Each route was driven in 2018 in Seattle,
Los Angeles, Austin, Chicago or Boston. Amazon labeled each route **High, Medium, or Low** based on
how it was driven.

**Question:** Which features of the stop sequence a driver actually drove separate High-quality routes from the rest?

![dashboard](docs/dashboard.png)

## Results

The analysis covers 6,112 routes, 898,415 stops and 1,457,175 packages across 17 stations (19 Jul to 26 Aug 2018).

| | High | Medium | Low |
|---|---|---|---|
| Routes | 2,718 | 3,292 | 102 |
| Avg stops | 139.6 | 153.0 | 150.3 |
| Avg straight-line path (km) | 60.0 | 67.6 | 65.5 |
| Path ÷ nearest-neighbour tour | 1.064 | 1.081 | 1.073 |
| % of zones revisited | 3.9% | 4.0% | 3.2% |

1. **High routes are smaller routes.** They have about 13 fewer stops and about 8 km less driving than Medium routes, so workload is as much a part of "quality" as how the route is driven.
2. **High routes are driven slightly closer to a nearest-neighbour tour**, but the gap is small (1.064 vs 1.081).
3. **Zone discipline does not separate the quality levels.** Drivers at every level revisit only about 4% of their zones, which contradicted my initial hypothesis.
4. **Quality varies by city.** 50% of Seattle routes are High, compared with 33% in Boston.
5. A random forest that separates High from not-High reaches a **5-fold cross-validated ROC AUC of 0.69**. Path length, stop count and path ÷ NN are its top features. Route quality is therefore only partly explained by the observable sequence.

## Pipeline

```
data/raw/*.json  ──build_tables.py──▶  data/processed/{routes,stops,packages}.csv
                 ──analyze.py──────▶  data/processed/route_features.csv
                                      dashboard/data.js  ──▶  dashboard/index.html
```

| Step | What it does |
|---|---|
| `scripts/download.sh` | Downloads the training inputs over HTTPS. No AWS CLI is needed. `travel_times.json` (1.8 GB) is fetched only with `--with-travel-times`. |
| `scripts/build_tables.py` | Flattens the nested JSON into tidy route, stop and package tables. |
| `scripts/analyze.py` | Engineers route features, trains a cross-validated random forest, and exports dashboard data. |
| `scripts/analyze_extra.py` | High vs Low comparison, fleet capacity, service-time model and station benchmark. |
| `scripts/analyze_travel.py` | Streams `travel_times.json` to compare driver and optimizer drive time and estimate time-window risk. |
| `dashboard/index.html` | A static dashboard built with Chart.js and MapLibre GL. Serve the folder over HTTP to open it. |

### Engineered features

| Feature | Meaning |
|---|---|
| `zones_revisited_pct` | Share of the route's planning zones that the driver left and came back to later |
| `zone_switches_per_zone` | Number of zone changes along the sequence ÷ number of distinct zones (1.0 means each zone was entered once) |
| `path_vs_nn` | Haversine length of the driven sequence ÷ a greedy nearest-neighbour tour from the station |
| `first_leg_km` | Distance from the station to the first stop |
| `capacity_util` | Total package volume ÷ vehicle capacity |
| `tw_pct`, `failed_pct` | Share of packages with a delivery time window, and share of packages not delivered |

## Run it

Requires Python 3.9+ with `pandas`, `numpy` and `scikit-learn`.

```bash
sh scripts/download.sh --with-travel-times   # ~2.3 GB
python scripts/build_tables.py               # ~1 min
python scripts/analyze.py                    # ~2 min
python scripts/analyze_extra.py              # ~3 min
python scripts/analyze_travel.py             # ~30 min on 16 cores
cd dashboard && python -m http.server 8765   # open http://127.0.0.1:8765/
```

## Caveats

- Amazon obfuscated and perturbed the coordinates, so distances are straight-line approximations, not road distances.
- The labels are imbalanced: only 102 routes are Low. The model therefore predicts **High vs. not High**.
- Amazon derived its quality labels partly from factors that are not in the data, such as driver experience and customer feedback, so a modest AUC is expected.
- Only the training split is used. The evaluation split contains only High routes, so it cannot be used to compare quality levels.

## License and attribution

**This project, as published, is for non-commercial use only**, because the dataset it analyzes is licensed
CC BY-NC 4.0. Everything else in the project (code, fonts, libraries, map data) also allows commercial use.
To reuse the project commercially, keep the code and replace the data with data whose license allows it.

### Data

| Source | License | Use in this project |
|---|---|---|
| 2021 Amazon Last Mile Routing Research Challenge, Registry of Open Data on AWS (`s3://amazon-last-mile-challenges/almrrc2021/`) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | All route, stop, package and travel-time data. **Non-commercial only; attribution required.** |
| Merchán, D., Arora, J., Pachon, J., Konduri, K., Winkenbach, M., Parks, S., Noszek, J. (2024). *2021 Amazon Last Mile Routing Research Challenge: Data Set.* Transportation Science 58(1):8–11. [doi:10.1287/trsc.2022.1173](https://doi.org/10.1287/trsc.2022.1173) | © INFORMS, personal use | Cited as the dataset description. The PDF is not included in this repository. |
| [us-atlas 3.0.1](https://github.com/topojson/us-atlas), from US Census Bureau cartographic boundaries | ISC; source data public domain | State outlines on the overview map |
| [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, via [OpenMapTiles](https://openmaptiles.org/) | ODbL 1.0 / CC BY 4.0 | Streets and buildings on the 3D route map |
| [OpenFreeMap](https://openfreemap.org/) | Free, including commercial use | Vector basemap hosting for the 3D route map |

The map attributions required by OpenStreetMap, OpenMapTiles and OpenFreeMap are shown on the 3D map.
The dataset schema is documented at [MIT-CAVE/rc-cli](https://github.com/MIT-CAVE/rc-cli/blob/main/templates/data_structures.md).

### Software and fonts

| Component | License |
|---|---|
| [Outfit](https://fonts.google.com/specimen/Outfit) typeface, loaded from Google Fonts | SIL Open Font License 1.1 |
| [Chart.js](https://www.chartjs.org/) 4.4.1 | MIT |
| [MapLibre GL JS](https://maplibre.org/) 4.7.1 | BSD-3-Clause |
| pandas, NumPy, scikit-learn | BSD-3-Clause |
| Icons in the dashboard | Drawn for this project |

### Names and design references

- "Amazon" is used only to identify the data source. This project is not affiliated with or endorsed by Amazon.
- The glass styling is inspired by Apple's Liquid Glass design language. No Apple assets are used, and "Liquid Glass" is Apple's name for it.
- The chart styling takes visual cues from third-party dashboard mockups. No assets were copied from them.

<h1 align="center">
<i>Researcher Population Pyramids</i>
</h1>

<p align="center">
<a href="https://doi.org/10.1093/pnasnexus/pgag059" target="_blank">
<img alt="DOI: 10.1093/pnasnexus/pgag059" src="https://img.shields.io/badge/DOI-10.1093/pnasnexus/pgag059-blue.svg">
</a>
<a href="https://github.com/kazuibasou/researcher_population_pyramids/blob/main/LICENSE" target="_blank">
<img alt="License: MIT" src="https://img.shields.io/github/license/kazuibasou/researcher_population_pyramids">
</a>
</p>

An interactive visualization of researcher demographics and gender balance
across 58 countries, based on the population pyramid framework introduced in
our paper. The site lets users explore how researcher cohorts accumulate and
evolve from 1970 through projections to 2050.

**Live site:**
[kazuibasou.github.io/researcher_population_pyramids](https://kazuibasou.github.io/researcher_population_pyramids/)

**Paper:**
K. Nakajima and T. Mizuno (2026).
*Researcher population pyramids: Tracking demographic and gender trajectories
across countries*. PNAS Nexus 5(3), pgag059.
[doi.org/10.1093/pnasnexus/pgag059](https://doi.org/10.1093/pnasnexus/pgag059)

---

## What the site shows

The site has three visualizations:

- **Pyramid** — the researcher population pyramid for a chosen country, year,
  and research domain. Active authors of each gender are plotted by their
  cumulative productivity.
- **Trends** — time series of active-author counts and the proportion of
  female active authors for selected countries.
- **Comparison** — every country positioned by its researcher inflow against
  its gender gap in cumulative productivity at a chosen year.

Years 2024–2050 are projected from the 2023 pyramid under a stationarity
assumption (see the [About page](https://kazuibasou.github.io/researcher_population_pyramids/about.html)
for definitions and caveats; see the paper for the full methodology).

## How to cite

If you use the figures, data, or interface here in your work, please cite
the paper:

```bibtex
@article{Nakajima2026PopulationPyramids,
  title   = {Researcher population pyramids: Tracking demographic and
             gender trajectories across countries},
  author  = {Nakajima, Kazuki and Mizuno, Takayuki},
  journal = {PNAS Nexus},
  volume  = {5},
  number  = {3},
  pages   = {pgag059},
  year    = {2026},
  doi     = {10.1093/pnasnexus/pgag059}
}
```

## Repository structure

```
researcher_population_pyramids/
├── docs/                       # GitHub Pages site (this is what is served)
│   ├── index.html              #   Pyramid page
│   ├── trends.html             #   Trends page
│   ├── comparison.html         #   Comparison page
│   ├── about.html              #   About page
│   ├── css/, js/               #   Stylesheets and front-end logic
│   └── data/                   #   Aggregated JSON used by the site
│       ├── countries.json
│       ├── pyramids/<CC>.json              # 1970–2023 per country
│       ├── pyramids_projection/<CC>.json   # 2024–2050 per country
│       ├── timeline.json
│       └── scatter.json
├── scripts/                    # Data pipeline that produced docs/data/
│   ├── prepare_compact_pickles.py
│   ├── compute_pyramids.py
│   └── nqg_thresholds.json
├── LICENSE                     # MIT
└── README.md
```

## Data pipeline

The JSON files under `docs/data/` are produced by the scripts in `scripts/`.
The pipeline runs in two steps.

1. **Prepare compact per-country pickles.** Reads the per-country OpenAlex
   author pickles (private), assigns gender with
   [nomquamgender](https://github.com/ianvanbuskirk/nomquamgender) using the
   country-specific thresholds in `scripts/nqg_thresholds.json`, and writes
   one gzipped pickle per country.
   ```
   python scripts/prepare_compact_pickles.py
   ```
2. **Compute pyramids, projections, and aggregates.** Reads the compact
   pickles, computes inter-publication interval thresholds, constructs
   population pyramids for 1970–2023, projects them to 2050 under the
   stationarity assumption, and writes the per-country JSON files plus the
   site-wide aggregates (`timeline.json`, `scatter.json`, etc.).
   ```
   python scripts/compute_pyramids.py
   ```

The raw OpenAlex per-country pickles are not redistributed here — they
were extracted from the September 27, 2024 OpenAlex snapshot and are
several tens of gigabytes in total. The `docs/data/` JSON files in this
repository are the aggregated outputs of the pipeline and can be served
directly without rerunning anything.

## Running the site locally

```
cd docs/
python3 -m http.server 8000
# open http://localhost:8000/
```

No build step is required: the front-end is plain HTML / CSS / JavaScript
with [Plotly.js](https://plotly.com/javascript/) and
[Choices.js](https://github.com/Choices-js/Choices) loaded from a CDN.

## License

The code in this repository is released under the [MIT License](LICENSE).

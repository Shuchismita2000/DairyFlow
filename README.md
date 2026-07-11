# DairyFlow AI

Demand forecasting and inventory analytics for dairy supply chains, built around a hierarchical Bayesian model that pools sparse product × region transaction histories into a single, uncertainty-aware forecast.

Built for [hackathon name] · problem statement: *Demand Forecasting & Inventory Analytics*.

**[Live overview](./dairyflow_landing_page.html)** · **[Live console](./dairyflow_console.html)** · **[Data explorer](./dairyflow_eda_dashboard.html)** · **[Pitch content](./PITCH.md)**

---

## Contents

- [What this is](#what-this-is)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [How the model works](#how-the-model-works)
- [Data](#data)
- [Tech stack](#tech-stack)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [License](#license)

## What this is

Most dairy product × region segments in the source data have only ~28 recorded transactions across four years — too little history to forecast reliably on their own, and too much structure to ignore by using one flat rule for everything. DairyFlow fits a hierarchical Bayesian model (PyMC, NUTS) that partially pools estimates across products and regions, then converts each forecast's own uncertainty into a segment-specific safety stock and reorder recommendation.

The output is three static pages plus the pipeline that generates their data:

| Page | Purpose |
|---|---|
| `dairyflow_landing_page.html` | Narrative overview and methodology explainer |
| `dairyflow_console.html` | Interactive console — filterable forecast table, per-segment driver breakdown, natural-language query panel |
| `dairyflow_eda_dashboard.html` | Source-data exploration — volume trends, breakdowns, stockout-rate heatmap |

All three are self-contained static HTML (no build step, no server) and read from data computed offline by the pipeline — nothing calls a model live at demo time.

## Project structure

```
.
├── dairyflow_pipeline.py         # loads the CSV, fits the hierarchical model, exports forecast_output.json
├── forecast_output.json          # pipeline output: 150 product x region segments with forecasts + risk
├── dairyflow_landing_page.html   # overview / narrative page
├── dairyflow_console.html        # interactive console (forecast_output.json embedded)
├── dairyflow_eda_dashboard.html  # EDA dashboard (aggregated source-data stats embedded)
├── README.md                     # this file
└── PITCH.md                      # pitch-ready summary and suggested slide outline
```

## Getting started

The three HTML pages are pre-built with their data already embedded — clone the repo and open `dairyflow_landing_page.html` directly in a browser, no server required.

To regenerate `forecast_output.json` from the raw dataset (e.g. after changing the model):

```bash
pip install pymc arviz pandas numpy
python dairyflow_pipeline.py
```

This re-fits the model and overwrites `forecast_output.json`. The console and EDA pages currently have that file's contents embedded inline for portability — after regenerating, re-embed the JSON into `dairyflow_console.html` and `dairyflow_eda_dashboard.html` (replacing the `DATA = {...}` literal) if you want the pages to reflect the new run.

## How the model works

```
Raw transactions            Hierarchical model              Inventory logic              Recommendation
(4,325 rows,        →       (PyMC/NUTS, partial      →      (posterior percentiles  →    (reorder point +
 ~28 per segment)             pooling on product              -> safety stock,              risk level per
                              x region)                        robust to skew)               segment)
```

- **Model**: `log1p(quantity sold) ~ global intercept + product effect + region effect + product×region interaction (all partially pooled, non-centered parametrization) + monthly seasonality (sin/cos)`.
- **Sampling**: NUTS, 2 chains × 800 draws (800 tuning steps). Convergence checked via r-hat and effective sample size — see `forecast_output.json["meta"]` for the run's diagnostics.
- **Forecast**: posterior predictive samples per product × region, back-transformed from log scale.
- **Safety stock**: `p95 − p50` of the posterior predictive distribution (percentile-based, not mean ± z·σ — demand here is right-skewed, so a variance-based buffer overstates what's needed).
- **Risk level**: derived from historical stockout rate (share of records where recorded stock was below the threshold on file), which varies 0–39% across segments and is a more informative discriminator than the raw threshold column.

Full methodology writeup and findings are in [`PITCH.md`](./PITCH.md).

## Data

Source: a synthetic-style dairy supply chain dataset — 4,325 transactions, Jan 2019–Dec 2022, across 15 Indian states, 10 products, 11 brands. Columns include farm metadata (land area, cow count, farm size), production and sales quantities, pricing, shelf life and storage condition, and inventory fields (stock on hand, minimum threshold, reorder quantity).

## Tech stack

- **Modeling**: Python, PyMC, ArviZ, pandas, NumPy
- **Frontend**: vanilla HTML/CSS/JS, no framework or build step
- **In-browser NLP**: [`transformers.js`](https://github.com/xenova/transformers.js) running `Xenova/all-MiniLM-L6-v2` client-side for the console's query panel — no API key, no server round-trip
- **Type**: Fraunces (display), Inter (body), IBM Plex Mono (data/labels) via Google Fonts

## Known limitations

- Transactions are irregularly timestamped rather than sampled on a fixed daily/weekly cadence, so reorder quantities are sized against per-transaction demand rather than a fixed review window.
- Product, region, and interaction effects are individually small — the dataset shows limited segment-level structure — so the more meaningful diagnostic is the shrinkage delta (how much pooling moved each estimate versus that segment's own data alone), included per segment in `forecast_output.json`.
- The natural-language console panel uses a small fixed intent set plus keyword entity extraction; it's a semantic-matching layer, not a general-purpose chat interface.

## Roadmap

- Replace the historical batch export with a live farm-to-collection-center data feed.
- Extend the pooling hierarchy with weather and festival-calendar covariates.
- Generate plain-English restocking notes per segment once the pipeline is validated on live data.

## License

Prototype built for hackathon submission — add a license here if the repo will be maintained beyond the event.

# DairyFlow AI — Pitch

Pitch-ready content for slides and the verbal pitch. For setup, architecture, and technical detail, see [`README.md`](./README.md).

Live pages: [`dairyflow_landing_page.html`](./dairyflow_landing_page.html) · [`dairyflow_console.html`](./dairyflow_console.html) · [`dairyflow_eda_dashboard.html`](./dairyflow_eda_dashboard.html)

---

## 60-second pitch (copy straight into slide 1 / the verbal pitch)

Dairy managers plan inventory manually against a fixed minimum-stock threshold. In our dataset, **21% of 4,325 historical records (918 transactions) were already below that threshold** — and it's not because demand is unpredictable, it's because the threshold barely scales with demand at all: it averages ~56 units across the board, while actual demand averages ~248 units per transaction, whether that's Milk in Bihar or Paneer in Kerala.

DairyFlow AI replaces the flat threshold with a **hierarchical Bayesian demand model** that pools statistical strength across products and regions — so a product-region pair with only ~28 historical transactions still gets a credible forecast, borrowed from how that product behaves nationally and how that region behaves across products. Every forecast carries its own uncertainty, which becomes a segment-specific safety stock and reorder point — not the same buffer applied everywhere.

The result: 150 product×region segments, each scored High/Medium/Low risk from real historical evidence, explorable through a live console with a free, local natural-language copilot — no API key, no server, no per-query cost.

---

## The problem

Dairy is perishable (15–72 day shelf life in this dataset) and supply chain data is naturally sparse at the segment level — most product-region pairs have only a few dozen recorded transactions across four years. That combination breaks two common approaches at once:

- **Manual weekly review** catches stockouts and overproduction only after they've happened — the review cycle is longer than the shelf life for several products.
- **One forecasting model per product-region pair** (the usual approach) overfits badly on ~28 data points. There isn't enough history in any single segment to trust in isolation.

## The approach: partial pooling

Instead of a model per segment, or one flat model for everything, DairyFlow fits a single hierarchical model where each product×region segment has its own effect, but that effect is *regularized toward the product's and region's group-level pattern*. Thin segments borrow strength from the group; segments with more data are trusted more on their own. This is a standard technique in Bayesian hierarchical modeling (partial pooling / shrinkage), applied here because the data's actual structure — sparse, unevenly-sized groups — calls for it, not because it's the flashiest option.

```
Raw transaction     →   Hierarchical model        →   Inventory logic       →   Recommendation
(4,325 rows,             (PyMC, NUTS,                  (uncertainty → safety     (reorder point +
 ~28/segment)              product × region              stock, percentile-        risk flag per
                           partial pooling)               based, robust to           segment)
                                                           right-skewed demand)
```

## What the model actually found

Fitted with PyMC (NUTS, 2 chains × 800 draws after 800 tuning steps). Convergence: **max r-hat 1.01, min ESS 519** — clean convergence, no divergences flagged.

- **150 product×region segments** scored: **22 High risk, 97 Medium, 31 Low**, ranked by real historical stockout rate (0–39% across segments) rather than the flat threshold column, which doesn't discriminate between segments.
- **Pooling materially changed estimates.** For several segments, the pooled forecast differs from what the segment's own ~25–30 records alone would have said by 70–170 units — a direct, quantifiable measure of how much the thin-data problem would have misled a naive per-segment model.
- **A secondary finding, not a modeling artifact**: the current `Minimum Stock Threshold` column averages ~56 units regardless of product or region, while demand averages ~248 units per transaction. The existing policy isn't really responsive to demand — that gap is worth stating plainly in the pitch, since it's evidence-based, not something the model manufactured.

## Honest scope notes (say these before a judge finds them)

- Transactions are timestamped irregularly, not on a fixed daily/weekly cadence, so "reorder point" here is sized against per-transaction demand rather than a fixed review window. That's a deliberate scoping choice given the data, not an oversight — worth one sentence in the demo.
- Demand is right-skewed (a few very large batches), so safety stock is computed from **posterior percentiles (p95 − p50)** rather than a mean ± z·σ rule, which would be dominated by the tail and wildly overstate the buffer needed.
- Product/region/interaction effects individually are small (the dataset looks close to randomly generated at the segment level) — the shrinkage magnitude is the more meaningful number to show, since it quantifies how much the naive per-segment estimate would have been wrong.

## Architecture

- **`dairyflow_pipeline.py`** — loads the CSV, fits the hierarchical model, exports `forecast_output.json` (150 segments: forecast percentiles, safety stock, reorder point, risk level, driver decomposition).
- **`dairyflow_landing_page.html`** — narrative overview, methodology explainer, the pooling diagram as the signature visual.
- **`dairyflow_console.html`** — the working dashboard: filterable/sortable segment table, per-segment driver panel, and a natural-language query panel.
- **`dairyflow_eda_dashboard.html`** — source-data exploration: volume trends, product/region/channel breakdowns, and the stockout-rate heatmap the risk scoring is built from.

All three HTML pages are static, self-contained, and read from data computed offline — no server, no live model calls at demo time.

### The copilot: free, local, no API key

The console's natural-language panel runs **MiniLM** (`Xenova/all-MiniLM-L6-v2`) entirely client-side via `transformers.js`. It combines simple keyword entity extraction (product/region/risk names) with semantic intent matching (cosine similarity against a small set of canned intents) so phrasing variation still resolves correctly — "show me the risky ones in Karnataka" and "high risk Karnataka segments" both work. If the model fails to load (offline demo, slow wifi), it falls back to keyword-only matching rather than breaking.

## Tech stack

Python (pandas, PyMC, ArviZ) for the model · vanilla HTML/CSS/JS for all three frontend pages (no build step, no framework) · `transformers.js` + MiniLM for the in-browser copilot · Google Fonts (Fraunces + Inter + IBM Plex Mono) for type.

## Running the pipeline

```bash
pip install pymc arviz pandas numpy
python dairyflow_pipeline.py
# writes forecast_output.json — the console and EDA pages already have a copy embedded
```

## What's next

- **Data**: replace the historical batch export with a live farm-to-collection-center feed.
- **Model**: extend the pooling hierarchy with weather and festival-calendar covariates — both known demand drivers for dairy in India — and add a proper time index once transaction cadence is regular enough to support it.
- **UX**: turn driver output into short, plain-English restocking notes for regional managers, once the numeric pipeline is validated against live data.

---

## Suggested slide outline

1. **Hook** — the 21% stat + the threshold-scale-mismatch finding (one sentence each).
2. **Problem** — two failure modes, dairy's short shelf life makes both expensive.
3. **Why partial pooling** — show the pooling diagram; explain thin segments (~28 records) in one line.
4. **Live demo** — open the console, filter to High risk, click a segment, ask the copilot a question.
5. **Rigor** — convergence diagnostics (r-hat 1.01), shrinkage magnitude as evidence pooling mattered.
6. **Honest scope** — the transaction-cadence caveat, stated proactively.
7. **What's next** — three bullets already above; keep it to three.

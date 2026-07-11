"""
DairyFlow AI — hierarchical Bayesian demand forecasting + inventory logic.
Pools sparse product x region transaction histories via partial pooling (PyMC/NUTS),
then converts posterior predictive uncertainty directly into safety stock / reorder
recommendations. Exports a static JSON consumed by the console frontend.
"""
import json
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

RNG_SEED = 42
np.random.seed(RNG_SEED)

# ---------------------------------------------------------------
# 1. Load + prep
# ---------------------------------------------------------------
df = pd.read_csv("/mnt/user-data/uploads/dairy_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["month"] = df["Date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

products = sorted(df["Product Name"].unique())
regions = sorted(df["Location"].unique())
p_idx_map = {p: i for i, p in enumerate(products)}
r_idx_map = {r: i for i, r in enumerate(regions)}
df["p_idx"] = df["Product Name"].map(p_idx_map)
df["r_idx"] = df["Location"].map(r_idx_map)

n_products = len(products)
n_regions = len(regions)

y = np.log1p(df["Quantity Sold (liters/kg)"].values)
p_idx = df["p_idx"].values
r_idx = df["r_idx"].values
month_sin = df["month_sin"].values
month_cos = df["month_cos"].values

print(f"n_obs={len(df)}  n_products={n_products}  n_regions={n_regions}")

# ---------------------------------------------------------------
# 2. Hierarchical model — partial pooling on product x region
# ---------------------------------------------------------------
with pm.Model() as model:
    global_intercept = pm.Normal("global_intercept", mu=float(y.mean()), sigma=1.0)

    sigma_product = pm.HalfNormal("sigma_product", sigma=1.0)
    product_raw = pm.Normal("product_raw", mu=0, sigma=1, shape=n_products)
    product_effect = pm.Deterministic("product_effect", product_raw * sigma_product)

    sigma_region = pm.HalfNormal("sigma_region", sigma=1.0)
    region_raw = pm.Normal("region_raw", mu=0, sigma=1, shape=n_regions)
    region_effect = pm.Deterministic("region_effect", region_raw * sigma_region)

    sigma_interaction = pm.HalfNormal("sigma_interaction", sigma=0.5)
    interaction_raw = pm.Normal("interaction_raw", mu=0, sigma=1, shape=(n_products, n_regions))
    interaction_effect = pm.Deterministic("interaction_effect", interaction_raw * sigma_interaction)

    b_sin = pm.Normal("b_sin", mu=0, sigma=1)
    b_cos = pm.Normal("b_cos", mu=0, sigma=1)

    mu = (
        global_intercept
        + product_effect[p_idx]
        + region_effect[r_idx]
        + interaction_effect[p_idx, r_idx]
        + b_sin * month_sin
        + b_cos * month_cos
    )

    sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
    pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y)

    trace = pm.sample(
        draws=800, tune=800, chains=2, cores=2,
        target_accept=0.9, random_seed=RNG_SEED, progressbar=True,
    )

# ---------------------------------------------------------------
# 3. Convergence diagnostics (kept for the README / rigor)
# ---------------------------------------------------------------
summary = az.summary(trace, var_names=["global_intercept", "sigma_product", "sigma_region",
                                        "sigma_interaction", "b_sin", "b_cos", "sigma_obs"])
max_rhat = float(summary["r_hat"].max())
min_ess = float(summary["ess_bulk"].min())
print(f"max r_hat={max_rhat:.4f}  min ess_bulk={min_ess:.1f}")

# ---------------------------------------------------------------
# 4. Posterior predictive per product x region, back-transformed
# ---------------------------------------------------------------
post = trace.posterior
gi = post["global_intercept"].values.reshape(-1)
pe = post["product_effect"].values.reshape(-1, n_products)
re = post["region_effect"].values.reshape(-1, n_regions)
ie = post["interaction_effect"].values.reshape(-1, n_products, n_regions)
sig = post["sigma_obs"].values.reshape(-1)
n_draws = gi.shape[0]

results = []
for pi, prod in enumerate(products):
    for ri, reg in enumerate(regions):
        seg = df[(df["p_idx"] == pi) & (df["r_idx"] == ri)]
        n_obs = len(seg)
        if n_obs == 0:
            continue

        # posterior predictive samples on log scale (month effect averaged out ~ 0)
        mu_draws = gi + pe[:, pi] + re[:, ri] + ie[:, pi, ri]
        pred_log = np.random.normal(mu_draws, sig)
        pred = np.expm1(pred_log)
        pred = np.clip(pred, 0, None)

        mean_demand = float(np.mean(pred))
        p05 = float(np.percentile(pred, 5))
        p50 = float(np.percentile(pred, 50))
        p95 = float(np.percentile(pred, 95))
        std_demand = float(np.std(pred))

        # naive raw estimate using only this segment's own data (no pooling)
        raw_mean = float(seg["Quantity Sold (liters/kg)"].mean())
        shrinkage_delta = float(mean_demand - raw_mean)  # how much pooling moved the estimate

        # demand here is right-skewed (log-normal shaped), so a percentile-based
        # buffer is far more stable than sigma*z: std is dominated by the rare
        # large-batch tail and wildly overstates the buffer needed.
        safety_stock = float(max(p95 - p50, 0.0))
        reorder_point = float(p95)

        current_threshold = float(seg["Minimum Stock Threshold (liters/kg)"].mean())
        current_reorder_qty = float(seg["Reorder Quantity (liters/kg)"].mean())
        current_stock_avg = float(seg["Quantity in Stock (liters/kg)"].mean())
        pct_below_threshold_hist = float(
            (seg["Quantity in Stock (liters/kg)"] < seg["Minimum Stock Threshold (liters/kg)"]).mean() * 100
        )

        # risk = actual historical stockout evidence for this segment (how often
        # recorded stock already sat below the threshold on file). This is real
        # signal in the data and varies 0-39% across segments -- unlike comparing
        # against the threshold column directly, which is flat (roughly 10-99
        # units) regardless of whether typical demand there is 100 or 900 units,
        # so it cannot be used as a per-segment risk discriminator on its own.
        gap_ratio = float((safety_stock - current_threshold) / max(safety_stock, 1e-6))
        overstocked = current_reorder_qty > 2 * mean_demand

        if pct_below_threshold_hist >= 30:
            risk = "High"
        elif pct_below_threshold_hist >= 15:
            risk = "Medium"
        else:
            risk = "Low"
        risk_prob = round(pct_below_threshold_hist / 100.0, 3)

        results.append({
            "product": prod,
            "region": reg,
            "n_obs": n_obs,
            "unit": "kg" if prod in ["Paneer", "Cheese", "Butter", "Ghee"] else "liters",
            "forecast_mean": round(mean_demand, 2),
            "forecast_p05": round(p05, 2),
            "forecast_p50": round(p50, 2),
            "forecast_p95": round(p95, 2),
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(reorder_point, 2),
            "risk_level": risk,
            "risk_prob": risk_prob,
            "overstock_flag": bool(overstocked),
            "current_threshold": round(current_threshold, 2),
            "current_reorder_qty": round(current_reorder_qty, 2),
            "current_stock_avg": round(current_stock_avg, 2),
            "pct_below_threshold_hist": round(pct_below_threshold_hist, 1),
            "drivers": {
                "product_effect": round(float(np.mean(pe[:, pi])), 3),
                "region_effect": round(float(np.mean(re[:, ri])), 3),
                "interaction_effect": round(float(np.mean(ie[:, pi, ri])), 3),
                "shrinkage_delta": round(shrinkage_delta, 2),
            },
        })

results.sort(key=lambda r: (-{"High": 2, "Medium": 1, "Low": 0}[r["risk_level"]], -r["risk_prob"]))

output = {
    "meta": {
        "n_obs": int(len(df)),
        "n_products": n_products,
        "n_regions": n_regions,
        "products": products,
        "regions": regions,
        "date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "model": "hierarchical partial pooling (product x region), PyMC NUTS",
        "draws": int(n_draws),
        "max_r_hat": round(max_rhat, 4),
        "min_ess_bulk": round(min_ess, 1),
        "pct_below_threshold_overall": round(
            float((df["Quantity in Stock (liters/kg)"] < df["Minimum Stock Threshold (liters/kg)"]).mean() * 100), 1
        ),
        "note_threshold_scale_mismatch": (
            "Minimum Stock Threshold in the source data averages about 56 units (range 10-99) "
            "and does not scale with actual demand, which averages about 248 units per "
            "transaction (range 1-960). That mismatch is itself a finding: the current policy "
            "applies roughly the same buffer regardless of how much a segment actually sells."
        ),
    },
    "segments": results,
}

with open("/home/claude/dairyflow/forecast_output.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved forecast_output.json with", len(results), "segments")
print("High risk:", sum(1 for r in results if r["risk_level"] == "High"))
print("Medium risk:", sum(1 for r in results if r["risk_level"] == "Medium"))
print("Low risk:", sum(1 for r in results if r["risk_level"] == "Low"))

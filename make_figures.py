"""
Paper figures from results/*.csv (matplotlib only). Writes PDF (for LaTeX) and PNG (300 dpi) into results/.
  fig_perf_violations : held-out RMSE | Durrleman violations | calendar violations   (clean data)
  fig_slices          : illustrative slices, stressed regime, seed 0 (ground truth, SSVI, hard-constrained, training points)
  fig_stress_recovery : error at distorted points vs clean truth | vs distorted data  (stress test)
Bars = mean over seeds, whiskers = sample sd (lower whisker truncated at 0 for rates/errors), dots = individual seeds.
Usage: python make_figures.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import config, ivs_core as core
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
MAIN = ["svi", "ssvi", "soft_penalty", "hard_constrained"]
LAB = {"svi": "Per-slice SVI", "ssvi": "SSVI", "soft_penalty": "Soft-penalty MLP", "hard_constrained": "Hard-constrained (ours)"}
COL = {"svi": "#4C72B0", "ssvi": "#DD8452", "soft_penalty": "#55A868", "hard_constrained": "#C44E52"}
RN = {"normal": "normal", "high_skew": "high-skew", "stressed": "stressed"}
REG = config.ACTIVE_REGIMES

def grouped(ax, df, metric, title, ylabel, scale=1.0):
    w = 0.2
    for i, m in enumerate(MAIN):
        x = np.arange(len(REG)) + (i - 1.5) * w
        vals = [df[(df.regime == r) & (df.model == m)][metric].astype(float).values * scale for r in REG]
        mean = np.array([v.mean() for v in vals]); sd = np.array([v.std(ddof=1) for v in vals])
        ax.bar(x, mean, w, yerr=[np.minimum(sd, mean), sd], capsize=2, color=COL[m], alpha=0.85, label=LAB[m], error_kw=dict(lw=1))
        for xi, v in zip(x, vals):
            ax.scatter(xi + np.linspace(-w * 0.25, w * 0.25, len(v)), v, s=7, color="k", alpha=0.6, zorder=3)
    ax.set_xticks(range(len(REG))); ax.set_xticklabels([RN[r] for r in REG])
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel); ax.set_xlabel("Regime"); ax.set_ylim(bottom=0)

def save(fig, name):
    fig.savefig(os.path.join(RES, name + ".pdf")); fig.savefig(os.path.join(RES, name + ".png"), dpi=300); plt.close(fig)

df = pd.read_csv(os.path.join(RES, "multiseed_results.csv")); df = df[df.model.isin(MAIN)]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
grouped(ax[0], df, "rmse_heldout", "Held-out RMSE", "RMSE (total variance $w$)")
grouped(ax[1], df, "durrleman_violation_rate", "Durrleman violation rate", "violations (%)", 100)
grouped(ax[2], df, "calendar_violation_rate", "Calendar violation rate", "violations (%)", 100)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "fig_perf_violations")

sdf = pd.read_csv(os.path.join(RES, "stress_results.csv")); sdf = sdf[sdf.model.isin(MAIN)]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
grouped(ax[0], sdf, "rmse_distorted_vs_truth", "Error at distorted points vs clean surface", "RMSE ($w$)")
grouped(ax[1], sdf, "rmse_distorted_vs_target", "Error at distorted points vs distorted data", "RMSE ($w$)")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.09, 1, 1)); save(fig, "fig_stress_recovery")

seed, regime = 0, "stressed"
k_grid = np.linspace(config.K_GRID_MIN, config.K_GRID_MAX, config.K_GRID_POINTS)
params = core.make_regime_svi_params(config.MATURITIES, config.SVI_REGIMES[regime], seed=seed)
gt = core.generate_ground_truth_grid(k_grid, config.MATURITIES, params)
train, mask = core.corrupt_data(gt, config.CORRUPTION_NOISE_STD, config.CORRUPTION_KEEP_FRAC, seed=seed)
ssvi_fn = core.fit_ssvi(train, config.MATURITIES, n_starts=config.SSVI_N_STARTS, seed=seed)
hard_fn = core.torch_w_fn(core.train_hard_constrained(train, config.MATURITIES, seed=seed)[0])
fine_k = np.linspace(config.K_GRID_MIN, config.K_GRID_MAX, 201)
fig, axes = plt.subplots(1, len(config.MATURITIES), figsize=(16, 3.6), sharey=False)
for i, T in enumerate(config.MATURITIES):
    a = axes[i]; Tk = np.full_like(fine_k, T)
    a.plot(fine_k, core.svi_total_variance(fine_k, *params[i]), "k-", lw=2, label="Ground truth")
    a.plot(fine_k, ssvi_fn(fine_k, Tk), "--", color=COL["ssvi"], label="SSVI")
    a.plot(fine_k, hard_fn(fine_k, Tk), "-.", color=COL["hard_constrained"], label="Hard-constrained")
    m = np.isclose(train[:, 1], T); a.scatter(train[m, 0], train[m, 2], c="k", s=14, zorder=5, label="Training points")
    a.set_title(f"T = {T}", fontsize=10); a.set_xlabel("log-moneyness $k$")
    if i == 0: a.set_ylabel("total variance $w$")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1)); save(fig, "fig_slices")
print("figures written to", RES)

"""
Turns results/multiseed_results.csv and results/stress_results.csv into paper-ready tables:
  results/tables.md   (readable)     results/tables.tex  (LaTeX rows)

Reports mean +/- std (ddof=1), median, and PAIRED per-seed comparisons of the hard-constrained
model against each other model (win counts, mean paired difference). With only a handful of
seeds no p-values are reported on purpose: n=5 cannot support them.
Usage: python make_tables.py
"""
import os

import numpy as np
import pandas as pd

import config

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
NAMES = {"svi": "SVI (per slice)", "ssvi": "SSVI", "soft_penalty": "Soft-penalty MLP",
         "hard_constrained": "Hard-constrained (ours)", "hard_no_calib": "Hard, no init calibration",
         "dugas_style": "Dugas-style (naive init)"}


def ms(x, digits=4, pct=False):
    x = np.asarray(x, float)
    if np.isnan(x).all():
        return "n/a"
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    if pct:
        return f"{100 * x.mean():.2f}% ± {100 * sd:.2f}%"
    return f"{x.mean():.{digits}f} ± {sd:.{digits}f}"


def table(df, cols, header, pct_cols=(), digits=4, extra_med=None):
    lines = [f"| Regime | Model | n | " + " | ".join(header) + " |",
             "|---|---|---|" + "---|" * len(header)]
    tex = []
    for regime in config.ACTIVE_REGIMES:
        for model in config.MODELS:
            g = df[(df.regime == regime) & (df.model == model)]
            if g.empty:
                continue
            cells = [ms(g[c], digits, pct=(c in pct_cols)) for c in cols]
            if extra_med:
                cells[0] += f" (med {g[extra_med].median():.{digits}f})"
            lines.append(f"| {regime} | {NAMES[model]} | {len(g)} | " + " | ".join(cells) + " |")
            tex.append(f"{regime.replace('_', ' ')} & {NAMES[model]} & " + " & ".join(cells).replace("%", "\\%")
                       .replace("±", "$\\pm$") + " \\\\")
    return "\n".join(lines), "\n".join(tex)


def paired(df, metric, ref="hard_constrained"):
    lines = [f"| Regime | vs model | mean({ref} − model) | {ref} lower in |", "|---|---|---|---|"]
    for regime in config.ACTIVE_REGIMES:
        a = df[(df.regime == regime) & (df.model == ref)].set_index("seed")[metric]
        for model in config.MODELS:
            if model == ref:
                continue
            b = df[(df.regime == regime) & (df.model == model)].set_index("seed")[metric]
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            d = a[common] - b[common]
            lines.append(f"| {regime} | {NAMES[model]} | {d.mean():+.5f} | {(d < 0).sum()}/{len(common)} seeds |")
    return "\n".join(lines)


def main():
    md, tex = [], []
    for fname, title in [("multiseed_results.csv", "Clean data"), ("stress_results.csv", "Calendar-inversion stress test")]:
        path = os.path.join(RES, fname)
        if not os.path.exists(path):
            print(f"missing {path}, skipping")
            continue
        df = pd.read_csv(path)
        md.append(f"\n## {title}\n")
        md.append("### Accuracy vs clean ground truth (total variance w; iv_rmse in vol units)\n")
        cols = ["rmse_all", "rmse_heldout", "iv_rmse_all"]
        if "rmse_distorted_vs_truth" in df:
            cols += ["rmse_distorted_vs_truth", "rmse_distorted_vs_target"]
        t, x = table(df, cols, cols, extra_med="rmse_heldout" if False else None)
        md.append(t); tex.append(f"% {title}: accuracy\n{x}")
        md.append("\n### No-arbitrage diagnostics at the 5 maturities (fine k-grid; same finite-difference checker for all models)\n")
        vc = ["butterfly_violation_rate", "durrleman_violation_rate", "calendar_violation_rate", "calendar_max_violation"]
        t, x = table(df, vc, ["butterfly (convexity)", "Durrleman g(k)", "calendar", "max calendar viol. (w units)"],
                     pct_cols=vc[:3], digits=5)
        md.append(t); tex.append(f"% {title}: violations\n{x}")
        md.append("\n### Continuous-in-T models: violations on a dense T grid between maturities\n")
        dc = ["butterfly_violation_rate_denseT", "durrleman_violation_rate_denseT", "calendar_violation_rate_denseT"]
        dd = df[df.model.isin(["soft_penalty", "hard_constrained", "hard_no_calib", "dugas_style"])]
        t, _ = table(dd, dc, ["butterfly (convexity)", "Durrleman g(k)", "calendar"], pct_cols=dc)
        md.append(t)
        md.append("\n### Paired per-seed comparison on held-out RMSE (negative = hard-constrained better)\n")
        md.append(paired(df, "rmse_heldout"))
        md.append("\n### Optimisation diagnostics (final training MSE)\n")
        t, _ = table(df[df.model.isin(["soft_penalty", "hard_constrained", "hard_no_calib", "dugas_style"])],
                     ["first_train_mse", "final_train_mse"], ["initial train MSE", "final train MSE"], digits=6)
        md.append(t)
    with open(os.path.join(RES, "tables.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(RES, "tables.tex"), "w") as f:
        f.write("\n\n".join(tex))
    print("\n".join(md))
    print(f"\nWrote {RES}/tables.md and tables.tex")


if __name__ == "__main__":
    main()

"""
Clean-data multi-seed experiment: SVI, SSVI, soft-penalty, hard-constrained,
hard-constrained-without-calibration, Dugas-style -- for every regime in
config.ACTIVE_REGIMES and every seed in config.SEEDS.

Each seed controls the SVI parameter draw, the noise/mask draw and the network
initialisation. Every model in a (regime, seed) sees the IDENTICAL training set.

Writes results/multiseed_results.csv (one row per regime/seed/model). The run is
resumable: re-running skips (regime, seed) blocks already present in the CSV.

Usage:  python run_multiseed.py            # full run (config.SEEDS)
        python run_multiseed.py --smoke    # 1 seed, 1 regime, few epochs (pipeline check only)
"""
import argparse
import csv
import os
import sys

import numpy as np

import config
import ivs_core as core

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIELDS = ["regime", "seed", "model",
          "rmse_all", "rmse_heldout", "rmse_train_pts", "iv_rmse_all",
          "butterfly_violation_rate", "durrleman_violation_rate", "nonpositive_rate",
          "calendar_violation_rate", "calendar_max_violation",
          "butterfly_violation_rate_denseT", "durrleman_violation_rate_denseT",
          "calendar_violation_rate_denseT", "calendar_max_violation_denseT",
          "final_train_mse", "first_train_mse", "train_time_s"]


def run_single(regime_name, regime_config, seed):
    k_grid = np.linspace(config.K_GRID_MIN, config.K_GRID_MAX, config.K_GRID_POINTS)
    params = core.make_regime_svi_params(config.MATURITIES, regime_config, seed=seed)
    ground_truth = core.generate_ground_truth_grid(k_grid, config.MATURITIES, params)
    train_grid, mask = core.corrupt_data(ground_truth, config.CORRUPTION_NOISE_STD,
                                         config.CORRUPTION_KEEP_FRAC, seed=seed)
    rows = []
    for name in config.MODELS:
        w_fn, info = core.fit_model(name, train_grid, config.MATURITIES, seed)
        metrics = core.evaluate_surface(w_fn, ground_truth, mask, config.MATURITIES)
        rows.append({"regime": regime_name, "seed": seed, "model": name, **metrics, **info})
        print(f"  {name:<18} rmse_all={metrics['rmse_all']:.5f} heldout={metrics['rmse_heldout']:.5f} "
              f"bfly={metrics['butterfly_violation_rate']:.3%} durr={metrics['durrleman_violation_rate']:.3%} "
              f"cal={metrics['calendar_violation_rate']:.3%}", flush=True)
    return rows


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                done.setdefault((r["regime"], int(r["seed"])), set()).add(r["model"])
    return {key for key, models in done.items() if models >= set(config.MODELS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        config.SEEDS, config.ACTIVE_REGIMES = [0], ["normal"]
        config.SOFT_PENALTY_EPOCHS = config.HARD_EPOCHS = config.DUGAS_EPOCHS = 50
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "multiseed_smoke.csv" if args.smoke else "multiseed_results.csv")
    done = load_done(out_path)
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for regime_name in config.ACTIVE_REGIMES:
            for seed in config.SEEDS:
                if (regime_name, seed) in done:
                    print(f"[{regime_name}] seed={seed} already done, skipping")
                    continue
                print(f"[{regime_name}] seed={seed}", flush=True)
                writer.writerows(run_single(regime_name, config.SVI_REGIMES[regime_name], seed))
                f.flush()
    print(f"\nWrote {out_path}. Next: python make_tables.py")


if __name__ == "__main__":
    main()

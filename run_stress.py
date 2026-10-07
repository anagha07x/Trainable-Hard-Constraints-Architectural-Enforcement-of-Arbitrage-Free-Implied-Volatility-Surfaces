"""
Targeted calendar-inversion stress test (replaces evaluate_stress_regimes.py).

For every regime and seed:
  * build the same ground truth / noise / mask as run_multiseed.py;
  * replace the training data at maturity index config.STRESS_MATURITY_INDEX
    (|k| <= config.STRESS_K_BAND) with  w_true(k, T_{j-1}) - config.STRESS_DELTA,
    i.e. the later maturity's observations sit BELOW the earlier maturity's true slice;
  * train every model on that distorted set (nothing is duplicated into training);
  * report, in addition to the usual metrics:
      rmse_distorted_vs_truth  : error at the distorted points vs the CLEAN truth
                                 (small = the model corrected the distortion)
      rmse_distorted_vs_target : error at the distorted points vs the distorted data
                                 (small = the model reproduced the distortion)

Writes results/stress_results.csv (resumable). Usage: python run_stress.py [--smoke]
"""
import argparse
import csv
import os

import numpy as np

import config
import ivs_core as core
from run_multiseed import FIELDS as BASE_FIELDS, RESULTS_DIR

FIELDS = BASE_FIELDS + ["rmse_distorted_vs_truth", "rmse_distorted_vs_target"]


def run_single(regime_name, regime_config, seed):
    k_grid = np.linspace(config.K_GRID_MIN, config.K_GRID_MAX, config.K_GRID_POINTS)
    params = core.make_regime_svi_params(config.MATURITIES, regime_config, seed=seed)
    ground_truth = core.generate_ground_truth_grid(k_grid, config.MATURITIES, params)
    train_grid, mask, distorted, distorted_target = core.make_stress_train_set(
        ground_truth, config.MATURITIES, config.STRESS_MATURITY_INDEX, config.STRESS_DELTA,
        config.STRESS_K_BAND, config.CORRUPTION_NOISE_STD, config.CORRUPTION_KEEP_FRAC, seed)
    extra = {"rmse_distorted_vs_truth": (distorted, ground_truth[:, 2]),
             "rmse_distorted_vs_target": (distorted, distorted_target)}
    rows = []
    for name in config.MODELS:
        w_fn, info = core.fit_model(name, train_grid, config.MATURITIES, seed)
        metrics = core.evaluate_surface(w_fn, ground_truth, mask, config.MATURITIES, extra_rows=extra)
        rows.append({"regime": regime_name, "seed": seed, "model": name, **metrics, **info})
        print(f"  {name:<18} rmse_all={metrics['rmse_all']:.5f} "
              f"dist_vs_truth={metrics['rmse_distorted_vs_truth']:.5f} "
              f"dist_vs_target={metrics['rmse_distorted_vs_target']:.5f} "
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
    out_path = os.path.join(RESULTS_DIR, "stress_smoke.csv" if args.smoke else "stress_results.csv")
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

"""
Sanity checks -- run this BEFORE the experiments (takes ~1 minute). Every check prints PASS/FAIL
and the script exits non-zero if anything fails.

  1. Ground truth is arbitrage-free (numeric Durrleman + convexity + calendar) for 50 seeds x 3 regimes.
  2. Hard-constrained net: convex in k and non-decreasing in T for RANDOMLY PERTURBED weights
     (not just the initial or trained ones) -> the guarantee holds by construction.
  3. The hard net is positive everywhere and continuous in T (queried between maturities).
  4. Stratified masking keeps >= MIN_POINTS_PER_MATURITY points per maturity and both wing strikes.
  5. Stress-set construction really inverts the calendar order in the DATA.
Usage: python run_checks.py
"""
import sys

import numpy as np
import torch

import config
import ivs_core as core

failed = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failed.append(name)


fine_k = np.linspace(config.K_GRID_MIN, config.K_GRID_MAX, config.FINE_K_GRID_POINTS)
k_grid = np.linspace(config.K_GRID_MIN, config.K_GRID_MAX, config.K_GRID_POINTS)

# 1
for regime in config.ACTIVE_REGIMES:
    bad = 0
    for seed in range(50):
        p = core.make_regime_svi_params(config.MATURITIES, config.SVI_REGIMES[regime], seed)
        W = np.array([core.svi_total_variance(fine_k, *pp) for pp in p])
        v = core.matrix_violations(W, fine_k)
        bad += (v["durrleman_violation_rate"] > 0) or (v["butterfly_violation_rate"] > 0) or (v["calendar_violation_rate"] > 0)
    check(f"ground truth arbitrage-free [{regime}]", bad == 0, f"({bad}/50 seeds with violations)")

# 2 + 3
torch.manual_seed(0)
worst_bfly, worst_cal, min_w = 0.0, 0.0, 1e9
for trial in range(20):
    net = core.HardConstrainedIVSNet(config.MATURITIES, hidden=config.HARD_HIDDEN)
    with torch.no_grad():
        for prm in net.parameters():
            prm.add_(torch.randn_like(prm) * (0.05 + 0.5 * trial / 20))
    fn = core.torch_w_fn(net)
    Ts = np.linspace(config.MATURITIES[0], config.MATURITIES[-1], 97)
    W = core.surface_matrix(fn, fine_k, Ts)
    h = fine_k[1] - fine_k[0]
    d2 = (W[:, 2:] - 2 * W[:, 1:-1] + W[:, :-2]) / h ** 2
    worst_bfly = min(worst_bfly, d2.min())
    worst_cal = min(worst_cal, (W[1:] - W[:-1]).min())
    min_w = min(min_w, W.min())
check("hard net convex in k for perturbed weights", worst_bfly > -1e-6, f"(min w'' = {worst_bfly:.2e})")
check("hard net non-decreasing in T (dense T grid) for perturbed weights", worst_cal > -1e-9, f"(min dW = {worst_cal:.2e})")
check("hard net positive", min_w > 0, f"(min w = {min_w:.2e})")

# 4
p = core.make_regime_svi_params(config.MATURITIES, config.SVI_REGIMES["normal"], 0)
gt = core.generate_ground_truth_grid(k_grid, config.MATURITIES, p)
ok = True
for seed in range(20):
    tr, mask = core.corrupt_data(gt, config.CORRUPTION_NOISE_STD, config.CORRUPTION_KEEP_FRAC, seed)
    for T in config.MATURITIES:
        m = np.isclose(gt[:, 1], T)
        ok &= mask[m].sum() >= config.MIN_POINTS_PER_MATURITY
        if config.KEEP_ENDPOINTS:
            ok &= bool(mask[m][0] and mask[m][-1])
check("masking: >= min points per maturity and both wing strikes kept", bool(ok))

# 5
tr, mask, dist, target = core.make_stress_train_set(gt, config.MATURITIES, config.STRESS_MATURITY_INDEX,
                                                    config.STRESS_DELTA, config.STRESS_K_BAND,
                                                    config.CORRUPTION_NOISE_STD, config.CORRUPTION_KEEP_FRAC, 0)
j = config.STRESS_MATURITY_INDEX
rows_j = np.where(np.isclose(gt[:, 1], config.MATURITIES[j]) & (np.abs(gt[:, 0]) <= config.STRESS_K_BAND))[0]
rows_p = np.where(np.isclose(gt[:, 1], config.MATURITIES[j - 1]) & (np.abs(gt[:, 0]) <= config.STRESS_K_BAND))[0]
check("stress set: distorted data lies below previous maturity's TRUE slice",
      bool(np.all(target[rows_j] < gt[rows_p, 2])), f"(max gap {np.max(target[rows_j] - gt[rows_p, 2]):.4f})")
check("stress set: same mask as clean set",
      bool(np.array_equal(mask, core.corrupt_data(gt, config.CORRUPTION_NOISE_STD, config.CORRUPTION_KEEP_FRAC, 0)[1])))

print("\nALL CHECKS PASSED" if not failed else f"\nFAILED: {failed}")
sys.exit(1 if failed else 0)

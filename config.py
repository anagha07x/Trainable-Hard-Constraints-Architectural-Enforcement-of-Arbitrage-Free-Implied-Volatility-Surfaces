"""
Configuration for the arbitrage-free IV surface calibration experiments (v2).

Everything the scripts need lives here. Change a number here and every script
picks it up. See README notes at the bottom of each script for what it writes.
"""

# ---------------------------------------------------------------------------
# SVI parameter regimes (ground-truth generator)
# ---------------------------------------------------------------------------
# a = a_base + a_maturity_scale * sqrt(T); b, rho, m, sigma ~ U(range).
# v2 change: each drawn slice is REJECTION-SAMPLED so the ground truth is
# arbitrage-free by construction (Durrleman g(k) >= 0, w > 0, and each slice
# lies above the previous maturity's slice). v1 silently produced Durrleman
# violations in the "stressed" regime and patched calendar order with a max().
SVI_REGIMES = {
    "normal": {
        "a_base": 0.02, "a_maturity_scale": 0.04,
        "b_range": (0.15, 0.20), "rho_range": (-0.40, -0.30),
        "m_range": (0.00, 0.02), "sigma_range": (0.10, 0.15),
    },
    "high_skew": {
        "a_base": 0.02, "a_maturity_scale": 0.04,
        "b_range": (0.18, 0.25), "rho_range": (-0.85, -0.65),
        "m_range": (0.00, 0.03), "sigma_range": (0.08, 0.12),
    },
    "stressed": {
        "a_base": 0.06, "a_maturity_scale": 0.08,
        "b_range": (0.30, 0.45), "rho_range": (-0.95, -0.75),
        "m_range": (-0.02, 0.04), "sigma_range": (0.05, 0.10),
    },
}
ACTIVE_REGIMES = ["normal", "high_skew", "stressed"]

# ---------------------------------------------------------------------------
# Grid / data
# ---------------------------------------------------------------------------
MATURITIES = [0.1, 0.25, 0.5, 1.0, 2.0]
K_GRID_MIN, K_GRID_MAX = -0.5, 0.5
K_GRID_POINTS = 25
FINE_K_GRID_POINTS = 201        # violation-check grid in k (spacing 0.005)
DENSE_T_POINTS = 81             # continuous-in-T models are also checked between maturities

CORRUPTION_NOISE_STD = 0.02
CORRUPTION_KEEP_FRAC = 0.35     # v2: applied PER MATURITY (stratified), see corrupt_data
MIN_POINTS_PER_MATURITY = 6     # guarantees every SVI slice has data (no skipped slices)
KEEP_ENDPOINTS = True           # the extreme strikes (k=-0.5, +0.5) of every maturity are always observed

# Tolerances used by the violation counters (same for every model)
BUTTERFLY_TOL = 1e-4            # second derivative of w in k below -tol counts as violation
DURRLEMAN_TOL = 1e-4            # Durrleman g(k) below -tol counts as violation
CALENDAR_TOL = 1e-4             # w(T_j) - w(T_{j-1}) below -tol counts as violation

# ---------------------------------------------------------------------------
# Targeted stress test (v2: actually induces a calendar violation)
# ---------------------------------------------------------------------------
# In the stressed training set, at maturity index STRESS_MATURITY_INDEX the
# observations with |k| <= STRESS_K_BAND are replaced by
#     w_true(k, T_{j-1}) - STRESS_DELTA
# i.e. the data at the later maturity sits BELOW the earlier maturity's true
# slice by STRESS_DELTA. Same mask/noise as the clean training set.
STRESS_MATURITY_INDEX = 2       # T = 0.5 vs T = 0.25
STRESS_DELTA = 0.03
STRESS_K_BAND = 0.30

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]          # evaluation seeds (paper numbers)
DEV_SEEDS = [100, 101]           # hyper-parameter development only; never report these

# ---------------------------------------------------------------------------
# Training hyper-parameters (selected on DEV_SEEDS only)
# ---------------------------------------------------------------------------
SOFT_PENALTY_HIDDEN = 128
SOFT_PENALTY_EPOCHS = 3000
SOFT_PENALTY_LR = 2e-3
SOFT_PENALTY_LAMBDA = 0.3
SOFT_PENALTY_N_COLLOC = 512      # fresh random (k,T) collocation points per epoch

HARD_HIDDEN = 32
HARD_EPOCHS = 4000
HARD_LR = 1e-2
HARD_GRAD_CLIP = 10.0

DUGAS_HIDDEN = 32
DUGAS_EPOCHS = 4000
DUGAS_LR = 1e-2

SVI_N_STARTS = 10
SSVI_N_STARTS = 10

MODELS = ["svi", "ssvi", "soft_penalty", "hard_constrained", "hard_no_calib", "dugas_style"]

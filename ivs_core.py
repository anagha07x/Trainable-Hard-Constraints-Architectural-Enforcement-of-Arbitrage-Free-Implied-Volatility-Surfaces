"""
Core building blocks (v2) for the arbitrage-free IV surface experiments.

What changed vs. v1 (and why) -- see the list at the bottom of this docstring.

Models
------
* SVI (per-maturity, multi-start least squares)      -> `fit_svi_slice`, `SVISlices`
* SSVI (global surface fit, monotone theta)          -> `fit_ssvi`, `SSVISurface`
* Soft-penalty MLP (Ackerer-style)                   -> `SoftPenaltyMLP`, `train_soft_penalty`
* Hard-constrained ICNN + monotone-in-T increments   -> `HardConstrainedIVSNet`, `train_hard_constrained`
* Dugas-style sign-restricted MLP (naive-init ablation) -> `DugasStyleNet`, `train_dugas_style`
* Hard-constrained WITHOUT the data-driven output calibration ("hard_no_calib", init ablation)

Hard-constrained architecture (what is guaranteed, exactly)
-----------------------------------------------------------
    w(k,T) = softplus(ICNN_0(k)) + sum_i s_i(T) * softplus(ICNN_i(k))
with s_i(T) = clip((T - t_{i-1}) / (t_i - t_{i-1}), 0, 1), t = maturities.
  * w > 0 everywhere.
  * w is CONVEX in k for every T (non-negative combination of convex functions).
  * w is NON-DECREASING in T for every k (s_i non-decreasing, softplus >= 0),
    i.e. calendar-monotone for ALL T (not only the 5 training maturities).
  * Flat in T outside [t_0, t_last] (no extrapolation claim).
NOTE: convexity of TOTAL VARIANCE in k is a proxy for butterfly-freedom. The
exact condition is Durrleman's g(k) >= 0, which is NOT implied by convexity of
w. This code therefore reports BOTH convexity violations and Durrleman
violations for every model. Do not claim "no butterfly arbitrage" for the hard
model unless the Durrleman column is also ~0 in your run.

Changes vs v1
-------------
1. Ground truth is arbitrage-free by construction (rejection sampling + checks).
2. Stratified masking: every maturity keeps >= MIN_POINTS_PER_MATURITY points.
3. Hard model is continuous in T, has a positive output, calibrated output
   scale at init (v1 started ~10-100x too large -> RMSE ~0.125), bias in the
   first ICNN layer (v1 pinned every kink at k=0), no dead parameters.
4. Soft baseline penalises on fresh random collocation points over the whole
   (k, T) domain using autodiff (d2w/dk2 and dw/dT), as Ackerer et al. do,
   instead of only at the training points with a tiny finite-difference step.
5. Metrics: held-out RMSE, IV-vol RMSE, Durrleman g(k) violations, calendar
   violations between maturities for continuous models, violation magnitude,
   all computed in float64 and IDENTICALLY for every model (finite differences).
6. Baselines: multi-start SVI (no skipped slices), SSVI, Dugas-style ablation.
7. Targeted stress set that really induces a calendar inversion and is NOT
   duplicated into training.
"""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import least_squares

import config


# ---------------------------------------------------------------------------
# 1. Ground truth
# ---------------------------------------------------------------------------

def svi_total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_durrleman_g(k, a, b, rho, m, sigma):
    """Durrleman butterfly function g(k) for raw SVI (analytic derivatives). g>=0 <=> no butterfly arbitrage."""
    q = np.sqrt((k - m) ** 2 + sigma ** 2)
    w = svi_total_variance(k, a, b, rho, m, sigma)
    w1 = b * (rho + (k - m) / q)
    w2 = b * sigma ** 2 / q ** 3
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2


def make_regime_svi_params(maturities, regime_config, seed=0, max_tries=5000):
    """
    Draw one SVI slice per maturity from the regime's ranges, REJECTING draws that
    (i) give w <= 0, (ii) violate Durrleman g(k) >= 0 on a grid wider than the data
    range, or (iii) do not lie above the previous maturity's slice by >= 1e-3.
    """
    rng = np.random.default_rng(seed)
    a_base = regime_config["a_base"]
    a_scale = regime_config["a_maturity_scale"]
    k_chk = np.linspace(config.K_GRID_MIN - 0.25, config.K_GRID_MAX + 0.25, 601)

    params, prev_w = [], None
    for T in maturities:
        a = a_base + a_scale * np.sqrt(T)
        for _ in range(max_tries):
            b = rng.uniform(*regime_config["b_range"])
            rho = rng.uniform(*regime_config["rho_range"])
            m = rng.uniform(*regime_config["m_range"])
            sigma = rng.uniform(*regime_config["sigma_range"])
            w = svi_total_variance(k_chk, a, b, rho, m, sigma)
            if w.min() <= 0:
                continue
            if svi_durrleman_g(k_chk, a, b, rho, m, sigma).min() < 0:
                continue
            if prev_w is not None and (w - prev_w).min() < 1e-3:
                continue
            params.append((a, b, rho, m, sigma))
            prev_w = w
            break
        else:
            raise RuntimeError(f"Could not draw an arbitrage-free slice for T={T} (seed={seed}); "
                               f"loosen the regime ranges in config.py")
    return params


def generate_ground_truth_grid(k_grid, maturities, params):
    """(N,3) array of (k, T, w), ordered by maturity then k. Raises if calendar order is violated."""
    rows, prev_w = [], None
    for T, p in zip(maturities, params):
        w = svi_total_variance(k_grid, *p)
        if prev_w is not None and (w < prev_w).any():
            raise RuntimeError("Ground truth violates calendar monotonicity")
        prev_w = w
        rows.extend((kk, T, ww) for kk, ww in zip(k_grid, w))
    return np.array(rows)


def corrupt_data(grid, noise_std=0.02, keep_frac=0.35, seed=0):
    """
    Additive Gaussian noise on w + STRATIFIED sparse masking (per maturity).
    Returns (train_grid, mask) where mask marks which rows of `grid` were kept.
    """
    rng = np.random.default_rng(seed)
    noisy = grid.copy()
    noisy[:, 2] += rng.normal(0, noise_std, size=len(grid))
    mask = np.zeros(len(grid), dtype=bool)
    for T in np.unique(grid[:, 1]):
        idx = np.where(np.isclose(grid[:, 1], T))[0]
        n_keep = max(config.MIN_POINTS_PER_MATURITY, int(round(keep_frac * len(idx))))
        if config.KEEP_ENDPOINTS:
            # always observe the two extreme strikes of each maturity, so held-out
            # evaluation is interpolation (convex nets extrapolate unboundedly upward)
            mask[idx[0]] = mask[idx[-1]] = True
            interior = idx[1:-1]
            mask[rng.choice(interior, size=n_keep - 2, replace=False)] = True
        else:
            mask[rng.choice(idx, size=n_keep, replace=False)] = True
    return noisy[mask], mask


def make_stress_train_set(ground_truth, maturities, stress_idx, delta, k_band,
                          noise_std, keep_frac, seed):
    """
    Build a training set whose data at maturity `stress_idx` (for |k|<=k_band) is
    replaced by w_true(k, T_{j-1}) - delta, i.e. an induced calendar inversion,
    then corrupt with the SAME seed (=> same noise draw and same mask as the clean set).
    Returns (train_grid, mask, distorted_rows, distorted_target_full)
      distorted_rows : bool over full grid rows that were distorted
      distorted_target_full : (N,) w-values of the distorted (pre-noise) full grid
    """
    assert stress_idx >= 1
    Tj, Tp = maturities[stress_idx], maturities[stress_idx - 1]
    gt = ground_truth.copy()
    rows_j = np.where(np.isclose(gt[:, 1], Tj) & (np.abs(gt[:, 0]) <= k_band))[0]
    rows_p = np.where(np.isclose(gt[:, 1], Tp) & (np.abs(gt[:, 0]) <= k_band))[0]
    assert len(rows_j) == len(rows_p) and np.allclose(gt[rows_j, 0], gt[rows_p, 0])
    gt[rows_j, 2] = ground_truth[rows_p, 2] - delta
    distorted = np.zeros(len(gt), dtype=bool)
    distorted[rows_j] = True
    train, mask = corrupt_data(gt, noise_std=noise_std, keep_frac=keep_frac, seed=seed)
    return train, mask, distorted, gt[:, 2].copy()


# ---------------------------------------------------------------------------
# 2. Classical baselines: SVI (per slice) and SSVI (global)
# ---------------------------------------------------------------------------

def fit_svi_slice(k, w, n_starts=10, seed=0):
    """Multi-start bounded least squares; returns the best-cost parameter vector."""
    def resid(p):
        return svi_total_variance(k, *p) - w

    bounds = ([-1, 1e-4, -0.999, -1, 1e-4], [2, 2, 0.999, 1, 2])
    rng = np.random.default_rng(seed)
    starts = [[0.05, 0.2, -0.3, 0.0, 0.15]]
    for _ in range(n_starts - 1):
        starts.append([rng.uniform(0, 0.3), rng.uniform(0.05, 0.5), rng.uniform(-0.9, 0.9),
                       rng.uniform(-0.1, 0.1), rng.uniform(0.03, 0.3)])
    best = None
    for p0 in starts:
        try:
            res = least_squares(resid, p0, bounds=bounds)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        raise RuntimeError("SVI slice fit failed for all starts")
    return best.x


class SVISlices:
    """Per-maturity SVI; only defined at the fitted maturities."""
    continuous_in_T = False

    def __init__(self, maturities, params):
        self.maturities, self.params = list(maturities), params

    def __call__(self, k, T):
        k, T = np.asarray(k, float), np.asarray(T, float)
        out = np.full(k.shape, np.nan)
        for Tj, p in zip(self.maturities, self.params):
            m = np.isclose(T, Tj)
            if m.any():
                out[m] = svi_total_variance(k[m], *p)
        if np.isnan(out).any():
            raise ValueError("SVISlices queried at an unfitted maturity")
        return out


def fit_svi_baseline(train_grid, maturities, seed=0):
    params = []
    for j, T in enumerate(maturities):
        m = np.isclose(train_grid[:, 1], T)
        params.append(fit_svi_slice(train_grid[m, 0], train_grid[m, 2],
                                    n_starts=config.SVI_N_STARTS, seed=seed * 100 + j))
    return SVISlices(maturities, params)


def _ssvi_w(k, theta, rho, eta, gamma):
    phi = eta * theta ** (-gamma)
    return 0.5 * theta * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2))


class SSVISurface:
    """Gatheral-Jacquier SSVI with power-law phi(theta)=eta*theta^-gamma; theta non-decreasing in T."""
    continuous_in_T = False

    def __init__(self, maturities, x):
        self.maturities = list(maturities)
        n = len(maturities)
        self.theta = np.cumsum(x[:n])
        self.rho, self.eta, self.gamma = x[n], x[n + 1], x[n + 2]

    def __call__(self, k, T):
        k, T = np.asarray(k, float), np.asarray(T, float)
        out = np.full(k.shape, np.nan)
        for Tj, th in zip(self.maturities, self.theta):
            m = np.isclose(T, Tj)
            if m.any():
                out[m] = _ssvi_w(k[m], th, self.rho, self.eta, self.gamma)
        if np.isnan(out).any():
            raise ValueError("SSVISurface queried at an unfitted maturity")
        return out


def fit_ssvi(train_grid, maturities, n_starts=10, seed=0):
    n = len(maturities)
    tidx = np.array([int(np.argmin(np.abs(np.array(maturities) - t))) for t in train_grid[:, 1]])
    k, w = train_grid[:, 0], train_grid[:, 2]

    def resid(x):
        theta = np.cumsum(x[:n])
        return _ssvi_w(k, theta[tidx], x[n], x[n + 1], x[n + 2]) - w

    lo = [1e-4] + [0.0] * (n - 1) + [-0.999, 1e-3, 0.0]
    hi = [3.0] + [3.0] * (n - 1) + [0.999, 20.0, 1.0]
    rng = np.random.default_rng(seed)
    starts = []
    for _ in range(n_starts):
        th = np.sort(rng.uniform(0.02, 0.6, n))
        inc = np.concatenate([[th[0]], np.diff(th)])
        starts.append(np.concatenate([inc, [rng.uniform(-0.9, 0.0), rng.uniform(0.2, 2.0), rng.uniform(0.1, 0.9)]]))
    best = None
    for x0 in starts:
        try:
            res = least_squares(resid, np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9), bounds=(lo, hi))
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        raise RuntimeError("SSVI fit failed for all starts")
    return SSVISurface(maturities, best.x)


# ---------------------------------------------------------------------------
# 3. Soft-penalty MLP baseline (Ackerer-style)
# ---------------------------------------------------------------------------

class SoftPenaltyMLP(nn.Module):
    continuous_in_T = True

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, k, T):
        return self.net(torch.stack([k, T], dim=-1)).squeeze(-1)


def arbitrage_penalty(model, k, T):
    """Autodiff penalty: relu(-d2w/dk2) (butterfly proxy) + relu(-dw/dT) (calendar), on collocation points."""
    k = k.clone().requires_grad_(True)
    T = T.clone().requires_grad_(True)
    w = model(k, T)
    dw_dk, dw_dT = torch.autograd.grad(w.sum(), [k, T], create_graph=True)
    d2w_dk2 = torch.autograd.grad(dw_dk.sum(), k, create_graph=True)[0]
    return F.relu(-d2w_dk2).mean() + F.relu(-dw_dT).mean()


def _tensors(train_grid):
    return (torch.tensor(train_grid[:, 0], dtype=torch.float32),
            torch.tensor(train_grid[:, 1], dtype=torch.float32),
            torch.tensor(train_grid[:, 2], dtype=torch.float32))


def train_soft_penalty(train_grid, maturities, epochs=None, lam=None, lr=None, seed=0, verbose=False):
    epochs = epochs or config.SOFT_PENALTY_EPOCHS
    lam = config.SOFT_PENALTY_LAMBDA if lam is None else lam
    lr = lr or config.SOFT_PENALTY_LR
    torch.manual_seed(seed)
    model = SoftPenaltyMLP(hidden=config.SOFT_PENALTY_HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    k, T, w = _tensors(train_grid)
    gen = torch.Generator().manual_seed(seed + 12345)
    kmin, kmax = config.K_GRID_MIN, config.K_GRID_MAX
    tmin, tmax = float(min(maturities)), float(max(maturities))
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        fit_loss = ((model(k, T) - w) ** 2).mean()
        kc = kmin + (kmax - kmin) * torch.rand(config.SOFT_PENALTY_N_COLLOC, generator=gen)
        Tc = tmin + (tmax - tmin) * torch.rand(config.SOFT_PENALTY_N_COLLOC, generator=gen)
        pen = arbitrage_penalty(model, kc, Tc)
        (fit_loss + lam * pen).backward()
        opt.step()
        history.append(fit_loss.item())
        if verbose and epoch % 500 == 0:
            print(f"  [soft] epoch {epoch:4d} fit={fit_loss.item():.6f} pen={pen.item():.6f}")
    return model, history


# ---------------------------------------------------------------------------
# 4. Hard-constrained architecture
# ---------------------------------------------------------------------------

def inv_softplus(x):
    x = max(float(x), 1e-6)
    return math.log(math.expm1(x))


class ICNNBlock(nn.Module):
    """1-D input-convex network (Amos et al. 2017): output is convex in k."""

    def __init__(self, hidden=32, depth=2):
        super().__init__()
        self.depth = depth
        self.y_layers = nn.ModuleList(
            [nn.Linear(1, hidden, bias=(i == 0)) for i in range(depth)])
        self.z_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=True) for _ in range(depth - 1)])
        self.out_z = nn.Linear(hidden, 1, bias=True)
        self.out_y = nn.Linear(1, 1, bias=False)
        # z-path raw weights start very negative so softplus(weight) ~ 0.05: prevents the
        # multiplicative blow-up across width/depth that plagues naive positive-weight nets.
        for layer in list(self.z_layers) + [self.out_z]:
            nn.init.normal_(layer.weight, mean=-3.0, std=0.1)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.y_layers[0].weight, mean=0.0, std=3.0)   # diverse kink slopes
        nn.init.normal_(self.y_layers[0].bias, mean=0.0, std=0.5)     # diverse kink locations
        for layer in list(self.y_layers)[1:]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.out_y.weight, mean=0.0, std=0.05)

    def forward(self, k):
        k_in = k.unsqueeze(-1)
        z = F.softplus(self.y_layers[0](k_in))
        for i in range(1, self.depth):
            zl = self.z_layers[i - 1]
            z = F.softplus(F.linear(z, F.softplus(zl.weight), zl.bias) + self.y_layers[i](k_in))
        out = F.linear(z, F.softplus(self.out_z.weight), self.out_z.bias) + self.out_y(k_in)
        return out.squeeze(-1)

    @torch.no_grad()
    def calibrate_mean(self, target_preactivation):
        """Shift the output bias so the mean pre-activation over the k-domain equals the target."""
        kk = torch.linspace(config.K_GRID_MIN, config.K_GRID_MAX, 101)
        self.out_z.bias += (target_preactivation - self.forward(kk).mean())


class HardConstrainedIVSNet(nn.Module):
    """See module docstring for the exact guarantees."""
    continuous_in_T = True

    def __init__(self, maturities, hidden=32):
        super().__init__()
        self.register_buffer("knots", torch.tensor(list(maturities), dtype=torch.float32))
        self.base = ICNNBlock(hidden=hidden)
        self.increments = nn.ModuleList([ICNNBlock(hidden=hidden) for _ in range(len(maturities) - 1)])

    def forward(self, k, T):
        knots = self.knots.to(k.dtype)
        w = F.softplus(self.base(k))
        for i, blk in enumerate(self.increments, start=1):
            s = torch.clamp((T - knots[i - 1]) / (knots[i] - knots[i - 1]), 0.0, 1.0)
            w = w + s * F.softplus(blk(k))
        return w

    def init_output_scale(self, train_grid, maturities):
        """Data-driven init: make w start at the observed per-maturity mean level (v1 started ~10-100x too high)."""
        means = []
        for T in maturities:
            m = np.isclose(train_grid[:, 1], T)
            means.append(train_grid[m, 2].mean())
        means = np.maximum.accumulate(np.array(means))
        self.base.calibrate_mean(inv_softplus(means[0]))
        for i, blk in enumerate(self.increments, start=1):
            blk.calibrate_mean(inv_softplus(max(means[i] - means[i - 1], 1e-3)))


def train_hard_constrained(train_grid, maturities, epochs=None, lr=None, seed=0, verbose=False,
                           calibrate=True):
    epochs = epochs or config.HARD_EPOCHS
    lr = lr or config.HARD_LR
    torch.manual_seed(seed)
    model = HardConstrainedIVSNet(maturities, hidden=config.HARD_HIDDEN)
    if calibrate:
        model.init_output_scale(train_grid, maturities)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    k, T, w = _tensors(train_grid)
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = ((model(k, T) - w) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.HARD_GRAD_CLIP)
        opt.step()
        sched.step()
        history.append(loss.item())
        if verbose and epoch % 500 == 0:
            print(f"  [hard] epoch {epoch:4d} fit={loss.item():.6f}")
    return model, history


# ---------------------------------------------------------------------------
# 5. Dugas-style ablation: all weights sign-restricted via softplus, default init
# ---------------------------------------------------------------------------

class DugasStyleNet(nn.Module):
    """
    Weight-sign-restricted MLP in the spirit of Dugas et al. (adapted to total variance):
    inputs (k, -k, T), every weight passed through softplus, softplus activations.
    Convex in k and non-decreasing in T by construction, but with standard (naive)
    initialisation and no output calibration -- this is the trainability ablation.
    """
    continuous_in_T = True

    def __init__(self, hidden=32, depth=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3 if i == 0 else hidden, hidden) for i in range(depth)])
        self.out = nn.Linear(hidden, 1)

    def forward(self, k, T):
        x = torch.stack([k, -k, T], dim=-1)
        for layer in self.layers:
            x = F.softplus(F.linear(x, F.softplus(layer.weight), layer.bias))
        return F.linear(x, F.softplus(self.out.weight), self.out.bias).squeeze(-1)


def train_dugas_style(train_grid, maturities, epochs=None, lr=None, seed=0, verbose=False):
    epochs = epochs or config.DUGAS_EPOCHS
    lr = lr or config.DUGAS_LR
    torch.manual_seed(seed)
    model = DugasStyleNet(hidden=config.DUGAS_HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    k, T, w = _tensors(train_grid)
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = ((model(k, T) - w) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        history.append(loss.item())
        if verbose and epoch % 500 == 0:
            print(f"  [dugas] epoch {epoch:4d} fit={loss.item():.6f}")
    return model, history


# ---------------------------------------------------------------------------
# 6. Evaluation (identical finite-difference float64 metrics for every model)
# ---------------------------------------------------------------------------

def rmse(pred, target):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(target)) ** 2)))


def torch_w_fn(model):
    """float64 numpy callable w(k, T) from a torch model (copy; original untouched)."""
    m = copy.deepcopy(model).double().eval()

    def fn(k, T):
        with torch.no_grad():
            return m(torch.tensor(np.asarray(k, float), dtype=torch.float64),
                     torch.tensor(np.asarray(T, float), dtype=torch.float64)).numpy()
    fn.continuous_in_T = getattr(model, "continuous_in_T", False)
    return fn


def surface_matrix(w_fn, fine_k, Ts):
    return np.array([w_fn(fine_k, np.full_like(fine_k, T)) for T in Ts])


def matrix_violations(W, fine_k):
    """
    W: (nT, nk) total variance, rows ordered by increasing T.
    Returns violation rates and max calendar violation magnitude.
    Butterfly (convexity proxy): w'' < -tol. Durrleman: g < -tol (needs w>0; w<=0 counts as violation).
    """
    h = fine_k[1] - fine_k[0]
    w0, wm, wp = W[:, 1:-1], W[:, :-2], W[:, 2:]
    k_in = fine_k[1:-1][None, :]
    d1 = (wp - wm) / (2 * h)
    d2 = (wp - 2 * w0 + wm) / h ** 2
    butterfly = d2 < -config.BUTTERFLY_TOL
    nonpos = W <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1 - k_in * d1 / (2 * w0)) ** 2 - (d1 ** 2 / 4) * (1 / w0 + 0.25) + d2 / 2
    durr = (~np.isfinite(g)) | (g < -config.DURRLEMAN_TOL) | (w0 <= 0)
    diff = W[1:] - W[:-1]
    calendar = diff < -config.CALENDAR_TOL
    return {
        "butterfly_violation_rate": float(butterfly.mean()),
        "durrleman_violation_rate": float(durr.mean()),
        "nonpositive_rate": float(nonpos.mean()),
        "calendar_violation_rate": float(calendar.mean()),
        "calendar_max_violation": float(max(0.0, -diff.min())),
    }


def evaluate_surface(w_fn, ground_truth, mask, maturities, extra_rows=None):
    """
    Returns a flat dict of metrics. RMSEs are vs the CLEAN ground truth.
      rmse_all      : all 125 grid points
      rmse_heldout  : grid points NOT used for training (mask==False)
      rmse_train_pts: grid points used for training (vs clean truth)
      iv_rmse_all   : RMSE in implied-vol units (vol points), sqrt(w/T)
    extra_rows: optional dict name -> (bool row mask, target array) for additional RMSEs.
    """
    pred = w_fn(ground_truth[:, 0], ground_truth[:, 1])
    truth = ground_truth[:, 2]
    T = ground_truth[:, 1]
    out = {
        "rmse_all": rmse(pred, truth),
        "rmse_heldout": rmse(pred[~mask], truth[~mask]),
        "rmse_train_pts": rmse(pred[mask], truth[mask]),
        "iv_rmse_all": rmse(np.sqrt(np.maximum(pred, 1e-12) / T), np.sqrt(truth / T)),
    }
    if extra_rows:
        for name, (rows, target) in extra_rows.items():
            out[name] = rmse(pred[rows], target[rows])

    fine_k = np.linspace(config.K_GRID_MIN, config.K_GRID_MAX, config.FINE_K_GRID_POINTS)
    out.update(matrix_violations(surface_matrix(w_fn, fine_k, maturities), fine_k))

    dense = {"butterfly_violation_rate_denseT": np.nan, "durrleman_violation_rate_denseT": np.nan,
             "calendar_violation_rate_denseT": np.nan, "calendar_max_violation_denseT": np.nan}
    if getattr(w_fn, "continuous_in_T", False):
        Ts = np.linspace(maturities[0], maturities[-1], config.DENSE_T_POINTS)
        v = matrix_violations(surface_matrix(w_fn, fine_k, Ts), fine_k)
        dense = {"butterfly_violation_rate_denseT": v["butterfly_violation_rate"],
                 "durrleman_violation_rate_denseT": v["durrleman_violation_rate"],
                 "calendar_violation_rate_denseT": v["calendar_violation_rate"],
                 "calendar_max_violation_denseT": v["calendar_max_violation"]}
    out.update(dense)
    return out


def fit_model(name, train_grid, maturities, seed):
    """Fit one named model. Returns (w_fn, info) where info has final_train_mse and train_time_s."""
    t0 = time.time()
    hist = None
    if name == "svi":
        w_fn = fit_svi_baseline(train_grid, maturities, seed=seed)
    elif name == "ssvi":
        w_fn = fit_ssvi(train_grid, maturities, n_starts=config.SSVI_N_STARTS, seed=seed)
    elif name == "soft_penalty":
        model, hist = train_soft_penalty(train_grid, maturities, seed=seed)
        w_fn = torch_w_fn(model)
    elif name == "hard_constrained":
        model, hist = train_hard_constrained(train_grid, maturities, seed=seed)
        w_fn = torch_w_fn(model)
    elif name == "hard_no_calib":
        model, hist = train_hard_constrained(train_grid, maturities, seed=seed, calibrate=False)
        w_fn = torch_w_fn(model)
    elif name == "dugas_style":
        model, hist = train_dugas_style(train_grid, maturities, seed=seed)
        w_fn = torch_w_fn(model)
    else:
        raise ValueError(name)
    if not hasattr(w_fn, "continuous_in_T"):
        w_fn.continuous_in_T = getattr(w_fn, "continuous_in_T", False)
    info = {
        "final_train_mse": float(np.mean(hist[-20:])) if hist else float("nan"),
        "first_train_mse": float(hist[0]) if hist else float("nan"),
        "train_time_s": time.time() - t0,
    }
    return w_fn, info

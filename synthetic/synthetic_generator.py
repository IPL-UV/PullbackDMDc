"""
Synthetic ensemble for testing PullbackDMDc: generation, fitting and diagnostics (monthly step, record 1850-2014).

System (M = 20 latitudes, -90 to 90 degrees, poles included, unit-norm patterns, T = 1980 monthly steps in the record):
    x(t) = A x(t-1) + B y(t) + xi(t),   A = W Lambda W^-1,   data = x^(f) + process noise response + measurement noise eps
    W columns: slow mode, two fast modes (ENSO-like pair), 17 noise modes. Lambda: slow real eigenvalue, fast pair (decay time, period), noise modes with small eigenvalues.
    Process noise is diagonal in the modal basis; y(t) is the mean-centered forcing with its history (spin-up).

Functions
    make_setup(spin_up, seed, n, delta, alpha)
        Builds W, W_inv, the forcing pattern b0 = [1, ..., 1], the forcing time axis and the mean-centered forcing y (and y0 = -record mean).
        delta: depth of the Gaussian dip of the forcing around 1965.
        alpha: overlap of the complement (noise) modes with the slow mode, q_k(alpha) = alpha w1 + sqrt(1 - alpha^2) q_k (0: orthogonal).
    generate_ensemble(n_members, tau1, variances, delta, alpha, spin_up_factor, w_seed, eig_seed, tau_pair, period_pair, lam_noise_max, rng)
        Simulates the ensemble (independent noise, common forcing) after a spin-up of spin_up_factor * max(tau1, tau_pair) years.
        rng: seed or numpy Generator of the noise realizations (None: unseeded).
        variances = (V_f, V_1, V_p, V_c, V_eps), up to a common factor: the variance of the forced response, of the slow, fast-pair
        and noise-mode internal variability and of the measurement noise; theoretical SNR = V_f / (V_1 + V_p + V_c + V_eps).
        The forced variance V_f = sum_i Var_t x_i^(f)(t) over the record is fixed by the system; the other components are set relative to it.
        B = b0 / c_B, with c_B chosen so that the quasi-equilibrium forced state at the start of the spin-up has unit norm (an
        implementation detail: every score is invariant to the scale of B).
        Returns a dict: data, forced, obs_noise (none of them centered), data_mean, years, y, years_full, y_full, y0, A, W, W_inv, Lambda_R, B, b_scale, modal_var, noise_std, sigma_eps, Vf, snr, spin_up.
    fit_pullback_ensemble(data, y, y_full, lag)
        Fits the repo's PullbackDMDc to each member separately (full truncation, transition_time = whole spin-up).
        Returns the fitted models and the estimated forced responses (n_members, T, M) from model.predict().

Plots: plot_generator_diagnostics.py (also the usage example). Ablation studies on top of generate_ensemble: ablations.py.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from utils.pullback_dmdc import PullbackDMDc

def make_setup(spin_up, seed=22, n=20, delta=0.15, alpha=0.0):
    """W, W_inv, forcing pattern b0, mean-centered forcing y on monthly steps from 1850 - spin_up to Dec 2014, and y0.

    spin_up: spin-up length in years (monthly steps, so 12 * spin_up months before January 1850).
    delta: depth of the Gaussian dip of the forcing around 1965 (delta = 0 gives a monotone forcing).
    alpha: overlap of every complement mode with the slow mode w1, in [0, 1): q_k(alpha) = alpha w1 + sqrt(1 - alpha^2) q_k,
           where q_k is the orthonormal complement mode at alpha = 0. The slow and fast patterns do not move, and the
           q_k(alpha) keep unit norm because q_k is orthogonal to w1; alpha = 0 gives the default W.
    Returns
      W     : (n, n), unit-norm columns [slow, fast1, fast2, 17 random complement modes q_k(alpha)]
      W_inv : (n, n), inverse of W (np.linalg.inv)                                               # NEW
      B     : (n,), forcing pattern b0 = [1, ..., 1] (not scaled)
      years : (n_spin + T,), calendar years of the monthly steps, years[k] = 1850 - spin_up + k/12
      y     : (n_spin + T,), forcing F(t) minus its mean over the record 1850-2014 (T = 1980 monthly steps)
      y0    : float, -mean: the constant level of y before the rise (y -> y0 as t -> -infinity),
              independent of spin_up
    """
    # grid and patterns (radians, poles included)
    phi = np.linspace(-np.pi / 2, np.pi / 2, n)
    unit = lambda v: v / np.linalg.norm(v)
    slow = np.cos(phi - np.pi / 4)
    fast1 = np.sinc(2 * phi)
    fast2 = (0.5 - np.cos(10 * phi)) / (0.5 + np.abs(10 * phi))
    S = np.column_stack([unit(slow), unit(fast1), unit(fast2)])

    # W: complement of the 3 structured modes, Haar-random rotation with the seed
    Q, _ = np.linalg.qr(S, mode="complete")
    Qp = Q[:, 3:]
    rng = np.random.default_rng(seed)
    R, T = np.linalg.qr(rng.standard_normal((n - 3, n - 3)))
    Qp = Qp @ (R * np.sign(np.diag(T)))
    Qp = alpha * S[:, :1] + np.sqrt(1 - alpha ** 2) * Qp                       # complement modes rotated toward w1 (overlap alpha)

    W = np.column_stack([S, Qp])
    W_inv = np.linalg.inv(W)                                                   

    # forcing pattern b0, uniform: 20 values on the grid (scaled in generate_ensemble)
    B = np.ones(n)

    # forcing: F(t) = exp((t - 2014)/65) - delta exp(-0.5 ((t - 1965)/15)^2), t in calendar years;
    # the constant c is not needed because the series is mean-centered over the record 1850-2014
    n_spin = int(round(12 * spin_up))
    n_rec = 12 * (2014 - 1850 + 1)                       # 1980 monthly steps, Jan 1850 to Dec 2014
    years = 1850.0 + np.arange(-n_spin, n_rec) / 12.0
    F = np.exp((years - 2014.0) / 65.0) - delta * np.exp(-0.5 * ((years - 1965.0) / 15.0) ** 2)
    y0 = -F[n_spin:].mean()                              # record mean (does not depend on spin_up)
    y = F + y0
    return W, W_inv, B, years, y, y0


def generate_ensemble(n_members, tau1, variances=(1, 1, 1, 1, 0), delta = 0.15, alpha = 0.0, spin_up_factor=100,
                      w_seed=22, eig_seed=20, tau_pair=1.5, period_pair=4., lam_noise_max=0.1, rng=None):
    """Ensemble of realizations of the synthetic system (monthly step, record 1850-2014).

    n_members       : number of ensemble members (independent noise, common forcing)
    tau1            : decay time of the slow mode, years
    variances       : variance components (V_f, V_1, V_p, V_c, V_eps), up to a common factor (variances[0] > 0):
                      the forced variance V_f = sum_i Var_t x_i^(f)(t) (sum over grid points of the time variance of the
                      forced response over the record), the internal variance of the slow mode (V_1), of the fast pair
                      (shared by its two components, V_p / 2 each), of the 17 noise modes (shared, V_c / (M-3) each),
                      and of the measurement noise (white, independent at each grid node, time step and member,
                      V_eps = M sigma_eps^2). Each component is set as variances[k] / variances[0] * V_f.
                      The theoretical SNR = V_f / (V_1 + V_p + V_c + V_eps).
    delta           : depth of the Gaussian dip of the forcing around 1965 (delta = 0 gives a monotone forcing)
    alpha           : overlap of the complement modes with the slow mode (see make_setup; 0: orthogonal)
    spin_up_factor  : spin-up length in units of the slowest decay time (max(tau1, tau_pair))
    w_seed, eig_seed: seeds of the noise modes and noise eigenvalues
    tau_pair, period_pair : decay time (yr) and period (yr) of the fast (ENSO-like) pair
    rng             : seed or numpy Generator of the noise realizations (None: unseeded)
    Returns a dict (record = Jan 1850 to Dec 2014, T = 1980 monthly steps, M = 20), including the theoretical SNR ('snr').
    data = forced response + process noise response + obs_noise, none of them centered (the forcing y is centered over the record).
    """
    M = 20
    rng = np.random.default_rng(rng)
    v = np.asarray(variances, dtype=float)
    v = v / v[0]                                                               # components relative to V_f

    # patterns, centered forcing with its history (spin-up in years)
    spin_up = spin_up_factor * max(tau1, tau_pair)
    W, W_inv, b, years, y, y0 = make_setup(spin_up, seed=w_seed, delta=delta, alpha=alpha)

    n_spin = int(round(12 * spin_up))
    T = len(y) - n_spin                                                        # record length (monthly steps)

    # eigenvalues at the monthly step and propagator
    lam1 = np.exp(-1 / (12 * tau1))
    rho, theta = np.exp(-1 / (12 * tau_pair)), 2 * np.pi / (period_pair * 12)
    lam_n = np.random.default_rng(eig_seed).uniform(0, lam_noise_max, M - 3)
    L = np.zeros((M, M))
    L[0, 0] = lam1
    c, s = rho * np.cos(theta), rho * np.sin(theta)
    L[1:3, 1:3] = [[c, s], [-s, c]]
    L[3:, 3:] = np.diag(lam_n)
    A = W @ L @ W_inv

    # scaling of B: unit norm of the quasi-equilibrium forced state at the start of the spin-up
    b_scale = np.abs(y0) * np.linalg.norm(np.linalg.solve(np.eye(M) - A, b))
    B = b / b_scale

    # exact forced response (common to all members), full spin-up from zero
    xf, forced = np.zeros(M), np.empty((T, M))
    for i, yi in enumerate(y):
        xf = A @ xf + B * yi
        if i >= n_spin:
            forced[i - n_spin] = xf
    Vf = forced.var(axis=0).sum()

    # modal variances and measurement noise from the variance components relative to V_f, noise std from the decay rates
    s2 = np.r_[v[1] * Vf, v[2] * Vf / 2, v[2] * Vf / 2, np.full(M - 3, v[3] * Vf / (M - 3))]
    sigma_eps = np.sqrt(v[4] * Vf / M)                                         # measurement noise std, the same at every grid node
    snr = Vf / (s2.sum() + M * sigma_eps ** 2)                                 # theoretical SNR (internal + measurement noise)
    sigma = np.sqrt(s2 * (1 - np.r_[lam1, rho, rho, lam_n] ** 2))

    # ensemble: the full system from zero over spin-up + record
    x, data = np.zeros((n_members, M)), np.empty((n_members, T, M))
    for i, yi in enumerate(y):
        x = x @ A.T + B * yi + (sigma * rng.standard_normal((n_members, M))) @ W.T
        if i >= n_spin:
            data[:, i - n_spin] = x

    obs = sigma_eps * rng.standard_normal(data.shape)       # measurement noise, added to the process data
    data = data + obs                                         # measured data, not centered
    data_mean = data.mean(axis=1, keepdims=True)              # record mean of each member (diagnostic)
    return dict(data=data, forced=forced, obs_noise=obs, data_mean=data_mean[:, 0, :],
                years=years[n_spin:], y=y[n_spin:], years_full=years, y_full=y, y0=y0,
                A=A, W=W, W_inv=W_inv, Lambda_R=L, B=B, b_scale=b_scale,
                modal_var=s2, noise_std=sigma, sigma_eps=sigma_eps, Vf=Vf, snr=snr, spin_up=spin_up)



def fit_pullback_ensemble(data, y, y_full, lag=3):
    """Fit the repo's PullbackDMDc to every ensemble member (each member as a single realization).

    data   : (n_members, T, M) ensemble data over the record
    y      : (T,) forcing over the record (mean-centered over the record)
    y_full : (n_spin + T,) forcing over spin-up + record, same units; the whole spin-up is given to the method
    lag    : lag of the fit in months (3 in the paper)
    Full truncation (M EOFs) and transition_time = n_spin = len(y_full) - T.
    Returns the list of fitted models (compute_modes already called) and forced_est (n_members, T, M) =
    forced PCs @ EOFs + data mean (model.predict())."""
    n_members, T, M = data.shape
    n_spin = len(y_full) - T
    if not np.allclose(y_full[-T:], y):
        raise ValueError("y must be the last T values of y_full")
    short_forcings, long_forcings = y[:, None], y_full[:, None]
    models, forced_est = [], np.empty_like(data)
    for m in range(n_members):
        model = PullbackDMDc(truncation=M, lag=lag, transition_time=n_spin)
        model.fit(data[m], short_forcings, long_forcings)
        model.compute_modes()
        models.append(model)
        forced_est[m] = model.predict()
        #print(f"Member {m + 1}/{n_members}: slowest eigenvalue = {model.eigvals[0]:.4f}, ", flush=True)
    return models, forced_est

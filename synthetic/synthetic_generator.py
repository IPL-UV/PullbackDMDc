"""
Synthetic ensemble for testing PullbackDMDc: generation, fitting and diagnostics (monthly step, record 1850-2014).

System (M = 20 latitudes, -90 to 90 degrees, poles included, unit-norm patterns):
    x(t) = A x(t-1) + B y(t) + xi(t),   A = W Lambda W^-1,   data = forced + internal + measurement noise
    W columns: slow mode, two fast modes (ENSO-like pair), 17 noise modes. Lambda: slow real eigenvalue, fast pair (decay time, period), noise modes with small eigenvalues.
    Process noise is diagonal in the modal basis; y(t) is the mean-centered forcing with its history (spin-up).

Functions
    make_setup(spin_up, seed, n, b_rot, depth)
        Builds W, W_inv, the unit-norm forcing pattern B, the forcing time axis and the mean-centered forcing y (and y0 = -record mean).
        b_rot: fraction of the angle between the flat forcing pattern and the slow mode to rotate B toward the slow mode (0: flat, 1: B = slow mode).
        depth: amplitude of the Gaussian dip of the forcing around 1965.
    generate_ensemble(n_members, tau1, split, b_rot, depth, spin_up_factor, w_seed, eig_seed, tau_pair, period_pair, lam_noise_max, rng)
        Simulates the ensemble (independent noise, common forcing) after a spin-up of spin_up_factor * max(tau1, tau_pair) years.
        rng: seed or numpy Generator of the noise realizations (None: unseeded).
        split = (V_f, s1, sp, sc, v): fractions of the total variance in the forced response, slow, fast-pair, noise-mode internal variability and measurement noise; theoretical SNR = split[0] / sum(split[1:]).
        B is scaled so that the quasi-equilibrium forced state at the start of the spin-up has unit norm.
        Returns a dict: data, forced, internal, obs_noise (all zero-mean per member, data = forced + internal + obs_noise), data_mean, years, y, years_full, y_full, y0, A, W, W_inv, Lambda_R, B, b_scale, modal_var, noise_std, obs_std, Vf, snr, spin_up.
    fit_pullback_ensemble(data, y, y_full, lag)
        Fits the repo's PullbackDMDc to each member separately (full truncation, transition_time = whole spin-up).
        Returns the fitted models and the estimated forced responses (n_members, N, M) from model.predict().

Plots: plot_generator_diagnostics.py (also the usage example). Ablation studies on top of generate_ensemble: ablations.py.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from utils.pullback_dmdc import PullbackDMDc

def make_setup(spin_up, seed=22, n=20, b_rot=0.0,depth = 0.25):                      
    """W, W_inv, unit-norm unscaled B, mean-centered forcing y on monthly steps from 1850 - spin_up to Dec 2014, and y0.

    spin_up: spin-up length in years (monthly steps, so 12 * spin_up months before January 1850).
    b_rot: fraction of the angle between the forcing pattern b and the slow mode to rotate b toward the slow mode (0: no rotation, 1: b aligned with the slow mode).
    depth: amplitude of the Gaussian dip of the forcing around 1965 (depth = 0 gives a monotone forcing).
    Returns
      W     : (n, n), unit-norm columns [slow, fast1, fast2, 17 random orthonormal complement modes]
      W_inv : (n, n), inverse of W (np.linalg.inv)                                               # NEW
      B     : (n,), unit-norm forcing pattern b (not scaled)
      years : (N,), calendar years of the monthly steps, years[k] = 1850 - spin_up + k/12
      y     : (N,), forcing F(t) minus its mean over the record 1850-2014 (1980 monthly steps)
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

    W = np.column_stack([S, Qp])
    W_inv = np.linalg.inv(W)                                                   

    # forcing pattern b (unit norm), 20 values on the grid
    b = np.ones(n)/np.sqrt(n) # other option could be 0.2 + 0.05* np.cos(2 * phi)
    B = unit(b)

    #rotate b toward the slow mode by the angle (1 - b_rot) * theta0, where b_rot in [0, 1] is the fraction of the angle to rotate
    u = W[:, 0]                                                                # unit slow mode
    cos0 = np.clip(b @ u, -1, 1)
    theta0 = np.arccos(cos0)                                                   # angle between b and the slow mode
    q = b - cos0 * u                                                           # part of b orthogonal to the slow mode
    b = np.cos((1 - b_rot) * theta0) * u + np.sin((1 - b_rot) * theta0) * q / np.linalg.norm(q)
    B = unit(b)

    # forcing: F(t) = a exp((t - 2014)/65) - 0.25 exp(-0.5 ((t - 1965)/15)^2), t in calendar years;
    # the constant c is not needed because the series is mean-centered over the record 1850-2014
    a = 2.
    n_spin = int(round(12 * spin_up))
    n_rec = 12 * (2014 - 1850 + 1)                       # 1980 monthly steps, Jan 1850 to Dec 2014
    years = 1850.0 + np.arange(-n_spin, n_rec) / 12.0
    F = a * np.exp((years - 2014.0) / 65.0) - depth * np.exp(-0.5 * ((years - 1965.0) / 15.0) ** 2)
    y0 = -F[n_spin:].mean()                              # record mean (does not depend on spin_up)
    y = F + y0
    return W, W_inv, B, years, y, y0


def generate_ensemble(n_members, tau1, split=(1, 1, 1, 1, 0), b_rot = 0.0, depth = 0.25, spin_up_factor=100,                   
                      w_seed=22, eig_seed=20, tau_pair=1.5, period_pair=4., lam_noise_max=0.1, rng=None):
    """Ensemble of realizations of the synthetic system (monthly step, record 1850-2014).

    n_members       : number of ensemble members (independent noise, common forcing)
    tau1            : decay time of the slow mode, years
    split           : fractions (V_f, s1, sp, sc, v) of the total variance V_f + s1^2 + 2 sp^2 + (M-3) sc^2 + M v^2,
                      normalized to sum 1: the forced variance V_f (sum over grid points of the time variance of the
                      forced response over the record, split[0] > 0), the internal variance of the slow mode (s1^2),
                      of the fast pair (shared by its two components, 2 sp^2), of the 17 noise modes (shared, (M-3) sc^2),
                      and of the measurement noise (white, independent at each grid node, time step and member, M v^2).
                      The theoretical SNR = V_f / (internal + measurement noise variance)
                      = split[0] / (split[1] + split[2] + split[3] + split[4]).
    b_rot           : fraction of the angle between the forcing pattern b and the slow mode to rotate b toward the slow mode (0: no rotation, 1: b aligned with the slow mode)
    depth           : amplitude of the Gaussian dip of the forcing around 1965 (depth = 0 gives a monotone forcing)
    spin_up_factor  : spin-up length in units of the slowest decay time (max(tau1, tau_pair))
    w_seed, eig_seed: seeds of the noise modes and noise eigenvalues
    tau_pair, period_pair : decay time (yr) and period (yr) of the fast (ENSO-like) pair
    rng             : seed or numpy Generator of the noise realizations (None: unseeded)
    Returns a dict (record = Jan 1850 to Dec 2014, N = 1980 monthly steps, M = 20), including the theoretical SNR ('snr').
    data = forced + internal + obs_noise, all with zero time mean for each member; 'internal' is the process internal
    variability without the measurement noise.
    """
    M = 20
    rng = np.random.default_rng(rng)
    f = np.asarray(split, dtype=float)
    f = f / f.sum()

    # patterns, centered forcing with its history (spin-up in years)
    spin_up = spin_up_factor * max(tau1, tau_pair)
    W, W_inv, b, years, y, y0 = make_setup(spin_up, seed=w_seed, b_rot=b_rot, depth=depth)
    
    n_spin = int(round(12 * spin_up))
    N = len(y)

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
    xf, forced = np.zeros(M), np.empty((N - n_spin, M))
    for i, yi in enumerate(y):
        xf = A @ xf + B * yi
        if i >= n_spin:
            forced[i - n_spin] = xf
    Vf = forced.var(axis=0).sum()

    # modal variances and measurement noise from the split of the total variance, noise std from the decay rates
    V_tot = Vf / f[0]
    s2 = np.r_[f[1] * V_tot, f[2] * V_tot / 2, f[2] * V_tot / 2, np.full(M - 3, f[3] * V_tot / (M - 3))]
    sigma_obs = np.sqrt(f[4] * V_tot / M)                                      # measurement noise std at each grid node
    snr = Vf / (s2.sum() + M * sigma_obs ** 2)                                 # theoretical SNR (internal + measurement noise)
    sigma = np.sqrt(s2 * (1 - np.r_[lam1, rho, rho, lam_n] ** 2))

    # ensemble: the full system from zero over spin-up + record
    x, data = np.zeros((n_members, M)), np.empty((n_members, N - n_spin, M))
    for i, yi in enumerate(y):
        x = x @ A.T + B * yi + (sigma * rng.standard_normal((n_members, M))) @ W.T
        if i >= n_spin:
            data[:, i - n_spin] = x

    internal = data - forced                                  # process internal variability (without measurement noise)
    obs = sigma_obs * rng.standard_normal(data.shape)       # measurement noise, added to the process data
    data_mean = (data + obs).mean(axis=1, keepdims=True)      # one common mean per member (over the record)
    forced = forced - forced.mean(axis=0)                     # all components are mean-centered over the record
    internal = internal - internal.mean(axis=1, keepdims=True)
    obs = obs - obs.mean(axis=1, keepdims=True)
    data = forced + internal + obs                            # measured data, mean-centered (= raw data - data_mean)
    return dict(data=data, forced=forced, internal=internal, obs_noise=obs, data_mean=data_mean[:, 0, :],
                years=years[n_spin:], y=y[n_spin:], years_full=years, y_full=y, y0=y0,
                A=A, W=W, W_inv=W_inv, Lambda_R=L, B=B, b_scale=b_scale,
                modal_var=s2, noise_std=sigma, obs_std=sigma_obs, Vf=Vf, snr=snr, spin_up=spin_up)



def fit_pullback_ensemble(data, y, y_full, lag=3):
    """Fit the repo's PullbackDMDc to every ensemble member (each member as a single realization).

    data   : (n_members, N, M) ensemble data over the record
    y      : (N,) forcing over the record (mean-centered over the record)
    y_full : (n_spin + N,) forcing over spin-up + record, same units; the whole spin-up is given to the method
    lag    : lag of the fit in months (3 in the paper)
    Full truncation (M EOFs) and transition_time = n_spin = len(y_full) - N.
    Returns the list of fitted models (compute_modes already called) and forced_est (n_members, N, M) =
    forced PCs @ EOFs, without the data mean added back by predict()."""
    n_members, N, M = data.shape
    n_spin = len(y_full) - N
    if not np.allclose(y_full[-N:], y):
        raise ValueError("y must be the last N values of y_full")
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

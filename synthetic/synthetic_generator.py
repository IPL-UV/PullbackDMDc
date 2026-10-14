"""
Synthetic ensemble for testing PullbackDMDc: generation, fitting and diagnostics (monthly step, record 1850-2014).

System (M = 20 latitudes, -90 to 90 degrees, poles included, unit-norm patterns):
    x(t) = A x(t-1) + B y(t) + xi(t),   A = W Lambda W^-1,   data = forced + internal + measurement noise
    W columns: slow mode, two fast modes (ENSO-like pair), 17 noise modes. Lambda: slow real eigenvalue, fast pair (decay time, period), noise modes with small eigenvalues.
    Process noise is diagonal in the modal basis; y(t) is the mean-centered forcing with its history (spin-up).

Functions
    make_setup(spin_up, seed, n, overlap, depth)
        Builds W, W_inv, the unit-norm forcing pattern B, the forcing time axis and the mean-centered forcing y (and y0 = -record mean).
        overlap: cosine between every non-slow mode (fast and noise modes) and the slow mode; 0 keeps the random orthogonal complement.
        depth: amplitude of the Gaussian dip of the forcing around 1965.
    generate_ensemble(n_members, tau1, split, overlap, depth, spin_up_factor, w_seed, eig_seed, tau_pair, period_pair, lam_noise_max)
        Simulates the ensemble (independent noise, common forcing) after a spin-up of spin_up_factor * max(tau1, tau_pair) years.
        split = (V_f, s1, sp, sc, v): fractions of the total variance in the forced response, slow, fast-pair, noise-mode internal variability and measurement noise; theoretical SNR = split[0] / sum(split[1:]).
        B is scaled so that the quasi-equilibrium forced state at the start of the spin-up has unit norm.
        Returns a dict: data, forced, internal, obs_noise (all zero-mean per member, data = forced + internal + obs_noise), data_mean, years, y, years_full, y_full, y0, A, W, W_inv, Lambda_R, B, b_scale, modal_var, noise_std, obs_std, Vf, snr, spin_up.
    fit_pullback_ensemble(data, y, y_full, lag)
        Fits the repo's PullbackDMDc to each member separately (full truncation, transition_time = whole spin-up).
        Returns the fitted models and the estimated forced responses (n_members, N, M) from model.predict().
    plot_setup(overlap, depth)
        Patterns (slow, fast, B), noise modes and the forcing with the spin-up and training interval; prints y0 and W^T W; saves figures/setup.pdf.
    plot_ensemble_diagnostics(out, area_weighted)
        (a) global-mean time series of the data and the forced response, (b) histogram over members of the empirical SNR vs the theoretical one, (c) empirical vs theoretical modal variances of the internal variability; saves figures/ensemble_diagnostics.pdf.
    plot_pullback_results(out, models, forced_est, area_weighted)
        (a) global-mean true and estimated forced responses with the ensemble mean, (b) histogram of the normalized MSE of the forced response per member, (c) estimated vs true decay times (years), (d) estimated vs true slow-mode pattern; prints the percentage of members with a slowest |eigenvalue| > 1; saves figures/pullback_results.pdf.

Usage: see the main block (generate the ensemble, fit PullbackDMDc, plot). Requires the PullbackDMDc repo (utils.pullback_dmdc) on the path and a figures/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator, ScalarFormatter
from utils.pullback_dmdc import PullbackDMDc

def make_setup(spin_up, seed=22, n=20, overlap=0.0,depth = 0.25):                      
    """W, W_inv, unit-norm unscaled B, mean-centered forcing y on monthly steps from 1850 - spin_up to Dec 2014, and y0.

    spin_up: spin-up length in years (monthly steps, so 12 * spin_up months before January 1850).
    overlap: cosine between every noise mode (columns 3..n-1 of W) and the slow mode, in [0, 1);
             0 gives the orthogonal complement. The noise columns keep unit norm but are no longer
             orthogonal to each other (cos = overlap**2) nor to the fast modes.                  # NEW
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

    u = W[:, [0]]                                                              # unit slow mode, (n, 1)
    Q = W[:, 1:]                                                       # modes to rotate toward the slow mode, (n, m)
    Q = Q - u * (u.T @ Q)                                                      # components orthogonal to the slow mode
    W[:, 1:] = overlap * u + np.sqrt(1 - overlap**2) * Q / np.linalg.norm(Q, axis=0)
    W_inv = np.linalg.inv(W)                                                   

    # forcing pattern b (unit norm), 20 values on the grid
    b = np.ones(n)/np.sqrt(n) # other option could be 0.2 + 0.05* np.cos(2 * phi)
    B = unit(b)

    # forcing: F(t) = a exp((t - 2014)/65) - 0.25 exp(-0.5 ((t - 1965)/15)^2), t in calendar years;
    # the constant c is not needed because the series is mean-centered over the record 1850-2014
    a = 1.9642341976276356
    n_spin = int(round(12 * spin_up))
    n_rec = 12 * (2014 - 1850 + 1)                       # 1980 monthly steps, Jan 1850 to Dec 2014
    years = 1850.0 + np.arange(-n_spin, n_rec) / 12.0
    F = a * np.exp((years - 2014.0) / 65.0) - depth * np.exp(-0.5 * ((years - 1965.0) / 15.0) ** 2)
    y0 = -F[n_spin:].mean()                              # record mean (does not depend on spin_up)
    y = F + y0
    return W, W_inv, B, years, y, y0


def generate_ensemble(n_members, tau1, split=(1, 1, 1, 1, 0), overlap = 0.0, depth = 0.25, spin_up_factor=100,                   
                      w_seed=22, eig_seed=20, tau_pair=1.5, period_pair=4., lam_noise_max=0.1):
    """Ensemble of realizations of the synthetic system (monthly step, record 1850-2014).

    n_members       : number of ensemble members (independent noise, common forcing)
    overlap         : cosine between every noise mode and the slow mode, in [0, 1) (see make_setup)
    tau1            : decay time of the slow mode, years
    split           : fractions (V_f, s1, sp, sc, v) of the total variance V_f + s1^2 + 2 sp^2 + (M-3) sc^2 + M v^2,
                      normalized to sum 1: the forced variance V_f (sum over grid points of the time variance of the
                      forced response over the record, split[0] > 0), the internal variance of the slow mode (s1^2),
                      of the fast pair (shared by its two components, 2 sp^2), of the 17 noise modes (shared, (M-3) sc^2),
                      and of the measurement noise (white, independent at each grid node, time step and member, M v^2).
                      The theoretical SNR = V_f / (internal + measurement noise variance)
                      = split[0] / (split[1] + split[2] + split[3] + split[4]).
    spin_up_factor  : spin-up length in units of the slowest decay time (max(tau1, tau_pair))
    w_seed, eig_seed: seeds of the noise modes and noise eigenvalues
    tau_pair, period_pair : decay time (yr) and period (yr) of the fast (ENSO-like) pair
    Returns a dict (record = Jan 1850 to Dec 2014, N = 1980 monthly steps, M = 20), including the theoretical SNR ('snr').
    data = forced + internal + obs_noise, all with zero time mean for each member; 'internal' is the process internal
    variability without the measurement noise.
    """
    M = 20
    f = np.asarray(split, dtype=float)
    f = f / f.sum()

    # patterns, centered forcing with its history (spin-up in years)
    spin_up = spin_up_factor * max(tau1, tau_pair)
    W, W_inv, b, years, y, y0 = make_setup(spin_up, seed=w_seed, overlap=overlap, depth=depth)
    
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
        x = x @ A.T + B * yi + (sigma * np.random.normal(size=(n_members, M))) @ W.T
        if i >= n_spin:
            data[:, i - n_spin] = x

    internal = data - forced                                  # process internal variability (without measurement noise)
    obs = sigma_obs * np.random.normal(size=data.shape)       # measurement noise, added to the process data
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

def plot_setup(overlap=0.0, depth=0.25):
    spin_up = 100                                            # years, for the plot
    W, W_inv, B, years, y, y0 = make_setup(spin_up, overlap=overlap, depth=depth)
    print("y0 =", y0)
    print("Uncentered correlation matrix of the eigenvectors (W^T W):")
    print(np.array2string(np.round(W.T @ W, 2) + 0.0, precision=2, max_line_width=250))

    n = W.shape[0]
    lat = np.linspace(-90, 90, n)                            # same grid as in make_setup
    patterns = [(W[:, 0], "slow mode", "#0072B2", "o", "-"),
                (W[:, 1], "fast mode 1", "#D55E00", "o", "-"),
                (W[:, 2], "fast mode 2", "#009E73", "o", "-"),
                (B, "forcing pattern B", "k", "o", "--")]

    fig, (ax1, axn, ax2) = plt.subplots(3, 1, figsize=(7, 10))  

    # (a) patterns, unit norm
    for v, label, color, marker, ls in patterns:
        hollow = ls == "--"
        ax1.plot(lat, v, color=color, ls=ls, lw=1.5, marker=marker, ms=4,
                mfc="white" if hollow else color, label=label)
    ax1.axhline(0, color="0.6", lw=0.8)
    ax1.set_xticks(np.arange(-90, 91, 30))
    ax1.set_xticklabels(["90S", "60S", "30S", "0", "30N", "60N", "90N"])
    ax1.set_xlabel("Latitude")
    ax1.set_ylabel("Loading (unit norm)")
    ax1.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    ax1.text(0.0, 1.12, "(a)", transform=ax1.transAxes, fontsize=11, fontweight="bold")

    # (b) noise modes (columns 3..n-1 of W), slow mode dashed for reference                  
    for k in range(3, n):                                                                      
        axn.plot(lat, W[:, k], color="0.45", lw=0.9, alpha=0.7, marker="o", ms=2.5,            
                 label=f"noise modes ({n - 3})" if k == 3 else None)                           
    axn.plot(lat, W[:, 0], color="#0072B2", ls="--", lw=1.6, label="slow mode")                
    axn.axhline(0, color="0.6", lw=0.8)                                                        
    axn.set_xticks(np.arange(-90, 91, 30))                                                     
    axn.set_xticklabels(["90S", "60S", "30S", "0", "30N", "60N", "90N"])                       
    axn.set_xlabel("Latitude")                                                                 
    axn.set_ylabel("Loading (unit norm)")                                                      
    axn.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)   
    axn.text(0.0, 1.12, "(b)", transform=axn.transAxes, fontsize=11, fontweight="bold")        

    # (c) forcing on the full interval, spin-up and training interval marked                   
    ax2.axvspan(years[0], 1850, color="0.5", alpha=0.12, lw=0)
    ax2.axvspan(1850, years[-1], color="#0072B2", alpha=0.10, lw=0)
    ax2.axvline(1850, color="k", ls="--", lw=0.8)
    ax2.plot(years, y, color="k", lw=1.7)
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.axhline(y0, color="#D55E00", ls=":", lw=1.2)
    ax2.text(years[-1], y0 + 0.04, f"y0 = {y0:.4f}", color="#D55E00", ha="right", va="bottom")
    ax2.text((years[0] + 1850) / 2, 1.0, f"spin-up ({spin_up} yr)", ha="center", color="0.3")
    ax2.text((1850 + years[-1]) / 2, 1.0, "training interval 1850-2014", ha="center", color="0.3")
    ax2.set_xlim(years[0], years[-1])
    ax2.set_ylim(-0.8, 1.45)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Forcing y(t)")
    ax2.text(0.0, 1.05, "(c)", transform=ax2.transAxes, fontsize=11, fontweight="bold")   # CHANGED: (b) -> (c)

    for ax in (ax1, axn, ax2):                                                                 
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.savefig("synthetic/figures/setup.pdf", bbox_inches="tight")

def plot_ensemble_diagnostics(out, area_weighted=True):
    """(a) global-mean time series of the data (all members) and of the forced response,
    (b) histogram over members of the empirical SNR, with the theoretical SNR (internal variability + measurement noise),
    (c) empirical variances of the internal variability projected onto the modes (W_inv) vs the theoretical ones.
    out: dict returned by generate_ensemble. area_weighted: cos(latitude) weights for the global mean."""
    data, forced, internal = out["data"], out["forced"], out["internal"]
    years, W_inv, s2, Vf = out["years"], out["W_inv"], out["modal_var"], out["Vf"]
    n_members, N, M = data.shape
    lat = np.linspace(-90, 90, M)                            # same grid as in make_setup
    w = np.cos(np.deg2rad(lat)) if area_weighted else np.ones(M)
    w = w / w.sum()

    # (a) global means
    gm_data, gm_forced = data @ w, forced @ w

    # (b) SNR: exact forced variance / empirical non-forced variance (internal + measurement noise, summed over grid points)
    snr_emp = Vf / (data - forced).var(axis=1).sum(axis=1)                      
    snr_th = out["snr"]                                                         

    # (c) modal variances of the internal variability, z = W_inv x
    z_var = (internal @ W_inv.T).var(axis=1)                 # (n_members, M)
    modes = np.arange(1, M + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 10))

    for m in range(n_members):
        ax1.plot(years, gm_data[m], color="#0072B2", lw=0.6, alpha=0.6,
                 label="ensemble members" if m == 0 else None)
    ax1.plot(years, gm_forced, color="k", lw=2, label="forced response")
    ax1.set_xlim(years[0], years[-1])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Global mean" if area_weighted else "Mean over latitudes")
    ax1.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    ax1.text(0.0, 1.12, "(a)", transform=ax1.transAxes, fontsize=11, fontweight="bold")

    ax2.hist(snr_emp, bins="auto", color="#0072B2", alpha=0.7, edgecolor="white")                       
    ax2.axvline(snr_emp.mean(), color="#0072B2", ls=":", lw=1.4, label=f"empirical mean = {snr_emp.mean():.3f}")   
    ax2.axvline(snr_th, color="#D55E00", ls="--", lw=1.5, label=f"theoretical = {snr_th:.3f}")                    
    ax2.set_xlabel("SNR")                                                                                         
    ax2.set_ylabel("Number of members")                                                                           
    ax2.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    ax2.text(0.0, 1.12, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold")

    for m in range(n_members):
        ax3.plot(modes, z_var[m], "o", color="0.55", ms=3, alpha=0.6,
                 label="members" if m == 0 else None)
    ax3.plot(modes, z_var.mean(axis=0), "ko", ms=5, label="ensemble mean")
    ax3.plot(modes, s2, "D", color="#D55E00", mfc="none", ms=7, label="theoretical")
    ax3.set_yscale("log")
    ax3.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))       # readable ticks on the log axis
    ax3.yaxis.set_major_formatter(ScalarFormatter())
    ax3.yaxis.set_minor_locator(NullLocator())
    ax3.set_xticks(modes)
    ax3.set_xlabel("Mode (1: slow, 2-3: fast pair, 4-20: noise)")
    ax3.set_ylabel("Variance of internal variability")
    ax3.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    ax3.text(0.0, 1.12, "(c)", transform=ax3.transAxes, fontsize=11, fontweight="bold")

    for ax in (ax1, ax2, ax3):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.savefig("synthetic/figures/ensemble_diagnostics.pdf", bbox_inches="tight")

def plot_pullback_results(out, models, forced_est, area_weighted=True):
    """(a) global-mean true and estimated forced responses, (b) histogram over members of the forced-response
    MSE (error accumulated over grid nodes / variance accumulated over grid nodes), (c) estimated decay times vs
    the true ones, (d) estimated slowest-mode pattern vs the true slow pattern.
    out: dict from generate_ensemble; models, forced_est: from fit_pullback_ensemble.
    The true and estimated forced responses are compared as they are (no mean removal). Decay times are in years.
    The ensemble mean of out["data"] is shown as a benchmark estimate of the forced response.
    Prints the percentage of members whose slowest estimated eigenvalue has |lambda| > 1."""
    forced, years, W, A = out["forced"], out["years"], out["W"], out["A"]
    n_members, N, M = forced_est.shape
    lat = np.linspace(-90, 90, M)                            # same grid as in make_setup
    w = np.cos(np.deg2rad(lat)) if area_weighted else np.ones(M)
    w = w / w.sum()

    tr, est = forced, forced_est
    ens_mean = out["data"].mean(axis=0)                      

    # (b) per-member normalized MSE
    nmse = ((est - tr) ** 2).mean(1).sum(-1) / tr.var(0).sum()
    nmse_em = ((ens_mean - tr) ** 2).mean(0).sum() / tr.var(0).sum()        

    # (c) decay times (years) sorted by |lambda|, from the lag-step eigenvalues of each fit and from the true monthly A
    def decay_years(eigvals, lag):
        a = np.sort(np.abs(eigvals))[::-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a < 1, -lag / np.log(a), np.nan) / 12
    lag = models[0].lag
    T_est = np.array([decay_years(m.eigvals, lag) for m in models])
    T_true = decay_years(np.linalg.eigvals(A), 1)
    pct = 100 * np.mean([np.abs(m.eigvals[0]) > 1 for m in models])    # eigvals are sorted by |lambda|: eigvals[0] is the slowest
    print(f"Members with the slowest |eigenvalue| > 1: {pct:.1f}%")

    # (d) slowest-mode patterns, unit norm, sign aligned with the true slow pattern
    true_pat = W[:, 0] / np.linalg.norm(W[:, 0])
    pats = []
    for m in models:
        v = m.modes[:, 0]
        v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
        v = v.real / np.linalg.norm(v.real)
        pats.append(v * np.sign(v @ true_pat))
    pats = np.array(pats)
    cosines = pats @ true_pat

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    for m in range(n_members):
        ax1.plot(years, est[m] @ w, color="#0072B2", lw=0.6, alpha=0.5,
                 label="estimated (each member)" if m == 0 else None)
    ax1.plot(years, ens_mean @ w, color="#D55E00", lw=1.5, ls="--", label=f"ensemble mean ({n_members} members)")   
    ax1.plot(years, tr @ w, color="k", lw=2, label="true forced response")
    ax1.set_xlim(years[0], years[-1])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Global mean" if area_weighted else "Mean over latitudes")
    ax1.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    ax1.text(0.0, 1.12, "(a)", transform=ax1.transAxes, fontsize=11, fontweight="bold")

    bins = np.logspace(np.log10(nmse.min()), np.log10(nmse.max()), 16) if nmse.min() > 0 and nmse.max() > nmse.min() else 10
    ax2.hist(nmse, bins=bins, color="#0072B2", alpha=0.7, edgecolor="white")
    ax2.set_xscale("log")
    ax2.axvline(np.median(nmse), color="k", ls="--", lw=1.2, label=f"median = {np.median(nmse):.2f}")
    ax2.axvline(nmse_em, color="#D55E00", ls="--", lw=1.5, label=f"ensemble mean = {nmse_em:.2f}")   
    ax2.axvline(1, color="0.5", ls=":", lw=1.4, label="error variance = forced variance")
    ax2.set_xlabel("Normalized MSE of the forced response")
    ax2.set_ylabel("Number of members")
    ax2.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    ax2.text(0.0, 1.12, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold")

    modes = np.arange(1, M + 1)
    for m in range(n_members):
        ax3.plot(modes, T_est[m], "o", color="0.55", ms=3, alpha=0.5, label="estimated (each member)" if m == 0 else None)
    ax3.plot(modes, np.nanmedian(T_est, axis=0), "ko", ms=5, label="estimated, median")
    ax3.plot(modes, T_true, "D", color="#D55E00", mfc="none", ms=7, label="true")
    ax3.set_yscale("log")
    ax3.set_xticks(modes)
    ax3.set_xlabel("Mode (sorted by |eigenvalue|)")
    ax3.set_ylabel("Decay time (years)")
    ax3.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    ax3.text(0.0, 1.12, "(c)", transform=ax3.transAxes, fontsize=11, fontweight="bold")

    for m in range(n_members):
        ax4.plot(lat, pats[m], color="#0072B2", lw=0.6, alpha=0.5, label="estimated (each member)" if m == 0 else None)
    ax4.plot(lat, true_pat, color="k", lw=2, marker="o", ms=4, label=f"true slow pattern (median |cos| = {np.median(np.abs(cosines)):.2f})")
    ax4.axhline(0, color="0.6", lw=0.8)
    ax4.set_xticks(np.arange(-90, 91, 30))
    ax4.set_xticklabels(["90S", "60S", "30S", "0", "30N", "60N", "90N"])
    ax4.set_xlabel("Latitude")
    ax4.set_ylabel("Loading (unit norm)")
    ax4.legend(ncol=1, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    ax4.text(0.0, 1.12, "(d)", transform=ax4.transAxes, fontsize=11, fontweight="bold")

    for ax in (ax1, ax2, ax3, ax4):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.savefig("synthetic/figures/pullback_results.pdf", bbox_inches="tight")

if __name__ == "__main__":
# Default parameters choice
    overlap=0.
    depth = 0.25    # 0.25 or 1 -- dip parameter
    tau1=20
    np.random.seed(0)    # random generator used for noise realizations
    # forced response variance, slow mode variance, fast pair variance, noise modes variance, measurement noise variance (0.1, 0.2, 0.5, 1., 2., 3.)
    Vf, s1, sp, sc, v = 1, 1, 1, 0.9, 0.1 

# Generate mean-centered ensemble (dict), then compute PullbackDMDc for the members
    ens=generate_ensemble(n_members=100, tau1=tau1, split=(Vf, s1, sp, sc, v), overlap=overlap, depth = depth)    
    fitted_models, fr_estimations = fit_pullback_ensemble(ens["data"], ens["y"], ens["y_full"], lag=3)

# Some plots to check
    print("Check SNR in the output: ", ens['snr'],'=', Vf/(s1+sp+sc+v))
    plot_setup(overlap=overlap, depth = depth)
    plot_ensemble_diagnostics(ens)
    plot_pullback_results(ens, fitted_models, fr_estimations)

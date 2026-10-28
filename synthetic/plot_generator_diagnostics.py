"""
Diagnostic figures of the synthetic ensemble and of PullbackDMDc fitted to it, for the configuration CFG
(config.py's DEFAULT; change CFG or config.py to draw another one):

    python plot_generator_diagnostics.py        

PullbackDMDc is fitted to the first N_FIT realizations with the ablations' own functions (methods.py). The data of both
figures are saved in fig_dir/plot_data.npz together with the configuration they were made for; the next call reloads
them and only draws. Delete the file to regenerate it.

    setup.pdf             (a) the slow and fast patterns with the uniform forcing pattern, (b) the forcing for each dip depth
    pullback_results.pdf  (a) true and estimated forced responses and slow internal variability, as standardized means over the
                          latitudes, (b) forced-response correlation of the fitted realizations (the metric of the ablations),
                          (c) decay times, the median and 5 to 95% range of the estimates and the true ones, (d) estimated and
                          true slow-mode pattern (largest real eigenvalue)
"""

import json
from dataclasses import asdict, replace

import matplotlib.pyplot as plt
import numpy as np

from ablations import make_level
from config import DEFAULT, FIGURES_DIR, run_name, slug
from methods import compute_eofs, fit_pullback_dmdc
from run_ablation_studies import centered_corr, decaying, slow_index
from synthetic_generator import make_setup

CFG = DEFAULT              # the configuration drawn; e.g. CFG = replace(DEFAULT, variances=(1, 0.33, 0.33, 0.33, 0.01))
SETUP_DELTAS = (0.2, 0.5)  # the dip depths of setup.pdf
N_FIT = 1000               # realizations PullbackDMDc is fitted to
N_SHOWN = 100             # estimated forced responses drawn in panel (a) of pullback_results.pdf
N_INTERNAL = 5             # realizations of slow internal variability drawn in panel (a)
DATA_FILE = "plot_data.npz"
TRUE_COLOR, ESTIMATE_COLOR, MEDIAN_COLOR = "k", "#0072B2", "#D55E00"   # the same in every panel
SLOW_INTERNAL_COLORS = ("tab:orange", "tab:purple", "tab:green", "tab:red", "tab:pink")               # one per realization of slow internal variability
STYLE = {"font.size": 12, "legend.fontsize": 10}                       # large text on small figures


def setup_data(spin_up, seed, deltas=SETUP_DELTAS):
    """The data of setup.pdf: the patterns W, the forcing pattern b0 and the forcing y(t) for each dip depth."""
    W, W_inv, b0, years, y, y0 = make_setup(spin_up, seed=seed)
    ys = np.array([make_setup(spin_up, seed=seed, delta=d)[4] for d in deltas])
    return dict(setup_W=W, setup_b0=b0, setup_years=years, setup_ys=ys, setup_deltas=np.array(deltas),
                setup_spin_up=np.array(spin_up))


def draw_setup(data, fig_dir):
    W, b0, years, ys = data["setup_W"], data["setup_b0"], data["setup_years"], data["setup_ys"]
    deltas, spin_up = data["setup_deltas"], float(data["setup_spin_up"])
    lat = np.linspace(-90, 90, W.shape[0])                   # the grid of make_setup
    patterns = [(W[:, 0], "slow mode", "#0072B2", "-"),
                (W[:, 1], "fast mode 1", "#D55E00", "-"),
                (W[:, 2], "fast mode 2", "#009E73", "-"),
                (b0 / np.linalg.norm(b0), "uniform forcing pattern $b_0$", "k", "--")]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

    # (a) patterns, unit norm
    for v, label, color, ls in patterns:
        ax1.plot(lat, v, color=color, ls=ls, lw=1.5, marker="o", ms=4, mfc="white" if ls == "--" else color, label=label)
    ax1.axhline(0, color="0.6", lw=0.8)
    ax1.set_xticks(np.arange(-90, 91, 30))
    ax1.set_xticklabels(["90S", "60S", "30S", "0", "30N", "60N", "90N"])
    ax1.set_xlabel("Latitude")
    ax1.set_ylabel("Loading (unit norm)")
    ax1.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax1.text(0.0, 1.25, "(a)", transform=ax1.transAxes, fontsize=14, fontweight="bold")

    # (b) forcing for each dip depth, from the last century of the spin-up: the full spin-up is an almost constant
    # forcing for hundreds to thousands of years and would squeeze the dips into a sliver
    start = max(years[0], 1750.0)
    ax2.axvspan(start, 1850, color="0.5", alpha=0.12, lw=0)
    ax2.axvspan(1850, years[-1], color="#0072B2", alpha=0.10, lw=0)
    ax2.axvline(1850, color="k", ls="--", lw=0.8)
    for delta, y_delta, ls in zip(deltas, ys, ("-", ":", "-.")):
        ax2.plot(years, y_delta, color="k", ls=ls, lw=1.7, label=rf"$\delta = {delta:g}$")
    ax2.axhline(0, color="0.6", lw=0.8)
    top = ax2.get_xaxis_transform()                          # x in years, y in axes fraction
    ax2.text((1850 + years[-1]) / 2, 0.93, "training interval 1850-2014", ha="center", color="0.3", transform=top)
    ax2.set_xlim(start, years[-1])
    shown = years >= start
    lo, hi = ys[:, shown].min(), ys[:, shown].max()
    lo, hi = lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)
    ax2.set_ylim(lo, hi + 0.25 * (hi - lo))                  # room for the interval labels
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Forcing y(t)")
    ax2.legend(ncol=len(deltas), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax2.text(0.0, 1.25, "(b)", transform=ax2.transAxes, fontsize=14, fontweight="bold")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.savefig(fig_dir / "setup.pdf", bbox_inches="tight")
    plt.close(fig)


def fit_pullback_realizations(ens, lag, n_fit):
    """The ablations' PullbackDMDc on the first n_fit realizations of ens (the dict make_level returns): EOFs per
    realization, the whole forcing history including the spin-up as the long forcing, methods.fit_pullback_dmdc.
    Returns the forced-response estimates (n_fit, T, M) and the (eigenvalues, grid-space eigenvectors) of the fits."""
    T = ens["data"].shape[1]
    long_forcings, short_forcings = ens["y_full"][:, None], ens["y"][:, None]
    spinup = len(ens["y_full"]) - T
    forced_est, eigs = [], []
    n_fit = min(n_fit, len(ens["data"]))
    for r, data in enumerate(ens["data"][:n_fit]):
        estimate, eig = fit_pullback_dmdc(data, long_forcings, short_forcings, spinup, compute_eofs(data), lag=lag)
        forced_est.append(estimate)
        eigs.append(eig)
        print(f"PullbackDMDc fitted to realization {r + 1}/{n_fit}", flush=True)
    return np.array(forced_est), eigs


def pullback_data(out, forced_est, eigs, lag, n_internal=N_INTERNAL):
    """The data of pullback_results.pdf: the truth, the fits, and the slow internal variability of the first n_internal
    realizations (the process internal variability, data minus the forced response and the measurement noise,
    in the coordinate of the slow mode, z_1(t))."""
    internal = out["data"][:n_internal] - out["forced"] - out["obs_noise"][:n_internal]
    return dict(pb_years=out["years"], pb_forced=out["forced"], pb_W=out["W"], pb_A=out["A"],
                pb_slow_internal_z=internal @ out["W_inv"][0], pb_forced_est=forced_est,
                pb_eigvals=np.array([eigvals for eigvals, _ in eigs]), pb_modes=np.array([modes for _, modes in eigs]),
                pb_lag=np.array(lag))


def draw_pullback(data, fig_dir, n_shown=N_SHOWN):
    forced, years, W, A = data["pb_forced"], data["pb_years"], data["pb_W"], data["pb_A"]
    forced_est, lag = data["pb_forced_est"], int(data["pb_lag"])
    eigs = list(zip(data["pb_eigvals"], data["pb_modes"]))
    n_members, _, M = forced_est.shape                       # the fitted realizations
    lat = np.linspace(-90, 90, M)                            # the grid of make_setup

    def standardized(x):                                     # a time series (..., T) with zero mean and unit variance
        x = x - x.mean(axis=-1, keepdims=True)
        return x / x.std(axis=-1, keepdims=True)

    # (a) means over the latitudes; the slow internal variability is w_1 z_1(t), whose mean is z_1 times that of w_1
    slow_internal = data["pb_slow_internal_z"] * W[:, 0].mean()

    # (b) forced-response correlation of each fitted realization, as scored in the ablations
    rho = np.array([centered_corr(est, forced) for est in forced_est])

    # (c) decay times (years) sorted by |lambda|, from the lag-step eigenvalues of each fit and from the true monthly A
    def decay_years(eigvals, lag):
        a = np.sort(np.abs(eigvals))[::-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a < 1, -lag / np.log(a), np.nan) / 12
    T_est = np.array([decay_years(eigvals, lag) for eigvals, _ in eigs])
    T_true = decay_years(np.linalg.eigvals(A), 1)

    # (d) slow-mode patterns: the eigenvector of the largest real eigenvalue (the rule of the ablations), unit norm, sign
    # aligned with the true slow pattern; realizations whose largest real eigenvalue does not decay are dropped
    true_pat = W[:, 0] / np.linalg.norm(W[:, 0])
    pats = []
    for eigvals, modes in eigs:
        index = slow_index(eigvals)
        if index is None or not decaying(eigvals[index].real):
            continue
        v = modes[:, index].real                             # eigenvector of a real eigenvalue: real up to roundoff
        v = v / np.linalg.norm(v)
        pats.append(v * (np.sign(v @ true_pat) or 1.0))
    print(f"Realizations dropped from (d), largest real eigenvalue >= 1 or none > 0: {n_members - len(pats)} of {n_members}")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6.5))

    for m in range(min(len(slow_internal), n_members)):      # slow internal variability, one color each, not in the legend
        ax1.plot(years, standardized(slow_internal[m]), color=SLOW_INTERNAL_COLORS[m % len(SLOW_INTERNAL_COLORS)], lw=0.9, alpha=0.9)
    for m in range(min(n_shown, n_members)):
        ax1.plot(years, standardized(forced_est[m].mean(axis=-1)), color=ESTIMATE_COLOR, lw=0.6, alpha=0.5,
                 label="estimated forced response" if m == 0 else None)
    ax1.plot(years, standardized(forced.mean(axis=-1)), color=TRUE_COLOR, lw=2, label="true forced response")
    ax1.set_xlim(years[0], years[-1])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Global mean, standardized")
    ax1.legend(ncol=1, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax1.text(0.0, 1.25, "(a)", transform=ax1.transAxes, fontsize=14, fontweight="bold")

    ax2.hist(rho, bins="auto", color=ESTIMATE_COLOR, alpha=0.7, edgecolor="white")
    ax2.axvline(np.median(rho), color=MEDIAN_COLOR, ls="--", lw=1.5, label=f"median = {np.median(rho):.3f}")
    ax2.set_xlabel("Forced-response correlation")
    ax2.set_ylabel("Number of realizations")
    ax2.set_xlim(0.97, 1.005)
    ax2.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax2.text(0.0, 1.25, "(b)", transform=ax2.transAxes, fontsize=14, fontweight="bold")

    modes = np.arange(1, M + 1)
    q16, median, q84 = np.nanpercentile(T_est, [16, 50, 84], axis=0)    # over the realizations, as the bands of the ablation sweeps
    ax3.vlines(modes, q16, q84, color=ESTIMATE_COLOR, lw=6, alpha=0.6, label="68% range")
    ax3.plot(modes, median, "o", color=MEDIAN_COLOR, ms=3, label="median")
    ax3.plot(modes, T_true, "D", color=TRUE_COLOR, mfc="none", ms=5, label="true")
    ax3.set_yscale("log")
    ax3.set_xticks(modes[::2])
    ax3.set_xlabel("Mode (sorted by |eigenvalue|)")
    ax3.set_ylabel("Decay time (years)")
    ax3.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handletextpad=0.2)
    ax3.text(0.0, 1.25, "(c)", transform=ax3.transAxes, fontsize=14, fontweight="bold")

    for m, pat in enumerate(pats):
        ax4.plot(lat, pat, color=ESTIMATE_COLOR, lw=0.6, alpha=0.5, label="estimated" if m == 0 else None)
    ax4.plot(lat, true_pat, color=TRUE_COLOR, lw=2, marker="o", ms=4, label="true slow pattern")
    ax4.axhline(0, color="0.6", lw=0.8)
    ax4.set_xticks(np.arange(-90, 91, 30))
    ax4.set_xticklabels(["90S", "60S", "30S", "0", "30N", "60N", "90N"])
    ax4.set_xlabel("Latitude")
    ax4.set_ylabel("Loading (unit norm)")
    ax4.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax4.text(0.0, 1.25, "(d)", transform=ax4.transAxes, fontsize=14, fontweight="bold")

    for ax in (ax1, ax2, ax3, ax4):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.savefig(fig_dir / "pullback_results.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_plot_data(path, data, meta):
    """All the data of the figures in one compressed file, with the configuration (meta, a JSON-able dict) they were made for."""
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **data)


def load_plot_data(path, meta):
    """The data save_plot_data stored, or None when there is no file or it was made for another configuration."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as f:
        if json.loads(str(f["meta"])) != meta:
            return None
        return {key: f[key] for key in f.files if key != "meta"}


def main():
    plt.rcParams.update(STYLE)
    cfg = CFG
    # by dip depth, plus any other departure from the defaults so that variants do not overwrite it
    others = slug(replace(cfg, delta=DEFAULT.delta))
    fig_dir = FIGURES_DIR / "generator" / (run_name(cfg) + ("" if others == "default" else f"__{others}"))
    fig_dir.mkdir(parents=True, exist_ok=True)

    # the saved data are reused only for the configuration (and the settings) they were made for
    meta = json.loads(json.dumps(dict(config=asdict(cfg), n_fit=N_FIT, n_internal=N_INTERNAL)))
    data_path = fig_dir / DATA_FILE
    data = load_plot_data(data_path, meta)
    if data is None:
        ens = make_level(cfg)
        print(f"{len(ens['data'])} realizations generated; fitting PullbackDMDc to the first {N_FIT}")
        forced_est, eigs = fit_pullback_realizations(ens, cfg.lag, N_FIT)
        print("Check SNR in the output: ", ens["snr"], "=", cfg.snr)
        data = {**setup_data(ens["spin_up"], cfg.w_seed), **pullback_data(ens, forced_est, eigs, cfg.lag)}
        save_plot_data(data_path, data, meta)
        print(f"saved the plotting data to {data_path}")
    else:
        print(f"reloaded the plotting data from {data_path} (delete it to regenerate)")
    draw_setup(data, fig_dir)
    draw_pullback(data, fig_dir)
    print(f"saved {fig_dir}/setup.pdf, pullback_results.pdf")


if __name__ == "__main__":
    main()
"""
Diagnostic figures of the synthetic ensemble (synthetic_generator.py) and of PullbackDMDc fitted to it.

    python plot_generator_diagnostics.py                                        # figures/generator/delta0.15/
    python plot_generator_diagnostics.py --set delta=0.5                        # figures/generator/delta0.5/
    python plot_generator_diagnostics.py --study obs_noise --level 1            # any ablation level

Functions
    plot_setup(spin_up, seed, deltas, fig_dir)
        (a) the slow and fast patterns with the uniform forcing pattern b0, (b) the forcing for each dip depth delta (solid, then dotted)
        from 1750, with the end of the spin-up and the training interval; prints W^T W; saves fig_dir/setup.pdf.
    plot_pullback_results(out, models, forced_est, area_weighted, fig_dir)
        (a) global-mean true and estimated forced responses with the ensemble mean, (b) histogram of the normalized MSE of the forced response per member, (c) estimated vs true decay times (years), (d) estimated vs true slow-mode pattern, the slow mode being the largest real eigenvalue; prints the number of members dropped from (d) because it does not decay; saves fig_dir/pullback_results.pdf.
"""

import argparse
import ast
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from ablations import find_level, level_kwargs, make_level
from config import DEFAULT, FIGURES_DIR, load_config, run_name, slug
from run_ablation_studies import decaying, slow_index
from synthetic_generator import fit_pullback_ensemble, make_setup

FIG_DIR = FIGURES_DIR / "generator" / run_name(DEFAULT)
SETUP_DELTAS = (0.15, 0.5)  # the two dip depths every experiment is run at

def plot_setup(spin_up=100, seed=22, deltas=SETUP_DELTAS, fig_dir=FIG_DIR):
    W, W_inv, b0, years, y, y0 = make_setup(spin_up, seed=seed)
    print("Uncentered correlation matrix of the eigenvectors (W^T W):")
    print(np.array2string(np.round(W.T @ W, 2) + 0.0, precision=2, max_line_width=250))

    n = W.shape[0]
    lat = np.linspace(-90, 90, n)                            # same grid as in make_setup
    patterns = [(W[:, 0], "slow mode", "#0072B2", "o", "-"),
                (W[:, 1], "fast mode 1", "#D55E00", "o", "-"),
                (W[:, 2], "fast mode 2", "#009E73", "o", "-"),
                (b0 / np.linalg.norm(b0), "uniform forcing pattern $b_0$", "k", "o", "--")]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.7))

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

    # (b) forcing for each dip depth, from the last century of the spin-up: the full spin-up is hundreds to
    # thousands of years of an almost constant forcing, and would squeeze the dips the panel compares into a sliver
    start = max(years[0], 1750.0)
    ax2.axvspan(start, 1850, color="0.5", alpha=0.12, lw=0)
    ax2.axvspan(1850, years[-1], color="#0072B2", alpha=0.10, lw=0)
    ax2.axvline(1850, color="k", ls="--", lw=0.8)
    for delta, ls in zip(deltas, ("-", ":", "-.")):
        y_delta = make_setup(spin_up, seed=seed, delta=delta)[4]
        ax2.plot(years, y_delta, color="k", ls=ls, lw=1.7, label=rf"$\delta = {delta:g}$")
    ax2.axhline(0, color="0.6", lw=0.8)
    top = ax2.get_xaxis_transform()                          # x in years, y in axes fraction
    ax2.text((start + 1850) / 2, 0.93, f"spin-up ({spin_up:g} yr)", ha="center", color="0.3", transform=top)
    ax2.text((1850 + years[-1]) / 2, 0.93, "training interval 1850-2014", ha="center", color="0.3", transform=top)
    ax2.set_xlim(start, years[-1])
    shown = years >= start
    lo, hi = min(make_setup(spin_up, seed=seed, delta=d)[4][shown].min() for d in deltas), y[shown].max()
    lo, hi = lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)
    ax2.set_ylim(lo, hi + 0.25 * (hi - lo))                  # room for the interval labels
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Forcing y(t)")
    ax2.legend(ncol=len(deltas), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    ax2.text(0.0, 1.12, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.savefig(fig_dir / "setup.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_pullback_results(out, models, forced_est, area_weighted=True, fig_dir=FIG_DIR):
    """(a) global-mean true and estimated forced responses, (b) histogram over members of the normalized MSE of the
    forced response, (1 / (T V_f)) sum_t ||x_hat^(f)(t) - x^(f)(t)||^2, (c) estimated decay times vs the true ones,
    (d) estimated slow-mode pattern vs the true slow pattern.
    out: dict from generate_ensemble; models, forced_est: from fit_pullback_ensemble.
    The true and estimated forced responses are compared as they are (no mean removal). Decay times are in years.
    The ensemble mean of out["data"] is shown as a benchmark estimate of the forced response.
    The estimated slow mode is the largest real eigenvalue (run_ablation_studies.slow_index, the rule the ablations
    are scored with); members where it does not decay (>= 1, or no positive real eigenvalue) are dropped from (d)
    and counted."""
    forced, years, W, A = out["forced"], out["years"], out["W"], out["A"]
    n_members, T, M = forced_est.shape
    lat = np.linspace(-90, 90, M)                            # same grid as in make_setup
    w = np.cos(np.deg2rad(lat)) if area_weighted else np.ones(M)
    w = w / w.sum()

    tr, est = forced, forced_est
    ens_mean = out["data"].mean(axis=0)                      

    # (b) per-member normalized MSE: squared error summed over time and grid nodes, over T V_f
    Vf = out["Vf"]                                           # sum over grid nodes of the time variance of x^(f)
    nmse = ((est - tr) ** 2).sum(axis=(1, 2)) / (T * Vf)
    nmse_em = ((ens_mean - tr) ** 2).sum() / (T * Vf)        

    # (c) decay times (years) sorted by |lambda|, from the lag-step eigenvalues of each fit and from the true monthly A
    def decay_years(eigvals, lag):
        a = np.sort(np.abs(eigvals))[::-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a < 1, -lag / np.log(a), np.nan) / 12
    lag = models[0].lag
    T_est = np.array([decay_years(m.eigvals, lag) for m in models])
    T_true = decay_years(np.linalg.eigvals(A), 1)

    # (d) slow-mode patterns: the eigenvector of the largest real eigenvalue, unit norm, sign aligned with the
    # true slow pattern; members whose largest real eigenvalue does not decay are dropped
    true_pat = W[:, 0] / np.linalg.norm(W[:, 0])
    pats = []
    for m in models:
        index = slow_index(m.eigvals)
        if index is None or not decaying(m.eigvals[index].real):
            continue
        v = m.modes[:, index].real                           # eigenvector of a real eigenvalue: real up to roundoff
        v = v / np.linalg.norm(v)
        pats.append(v * (np.sign(v @ true_pat) or 1.0))
    pats = np.array(pats).reshape(-1, M)
    n_dropped = n_members - len(pats)
    print(f"Members dropped from (d), largest real eigenvalue >= 1 or none > 0: {n_dropped} of {n_members}")
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
    ax2.set_xlabel(r"Normalized MSE $\frac{1}{T\,V_f}\sum_t \|\hat{x}^{(f)}(t) - x^{(f)}(t)\|^2$")
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

    for m in range(len(pats)):
        ax4.plot(lat, pats[m], color="#0072B2", lw=0.6, alpha=0.5,
                 label=f"estimated (each member; {n_dropped} non-decaying dropped)" if m == 0 else None)
    median_cos = f"{np.median(np.abs(cosines)):.2f}" if len(pats) else "n/a"
    ax4.plot(lat, true_pat, color="k", lw=2, marker="o", ms=4, label=f"true slow pattern (median |cos| = {median_cos})")
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
    plt.savefig(fig_dir / "pullback_results.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides (see config.py)")
    parser.add_argument("--from-run", help="start from results/<run>/config.json, then apply --set")
    parser.add_argument("--study", help="draw one level of this ablation study instead of the reference")
    parser.add_argument("--level", type=ast.literal_eval, help="the study level, e.g. 0.9 or (1, 20) for the joint study")
    args = parser.parse_args()

    cfg = load_config(args)
    level = find_level(cfg, args.study, args.level) if args.study else None
    kwargs = level_kwargs(cfg, args.study, level)
    # by dip depth, plus any other departure from the defaults so that variants do not overwrite it
    others = slug(replace(cfg, delta=DEFAULT.delta))
    fig_dir = FIGURES_DIR / "generator" / (run_name(cfg) + ("" if others == "default" else f"__{others}")
                                           + (f"__{args.study}={level}" if args.study else ""))
    fig_dir.mkdir(parents=True, exist_ok=True)

    ens = make_level(cfg, args.study, level)
    fitted_models, fr_estimations = fit_pullback_ensemble(ens["data"], ens["y"], ens["y_full"], lag=cfg.lag)
    variances = kwargs["variances"]
    print("Check SNR in the output: ", ens["snr"], "=", variances[0] / sum(variances[1:]))
    plot_setup(spin_up=ens["spin_up"], seed=kwargs["w_seed"], fig_dir=fig_dir)
    plot_pullback_results(ens, fitted_models, fr_estimations, fig_dir=fig_dir)
    print(f"saved {fig_dir}/setup.pdf, pullback_results.pdf")


if __name__ == "__main__":
    main()

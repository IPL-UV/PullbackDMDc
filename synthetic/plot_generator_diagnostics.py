"""
Diagnostic figures of the synthetic ensemble (synthetic_generator.py) and of PullbackDMDc fitted to it.

    python plot_generator_diagnostics.py                                        # figures/generator/default/
    python plot_generator_diagnostics.py --set depth=1
    python plot_generator_diagnostics.py --study obs_noise --level 1            # any ablation level

Functions
    plot_setup(b_rot, depth, spin_up, seed, fig_dir)
        Patterns (slow, fast, B), noise modes and the forcing with the spin-up and training interval; prints y0 and W^T W; saves fig_dir/setup.pdf.
    plot_ensemble_diagnostics(out, area_weighted, fig_dir)
        (a) global-mean time series of the data and the forced response, (b) histogram over members of the empirical SNR vs the theoretical one, (c) empirical vs theoretical modal variances of the internal variability; saves fig_dir/ensemble_diagnostics.pdf.
    plot_pullback_results(out, models, forced_est, area_weighted, fig_dir)
        (a) global-mean true and estimated forced responses with the ensemble mean, (b) histogram of the normalized MSE of the forced response per member, (c) estimated vs true decay times (years), (d) estimated vs true slow-mode pattern; prints the percentage of members with a slowest |eigenvalue| > 1; saves fig_dir/pullback_results.pdf.
"""

import argparse
import ast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullLocator, ScalarFormatter

from ablations import find_level, level_kwargs, make_level
from config import FIGURES_DIR, load_config, slug
from synthetic_generator import fit_pullback_ensemble, make_setup

FIG_DIR = FIGURES_DIR / "generator" / "default"

def plot_setup(b_rot=0.0, depth=0.25, spin_up=100, seed=22, fig_dir=FIG_DIR):
    W, W_inv, B, years, y, y0 = make_setup(spin_up, seed=seed, b_rot=b_rot, depth=depth)
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
    ax2.text((years[0] + 1850) / 2, 1.0, f"spin-up ({spin_up:g} yr)", ha="center", color="0.3")
    ax2.text((1850 + years[-1]) / 2, 1.0, "training interval 1850-2014", ha="center", color="0.3")
    ax2.set_xlim(years[0], years[-1])
    ax2.set_ylim(-0.8, 1.45)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Forcing y(t)")
    ax2.text(0.0, 1.05, "(c)", transform=ax2.transAxes, fontsize=11, fontweight="bold")   # CHANGED: (b) -> (c)

    for ax in (ax1, axn, ax2):                                                                 
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.savefig(fig_dir / "setup.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_ensemble_diagnostics(out, area_weighted=True, fig_dir=FIG_DIR):
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
    plt.savefig(fig_dir / "ensemble_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_pullback_results(out, models, forced_est, area_weighted=True, fig_dir=FIG_DIR):
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
    fig_dir = FIGURES_DIR / "generator" / (slug(cfg) + (f"__{args.study}={level}" if args.study else ""))
    fig_dir.mkdir(parents=True, exist_ok=True)

    ens = make_level(cfg, args.study, level)
    fitted_models, fr_estimations = fit_pullback_ensemble(ens["data"], ens["y"], ens["y_full"], lag=cfg.lag)
    split = kwargs["split"]
    print("Check SNR in the output: ", ens["snr"], "=", split[0] / sum(split[1:]))
    plot_setup(b_rot=kwargs["b_rot"], depth=kwargs["depth"], spin_up=ens["spin_up"], seed=kwargs["w_seed"],
               fig_dir=fig_dir)
    plot_ensemble_diagnostics(ens, fig_dir=fig_dir)
    plot_pullback_results(ens, fitted_models, fr_estimations, fig_dir=fig_dir)
    print(f"saved {fig_dir}/setup.pdf, ensemble_diagnostics.pdf, pullback_results.pdf")


if __name__ == "__main__":
    main()

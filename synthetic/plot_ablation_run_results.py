"""Figures for ablation runs (results/<run>/ from run_ablation_studies.py), written to figures/ablations/.

    python plot_ablation_run_results.py --runs baseline                # figures/ablations/baseline/
    python plot_ablation_run_results.py --runs baseline slow50         # figures/ablations/compare_baseline_vs_slow50/

With several runs the first is the reference: thin dotted medians under the later runs' solid lines and bands.
With exactly two runs a third figure maps the change in the phase diagram (later run minus reference).
"""

import argparse
import pathlib
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import RESULTS_DIR, Config, config_diff, from_json
from plot_style import save
from utils.params import colors, method_markers

FIGURES_DIR = pathlib.Path(__file__).resolve().parent / "figures" / "ablations"

METHOD_ORDER = ["PullbackDMDc", "LIM", "LIM-opt", "LR"]
# LIM and LIM-opt fit the same lag-step operator, so their structural scores coincide
OPERATOR_METHODS = {"PullbackDMDc": "PullbackDMDc", "LIM": "LIM / LIM-opt"}
SWEEP_STUDIES = {
    "total_snr": "total SNR",
    "partial_snr_slow": "partial SNR: slow",
    "partial_snr_pair": "partial SNR: pair",
    "partial_snr_complement": "partial SNR: complement",
    "slow_timescale_snr": r"slow timescale ($\mathrm{SNR}$ fixed)",
    "slow_timescale_modal_variance": r"slow timescale ($\sigma_1^2$ fixed)",
    "spatial_overlap": "spatial overlap",
}
PAIR_STUDIES = {"partial_snr_pair", "spatial_overlap"}
LINEAR_X = {"spatial_overlap"}
# the oracle error is ~0 for tau_1 <= 20 yr, so the RMSE contour is absolute, not relative to it
RMSE_THRESHOLD = 0.5
ORACLE_STYLE = dict(color="k", linestyle="--", linewidth=1.2)
REFERENCE_STYLE = dict(linestyle=":", linewidth=1.1, alpha=0.9)
VARIANT_LINESTYLES = ["-", "--", "-."]
PHASE_PANELS = [("forced_corr", "forced pattern correlation", "Purples", (None, 1)),
                ("forced_rel_rmse", "forced relative RMSE", "Purples_r", (0, None))]


@dataclass
class Run:
    name: str
    cfg: Config
    results: pd.DataFrame

    def label(self, base=None):
        diff = config_diff(self.cfg, base.cfg) if base is not None else config_diff(self.cfg)
        return f"{self.name} ({diff})" if diff else self.name

    @property
    def methods(self):
        return [m for m in METHOD_ORDER if m in set(self.results.method)]


def load_run(name):
    run_dir = RESULTS_DIR / name
    if not run_dir.is_dir():
        raise SystemExit(f"no run {name!r} in {RESULTS_DIR}; available: {sorted(p.name for p in RESULTS_DIR.iterdir())}")
    return Run(name, from_json(run_dir / "config.json"), pd.read_csv(run_dir / "ablations.csv"))


def quantiles(df, metric):
    grouped = df.groupby("param_value")[metric]
    return pd.DataFrame(dict(median=grouped.median(), lo=grouped.quantile(0.25), hi=grouped.quantile(0.75)))


def plot_band(ax, df, metric, method, label=None, linestyle="-"):
    q = quantiles(df[df.method == method], metric)
    ax.fill_between(q.index, q.lo, q.hi, color=colors[method], alpha=0.2, linewidth=0)
    ax.plot(q.index, q["median"], color=colors[method], marker=method_markers[method], markersize=5,
            linewidth=1.5, linestyle=linestyle, label=label or method)


def plot_reference(ax, df, metric, method):
    q = quantiles(df[df.method == method], metric)
    ax.plot(q.index, q["median"], color=colors[method], **REFERENCE_STYLE)


def structural_metric(study, lag):
    if study in PAIR_STUDIES:
        return "pair_angle", "pair-plane angle (deg)", "linear"
    power = "" if lag == 1 else f"^{{{lag}}}"
    return "slow_eig_err", rf"$|\hat\lambda_1 - \lambda_1{power}|$", "log"


def plot_sweeps(runs, path):
    reference, variants = (runs[0], runs[1:]) if len(runs) > 1 else (None, runs)
    studies = [s for s in SWEEP_STUDIES if any(s in set(r.results.study) for r in runs)]
    fig, axes = plt.subplots(len(studies), 3, figsize=(12, 2.6 * len(studies)), squeeze=False)
    for row, study in zip(axes, studies):
        if reference is not None:
            df = reference.results[reference.results.study == study]
            oracle = df[df.method == "oracle"].sort_values("param_value")
            for ax, metric in zip(row[:2], ("forced_corr", "forced_rel_rmse")):
                for method in reference.methods:
                    plot_reference(ax, df, metric, method)
                ax.plot(oracle.param_value, oracle[metric], color="0.5", **REFERENCE_STYLE)
            metric = structural_metric(study, reference.cfg.lag)[0]
            for method in OPERATOR_METHODS:
                if method in reference.methods:
                    plot_reference(row[2], df, metric, method)
        for run, linestyle in zip(variants, VARIANT_LINESTYLES):
            df = run.results[run.results.study == study]
            if df.empty:
                continue
            oracle = df[df.method == "oracle"].sort_values("param_value")
            for ax, metric in zip(row[:2], ("forced_corr", "forced_rel_rmse")):
                for method in run.methods:
                    plot_band(ax, df, metric, method, linestyle=linestyle)
                ax.plot(oracle.param_value, oracle[metric], label="oracle (true A, b)", **ORACLE_STYLE)
            metric, ylabel, yscale = structural_metric(study, run.cfg.lag)
            for method, label in OPERATOR_METHODS.items():
                if method in run.methods:
                    plot_band(row[2], df, metric, method, label=label, linestyle=linestyle)
            row[2].set_yscale(yscale)
            row[2].set_ylabel(ylabel)
            for ax in row:
                ax.set_xlabel(df.param_name.iloc[0])
        row[0].set_ylabel(f"{SWEEP_STUDIES[study]}\n\nforced corr")
        row[1].set_ylabel("forced rel. RMSE")
        for ax in row:
            ax.set_xscale("linear" if study in LINEAR_X else "log")
            ax.grid(alpha=0.3, linewidth=0.5)
    axes[0, 0].set_title("(a) forced pattern correlation")
    axes[0, 1].set_title("(b) forced relative RMSE")
    axes[0, 2].set_title("(c) operator recovery")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0, 1].legend(unique.values(), unique.keys())
    axes[0, 2].legend()
    if reference is not None:
        run_handles = [Line2D([], [], color="k", **REFERENCE_STYLE)] + [
            Line2D([], [], color="k", linestyle=ls) for _, ls in zip(variants, VARIANT_LINESTYLES)]
        run_labels = [f"{reference.label()} (reference)"] + [r.label(reference) for r in variants]
        fig.legend(run_handles, run_labels, loc="upper center", ncol=len(runs), frameon=False)
    else:
        fig.suptitle(variants[0].label())
    fig.text(0.5, 0.002, "median over realizations, shaded: interquartile range. (c) scores the eigen-decomposition "
             "of the fitted lag-step propagator; LIM-opt's forced estimate uses the SVD of its powered propagator, "
             "not these modes.", ha="center", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    save(fig, path, tight=False, bbox="tight")


def joint_results(run):
    df = run.results[run.results.study == "joint_snr_timescale"].copy()
    df["snr"], df["tau1_yr"] = df.param_value, df.tau1_yr.round(6)
    return df


def phase_grid(df, metric):
    grid = df.groupby(["snr", "tau1_yr"])[metric].median().unstack("tau1_yr")
    return grid.sort_index().sort_index(axis=1)


def phase_axes(n_rows, n_methods):
    return plt.subplots(n_rows, n_methods, figsize=(3.3 * n_methods, 3.1 * n_rows),
                        squeeze=False, sharex=True, sharey=True)


def label_phase_axes(ax, grid, method):
    """Tick the tau1 columns and SNR rows; returns the cell centers the contour overlays are drawn on."""
    x_centers, y_centers = np.arange(grid.shape[1]) + 0.5, np.arange(grid.shape[0]) + 0.5
    ax.set_title(method)
    ax.set_xticks(x_centers, [f"{tau:g}" for tau in grid.columns])
    ax.set_yticks(y_centers, [f"{snr:.2g}" for snr in grid.index])
    ax.grid(False)
    return x_centers, y_centers


def plot_phase(run, path):
    df = joint_results(run)
    if df.empty:
        return
    fig, axes = phase_axes(len(PHASE_PANELS), len(run.methods))
    overlays = {m: tuple(phase_grid(df[df.method == m], metric)
                         for metric in ("forced_corr", "forced_rel_rmse")) for m in run.methods}
    for row, (metric, title, cmap, (vmin, vmax)) in zip(axes, PHASE_PANELS):
        grids = {m: phase_grid(df[df.method == m], metric) for m in run.methods}
        values = np.concatenate([g.values.ravel() for g in grids.values()])
        vmin = values.min() if vmin is None else vmin
        vmax = min(values.max(), 2) if vmax is None else vmax
        for ax, method in zip(row, run.methods):
            corr_grid, rmse_grid = overlays[method]
            mesh = ax.pcolormesh(grids[method].values, cmap=cmap, vmin=vmin, vmax=vmax)
            centers = label_phase_axes(ax, grids[method], method)
            ax.contour(*centers, corr_grid.values, levels=[0.9], colors="k", linewidths=1.2)
            ax.contour(*centers, rmse_grid.values, levels=[RMSE_THRESHOLD], colors="k", linewidths=1.2,
                       linestyles="--")
        row[0].set_ylabel("SNR")
        fig.colorbar(mesh, ax=row, label=title, fraction=0.03, pad=0.02)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\tau_1$ (yr)")
    fig.suptitle(f"{run.label()}\nmedian over realizations; "
                 f"solid: corr = 0.9, dashed: rel. RMSE = {RMSE_THRESHOLD:g}")
    save(fig, path, tight=False, bbox="tight")


def plot_phase_change(reference, variant, path):
    ref, var = joint_results(reference), joint_results(variant)
    methods = [m for m in variant.methods if m in reference.methods]
    if ref.empty or var.empty or not methods:
        return
    fig, axes = phase_axes(len(PHASE_PANELS), len(methods))
    # blue = variant better: higher correlation, lower RMSE
    for row, (metric, title, _, _), cmap in zip(axes, PHASE_PANELS, ("RdBu", "RdBu_r")):
        deltas = {m: phase_grid(var[var.method == m], metric) - phase_grid(ref[ref.method == m], metric)
                  for m in methods}
        limit = np.nanmax([np.abs(delta.values).max() for delta in deltas.values()]) or 1.0
        for ax, method in zip(row, methods):
            mesh = ax.pcolormesh(deltas[method].values, cmap=cmap, vmin=-limit, vmax=limit)
            label_phase_axes(ax, deltas[method], method)
        row[0].set_ylabel("SNR")
        fig.colorbar(mesh, ax=row, label=f"change in {title}", fraction=0.03, pad=0.02)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\tau_1$ (yr)")
    fig.suptitle(f"{variant.label(reference)} minus {reference.name}: change in median (blue: better)")
    save(fig, path, tight=False, bbox="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["baseline"], help="run names; the first is the reference")
    args = parser.parse_args()

    runs = [load_run(name) for name in args.runs]
    out_dir = FIGURES_DIR / (runs[0].name if len(runs) == 1 else "compare_" + "_vs_".join(r.name for r in runs))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_sweeps(runs, out_dir / "results_sweeps.png")
    for run in runs:
        plot_phase(run, out_dir / f"results_phase_snr_timescale_{run.name}.png")
    if len(runs) == 2:
        plot_phase_change(*runs, out_dir / "results_phase_change.png")


if __name__ == "__main__":
    main()

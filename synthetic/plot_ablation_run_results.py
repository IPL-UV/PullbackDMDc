"""Figures for ablation runs (results/delta<delta>/ from run_ablation_studies.py), written to figures/ablations/.

    python plot_ablation_run_results.py                                   # dip depths 0.2 and 0.5, metric corr

The runs are named by their dip depth; the first is the reference, drawn solid with its 5th to 95th percentile band, and the later
run is drawn as dotted medians over it. --metric is the forced-response score of the top row of the sweeps and of the
phase diagrams: corr (default), rel_rmse or r2 (METRICS).

    results_sweeps_<metric>.png               one column per ablation study (DEFAULT_STUDIES); rows: the forced-response score, the
                                              fitted slow-mode decay time, the slow-pattern correlation; one legend across the top
    results_phase_snr_timescale_<run>_<metric>.png   the score over the SNR x tau_1 grid, one panel per method (one per run)
    results_forcing_change_<metric>.png       with exactly two runs: the forcing, the forced response it drives, the SNR sweep

The figures carry no captions. Sizes (figure, lines, text) are set at the top of each plotting function; the font family
and the method colours are shared: the colours come from utils/params.py, the same as the figures of the real climate example.
"""

import argparse
import pathlib
import sys
from dataclasses import dataclass, replace
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ablations import DEFAULT_STUDIES, LINEAR_X, STUDIES, make_level
from config import DEFAULT, FIGURES_DIR, RESULTS_DIR, from_json, run_name
from utils.params import colors, method_markers

plt.rcParams.update({  # the look shared by every figure; all sizes are set in the plotting functions
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "savefig.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})

FIGURES_DIR = FIGURES_DIR / "ablations"
METHOD_ORDER = ["PullbackDMDc", "LIM", "LIM-opt", "LR"]
OPERATOR_METHODS = ("PullbackDMDc", "LIM")  # the methods with a fitted propagator, scored in the decay-time and pattern rows
METHOD_LABELS = {"LR": "Linear reg."}       # legend names that differ from the method id, as in the main-text figures
FORCING_CHANGE_METHODS = ("PullbackDMDc", "LR")  # a method that uses the dynamics against one that only regresses on the forcing
DEFAULT_LINE_COLOR = "0.75"                 # the vertical line marking the default configuration in a sweep column


@dataclass(frozen=True)
class Metric:
    """A forced-response skill score: the top row of the sweeps and the phase diagrams (shaded, contoured at `levels`)."""
    column: str                   # results column
    label: str
    levels: tuple
    cmap: str                     # sequential, dark = good
    ylim: Optional[tuple] = None  # y range of the top row of the sweeps; None: the curves decide


# corr: each latitude is centered over time first, so only the shape counts; rel_rmse: rms(est - truth) over the time-centered
# RMS of the truth, an offset counts; r2: 1 - relative RMSE^2
METRICS = {
    "rel_rmse": Metric("forced_rel_rmse", "forced relative RMSE", (0.25, 0.5, 0.75, 1.0, 1.5), "Purples_r"),
    "corr": Metric("forced_corr", "forced-response correlation", (0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99), "gist_stern", (0.7, 1.01)),
    "r2": Metric("forced_r2", "forced $R^2$", (0, 0.25, 0.5, 0.75, 0.9), "Purples"),
}


@dataclass
class Run:
    name: str
    cfg: object              # config.Config of the run
    results: pd.DataFrame    # ablations.csv

    @property
    def methods(self):
        return [m for m in METHOD_ORDER if m in set(self.results.method)]

    @property
    def legend_label(self):
        return rf"$\delta = {self.cfg.delta:g}$"


def load_run(name):
    run_dir = RESULTS_DIR / name
    if not run_dir.is_dir():
        raise SystemExit(f"no run {name!r} in {RESULTS_DIR}; available: {sorted(p.name for p in RESULTS_DIR.iterdir())}")
    return Run(name, from_json(run_dir / "config.json"), pd.read_csv(run_dir / "ablations.csv"))


def default_level(study, cfg):
    """The x value of the default configuration in a study's sweep, where its vertical line is drawn."""
    return {"total_snr": cfg.snr, "slow_timescale_snr": cfg.tau1, "obs_noise": cfg.variances[4] / cfg.variances[0],
            "overlap": cfg.alpha}.get(study)


def draw_run(ax, run, df, column, methods, truth, linestyle, lw, ms):
    """One run in one panel. linestyle None draws the reference: solid medians over their 5th to 95th percentile bands; a linestyle
    draws a later run in it, wider medians with open circles. Returns the line of the true value (results column `truth`)."""
    for method in sorted((m for m in methods if m in run.methods), key=lambda m: m == "PullbackDMDc"):  # PullbackDMDc on top
        grouped = df[df.method == method].groupby("param_value")[column]
        median = grouped.median()
        if linestyle is None:
            ax.fill_between(median.index, grouped.quantile(0.16), grouped.quantile(0.84), color=colors[method], alpha=0.2, linewidth=0)
            ax.plot(median.index, median, color=colors[method], marker=method_markers[method], markersize=ms, linewidth=lw)
        else:
            ax.plot(median.index, median, color=colors[method], linestyle=linestyle, linewidth=lw + 0.5, marker="o",
                    markersize=ms - 1, markerfacecolor="white", markeredgecolor=colors[method])
    if truth is not None:                    # one value per level by construction, so the median is that value
        value = df.groupby("param_value")[truth].median()
        return ax.plot(value.index, value, color="k", linestyle="--" if linestyle is None else linestyle,
                       linewidth=1.2 if linestyle is None else lw + 0.5)[0]


def method_handles(methods, lw, ms):
    return [Line2D([], [], color=colors[m], marker=method_markers[m], markersize=ms, linewidth=lw,
                   label=METHOD_LABELS.get(m, m)) for m in METHOD_ORDER if m in methods]


def plot_sweeps(runs, path, metric):
    """One column per study in DEFAULT_STUDIES, one row per quantity, one legend across the top. Returns the figure."""
    col_w, row_h, header = 3., 2.7, 0.6       # inches: per study column, per row, and the strip kept for the legend
    lw, ms = 1.5, 5                            # line width and marker size of the reference; later runs are drawn wider
    text = {"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
            "legend.fontsize": 11}
    later_linestyles = [":", "-.", (0, (5, 2, 1, 2))]
    reference, variants = runs[0], runs[1:]
    studies = [s for s in DEFAULT_STUDIES if any(s in set(run.results.study) for run in runs)]
    # per row: y label, results column, y scale, y range (None: from the curves of the whole row), the methods drawn, the truth column
    rows = [(metric.label, metric.column, "linear", metric.ylim, METHOD_ORDER, None),
            ("slow-mode decay time\n" + r"$\hat\tau_1$ (yr)", "slow_tau_yr", "log", (0.5, None), OPERATOR_METHODS, "tau1_yr"),
            ("slow-pattern correlation", "slow_corr", "linear", (0.85, 1.01), OPERATOR_METHODS, None)]
    with plt.rc_context(text):
        fig, axes = plt.subplots(len(rows), len(studies), figsize=(col_w * len(studies), row_h * len(rows) + header), squeeze=False)
        for j, study in enumerate(studies):
            frames = {run.name: run.results[run.results.study == study] for run in runs}
            for i, (label, column, yscale, ylim, methods, truth) in enumerate(rows):
                ax, truth_line = axes[i, j], None
                for run, linestyle in zip(variants, later_linestyles):     # the later runs first, the reference on top of them
                    if not frames[run.name].empty:
                        draw_run(ax, run, frames[run.name], column, methods, truth, linestyle, lw, ms)
                if not frames[reference.name].empty:
                    truth_line = draw_run(ax, reference, frames[reference.name], column, methods, truth, None, lw, ms)
                ax.set_xscale("linear" if study in LINEAR_X else "log")    # per column: a property of the study
                ax.set_yscale(yscale)                                      # per row: a property of the quantity
                ax.grid(alpha=0.3, linewidth=0.5)
                default = default_level(study, reference.cfg)
                if default is not None:
                    ax.axvline(default, color=DEFAULT_LINE_COLOR, linewidth=1.4, zorder=0)
                if truth_line is not None:                                 # the true tau_1, in a legend inside the panel
                    ax.legend([truth_line], [r"true $\tau_1$"], loc="upper right", frameon=False)
            axes[0, j].set_title(STUDIES[study].title)
            axes[-1, j].set_xlabel(next(f.param_name.iloc[0] for f in frames.values() if not f.empty))
        for (label, _, _, ylim, _, _), row in zip(rows, axes):
            row[0].set_ylabel(label)
            # one y range per row, set after plotting: where the given range is open, the curves of the whole row decide
            limits = [ax.get_ylim() for ax in row]
            bottom, top = ylim or (None, None)
            bottom = min(low for low, _ in limits) if bottom is None else bottom
            top = max(high for _, high in limits) if top is None else top
            for ax in row:
                ax.set_ylim(bottom, top)
        # one legend across the top: the methods, the default configuration, and the runs (reference solid, later ones dotted)
        handles = method_handles(reference.methods, lw, ms)
        handles.append(Line2D([], [], color=DEFAULT_LINE_COLOR, linewidth=1.4, label="default configuration"))
        fig.tight_layout(rect=(0, 0, 1, 1 - header / fig.get_figheight()), h_pad=1.6)
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_phase(run, path, metric):
    """One panel per method: the metric over the SNR x tau_1 grid of the joint study, shaded and contoured at metric.levels,
    so the labelled contours are the colorbar read back onto the panel. Returns the figure, or None without a joint study."""
    panel_w, panel_h = 3.6, 3.4                # inches per method
    text = {"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10}
    contour_font = 9
    df = run.results[run.results.study == "joint_snr_timescale"]
    if df.empty:
        return None
    with plt.rc_context(text):
        fig, axes = plt.subplots(1, len(run.methods), figsize=(panel_w * len(run.methods), panel_h), squeeze=False,
                                 sharex=True, sharey=True)
        for ax, method in zip(axes[0], run.methods):
            d = df[df.method == method]
            grid = d.groupby([d.param_value, d.tau1_yr.round(6)])[metric.column].median().unstack().sort_index().sort_index(axis=1)
            x, y = np.arange(grid.shape[1]) + 0.5, np.arange(grid.shape[0]) + 0.5      # the cell centers
            # explicit levels rather than vmin/vmax: every panel shares one scale, and extend="both" colours the cells outside it
            filled = ax.contourf(x, y, grid.values, levels=metric.levels, cmap=metric.cmap, extend="both")
            lines = ax.contour(x, y, grid.values, levels=metric.levels, colors="k", linewidths=1.0)
            ax.clabel(lines, fmt="%.2g", fontsize=contour_font, inline=True)
            ax.set_title(METHOD_LABELS.get(method, method))
            ax.set_xticks(x, [f"{tau:g}" for tau in grid.columns])
            ax.set_yticks(y, [f"{snr:.3g}" for snr in grid.index])
            ax.set_xlabel(r"$\tau_1$ (yr)")
        axes[0, 0].set_ylabel("SNR")
        fig.colorbar(filled, ax=axes[0], ticks=metric.levels, label=metric.label, fraction=0.03, pad=0.02)
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_forcing_change(reference, variant, path, metric):
    """What changing the forcing's shape does: the forcing, the response it drives, and who notices. A deeper mid-century dip
    is a feature in time that the forcing and the forced response share, so a regression on the forcing can fit it directly,
    where PullbackDMDc has to get it through the dynamics; the SNR sweep shows which gains or loses. Returns the figure."""
    fig_size = (13, 3.8)                       # inches
    lw, ms = 1.6, 5
    text = {"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
            "legend.fontsize": 10}
    runs = [reference, variant]
    ensembles = {run.name: make_level(replace(run.cfg, n_realizations=1)) for run in runs}   # the forced response is noise free
    styles = {reference.name: dict(linestyle="-", color="k", linewidth=lw),
              variant.name: dict(linestyle=":", color="k", linewidth=lw + 0.5)}
    total = {run.name: run.results[run.results.study == "total_snr"] for run in runs}

    def standardized(v):
        return (v - v.mean()) / v.std()

    with plt.rc_context(text):
        fig, axes = plt.subplots(1, 3, figsize=fig_size)
        for run in runs:
            ens = ensembles[run.name]
            axes[0].plot(ens["years"], ens["y"], label=run.legend_label, **styles[run.name])
            axes[1].plot(ens["years"], standardized(ens["forced"].mean(axis=-1)), **styles[run.name])
        for ax in axes[:2]:
            ax.axhline(0, color="0.5", linewidth=0.6)
            ax.set_xlabel("year")
        axes[0].set_title("(a) forcing $y$, centered on the record")
        axes[0].set_ylabel("forcing")
        axes[0].legend(loc="upper left", frameon=False)
        axes[1].set_title("(b) forced response, global mean")
        axes[1].set_ylabel("standardized")
        for run, linestyle in zip(runs, (None, ":")):
            if not total[run.name].empty:
                draw_run(axes[2], run, total[run.name], metric.column, FORCING_CHANGE_METHODS, None, linestyle, lw, ms)
        axes[2].set_xscale("log")
        if metric.ylim is not None:
            axes[2].set_ylim(*metric.ylim)
        axes[2].grid(alpha=0.3, linewidth=0.5)
        axes[2].set_xlabel("SNR")
        axes[2].set_ylabel(metric.label)
        axes[2].set_title("(c) total SNR sweep")
        axes[2].legend(handles=method_handles(FORCING_CHANGE_METHODS, lw, ms), loc="lower right", frameon=False)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.2, 0.5],
                        help="dip depths of the runs to draw; the first is the reference (default: 0.2 0.5)")
    parser.add_argument("--metric", default="corr", choices=list(METRICS), help="forced-response skill score (default: corr)")
    args = parser.parse_args()

    runs = [load_run(run_name(replace(DEFAULT, delta=delta))) for delta in args.deltas]
    out_dir = FIGURES_DIR / (runs[0].name if len(runs) == 1 else "compare_" + "_vs_".join(r.name for r in runs))
    out_dir.mkdir(parents=True, exist_ok=True)
    metric = METRICS[args.metric]
    plot_sweeps(runs, out_dir / f"results_sweeps_{args.metric}.png", metric)
    for run in runs:
        plot_phase(run, out_dir / f"results_phase_snr_timescale_{run.name}_{args.metric}.png", metric)
    if len(runs) == 2:
        plot_forcing_change(*runs, out_dir / f"results_forcing_change_{args.metric}.png", metric)
    plt.close("all")
    print(f"saved the figures to {out_dir}")


if __name__ == "__main__":
    main()
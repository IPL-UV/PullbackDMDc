"""Figures for ablation runs (results/<run>/ from run_ablation_studies.py), written to figures/ablations/.

    python plot_ablation_run_results.py --runs baseline                # figures/ablations/baseline/
    python plot_ablation_run_results.py --runs baseline slow50         # figures/ablations/compare_baseline_vs_slow50/
    python plot_ablation_run_results.py --runs baseline --studies total_snr spatial_overlap

results_sweeps.png is one row per evaluation metric, (a) to (c), and one column per ablation study.
The columns are DEFAULT_STUDIES; --studies picks any other selection of SWEEP_STUDIES, left to right.

With several runs the first is the reference: thin dotted medians under the later runs' solid lines and bands.
With exactly two runs a third figure maps the change in the phase diagram (later run minus reference).
"""

import argparse
import pathlib
import sys
from dataclasses import dataclass, replace
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ablation_data import (DEFAULT_STUDIES, LINEAR_X, PHI, SWEEP_STUDIES, build_reference, global_mean,
                           record_years)
from config import RESULTS_DIR, Config, config_diff, from_json
from plot_ablation_diagnostics import standardized
from plot_style import (COLUMN_W, LEGEND_INCHES, PAGE_W, PANEL_TITLE, ROW_H, legend_rect, level_ticklabels,
                        reserve_legend_strip, save, zero_line)
from plot_system_diagnostics import reference_dataset
from utils.params import colors, method_markers

FIGURES_DIR = pathlib.Path(__file__).resolve().parent / "figures" / "ablations"

METHOD_ORDER = ["PullbackDMDc", "LIM", "LIM-opt", "LR"]
# LIM and LIM-opt fit the same lag-step operator, so their structural scores coincide
OPERATOR_METHODS = {"PullbackDMDc": "PullbackDMDc", "LIM": "LIM / LIM-opt"}
PAIR_STUDIES = {"partial_snr_pair"}  # studies scored on the oscillating pair rather than the slow mode
TRUTH_STYLE = dict(color="k", linestyle="--", linewidth=1.2)
REFERENCE_STYLE = dict(linestyle=":", linewidth=1.1, alpha=0.9)
VARIANT_LINESTYLES = ["-", "--", "-."]
DEFAULT_LINE_STYLE = dict(color="0.75", linewidth=0.8, zorder=0)  # marks the default system in a sweep column
# The phase figure maps one quantity, the forced relative RMSE: shaded by it, contoured in black at the levels
# below and labelled on the lines, so it carries its own key. The levels are absolute, so one contour means the
# same error in every panel and panels from different runs can be read against each other. There is no floor
# from a finite forcing window folded into them: the methods see the whole forcing series, back to where the
# truth itself starts, so a contour is the method's own error and nothing else.
PHASE_METRIC, PHASE_LABEL = "forced_rel_rmse", "forced relative RMSE"
PHASE_CMAP, PHASE_CHANGE_CMAP = "Purples_r", "RdBu_r"
RMSE_LEVELS = (0.25, 0.5, 0.75, 1.0, 1.5)  # 0.5 is the skill threshold the old single contour marked
PHASE_SUPTITLE_Y = 1.08  # the figure is one row, so the suptitle has to clear the panel titles
# the forcing-change figure contrasts a method that uses the dynamics with one that only regresses on
# the forcing, so a change in the forcing's shape is exactly what should separate them
FORCING_CHANGE_METHODS = ("PullbackDMDc", "LR")
MODE_SHAPE_CMAP = "viridis"  # sequential: the lines are ordered by their level, not categories


@dataclass
class Run:
    name: str
    cfg: Config
    results: pd.DataFrame
    modes: Optional[pd.DataFrame] = None  # slow_modes.npz, or None for a run made before it existed

    def label(self, base=None):
        diff = config_diff(self.cfg, base.cfg) if base is not None else config_diff(self.cfg)
        return f"{self.name} ({diff})" if diff else self.name

    @property
    def methods(self):
        return [m for m in METHOD_ORDER if m in set(self.results.method)]


def load_slow_modes(path):
    """slow_modes.npz as a DataFrame with a `pattern` column of arrays, or None when the run predates it."""
    if not path.exists():
        return None
    with np.load(path) as archive:
        patterns = archive["pattern"]
        frame = pd.DataFrame({k: archive[k] for k in archive.files if k != "pattern"})
    frame["pattern"] = list(patterns)
    return frame


def load_run(name):
    run_dir = RESULTS_DIR / name
    if not run_dir.is_dir():
        raise SystemExit(f"no run {name!r} in {RESULTS_DIR}; available: {sorted(p.name for p in RESULTS_DIR.iterdir())}")
    return Run(name, from_json(run_dir / "config.json"), pd.read_csv(run_dir / "ablations.csv"),
               load_slow_modes(run_dir / "slow_modes.npz"))


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


def spectral_metric(study=None, lag=None):
    """Row (b): the slow mode's fitted e-folding time in years, read against the true tau_1.

    The estimate itself rather than an error: with the truth drawn in the same panel the row shows
    which way a method is wrong -- too fast or too slow -- which no magnitude-only score can. Not the
    eigenvalue: tau = -1 / log|lambda| is steeply nonlinear near lambda = 1, so equal errors in lambda
    are wildly unequal in the timescale the row is about (at tau_1 = 20 yr, 0.0047 in lambda is a
    factor of ~130 in tau). tau is also lag-invariant, where lambda_1 ** lag is not.

    The quantity is the same in every column, so the symbol lives in the row label and no panel is
    annotated: the second return value is the per-panel annotation, and there is none.
    """
    return "slow_tau_yr", None


def shape_metric(study, lag=None):
    """Row (c): the correlation between a recovered mode subspace and the true one, in [0, 1].

    The cosine of their largest principal angle (run_ablation_studies.subspace_corr), so 1 is exact
    recovery of the pattern and the row reads the way the other skill scores do: up is better.

    Every study scores the slow mode except PAIR_STUDIES, which score the plane of the oscillating pair,
    so this row is not one quantity and its shared label cannot name both. The second return value is the
    symbol annotated inside that panel; every panel in the row carries one, so no column is a silent
    exception to the row label.
    """
    if study in PAIR_STUDIES:
        return "pair_corr", r"$\mathrm{corr}(\hat w_2 \hat w_3,\, w_2 w_3)$"
    return "slow_corr", r"$\mathrm{corr}(\hat w_1, w_1)$"


def unstable_note(df):
    """Per-method share of fits whose slow eigenvalue does not decay (|lambda| >= 1), or '' when there are none.

    A trend the fit cannot attribute to the forcing has to go somewhere, and in a propagator it goes into a
    non-decaying slow mode. Such a fit has no e-folding time at all, so row (b) drops it: where the share
    is large the curve is drawn from few fits, or stops, and this note is what carries the result.
    """
    if "slow_unstable" not in df:
        return ""
    shares = [(method, df[df.method == method].slow_unstable.mean()) for method in OPERATOR_METHODS]
    shown = [f"{method} {share:.0%}" for method, share in shares if share > 0.005]
    return "non-decaying dropped: " + ", ".join(shown) if shown else ""


@dataclass(frozen=True)
class RowSpec:
    """One row of the sweep figure: the same quantity in every study column.

    `metric` is a fixed column of the results table (the forced rows); `metric_for` is a per-study
    callable returning (metric, detail), where detail is a symbol annotated inside each panel when the
    row scores different objects in different studies. yscale belongs here, not to the callable: a row
    has to be scale-uniform, or its columns cannot be read against each other.
    """
    label: str                               # ylabel of column 0, carrying the panel letter
    yscale: str
    metric: Optional[str] = None
    metric_for: Optional[Callable] = None
    forced: bool = False                     # forced-response row: scored for every method, not just the operator ones
    note: bool = False                       # carries the dropped-fits note above its panels
    truth: Optional[str] = None              # results column of the true value, drawn as a dashed line
    truth_label: str = "truth"               # that line's legend entry

    def panel(self, study, lag):
        return (self.metric, None) if self.metric_for is None else self.metric_for(study, lag)

    def methods(self, run):
        """{method: legend label} for this row, in METHOD_ORDER."""
        if self.forced:
            return {m: m for m in run.methods}
        return {m: label for m, label in OPERATOR_METHODS.items() if m in run.methods}


ROW_SPECS = (
    RowSpec("(a) forced relative RMSE", "linear", metric="forced_rel_rmse", forced=True),
    RowSpec("(b) slow-mode decay time\n" + r"$\hat\tau_1$ (yr)", "log", metric_for=spectral_metric,
            truth="tau1_yr", truth_label=r"true $\tau_1$", note=True),
    RowSpec("(c) mode shape correlation", "linear", metric_for=shape_metric),
)
ANNOTATION_HEADROOM = 0.16  # fraction of the y-range left clear above the curves for a panel annotation
HEADER_INCHES = 0.35        # strip kept clear at the top of the figure for the run legend or suptitle
FOOTNOTE_FONTSIZE = 7
FOOTNOTE = ("median over realizations, shaded: interquartile range. Row (a) divides by the time-centered RMS of "
            "the true forced response, the same $V^{(f)}$ the SNR is built from; its numerator is uncentered, so a "
            "constant offset in the estimate still counts. Rows (b) and (c) score the eigen-decomposition of the "
            "fitted lag-step propagator; LIM-opt's forced estimate uses the SVD of its powered propagator, not "
            r"these modes. Row (b) is the fitted decay time itself against the true $\tau_1$ (dashed), so a curve "
            "above the line is too slow and one below it too fast. Row (c) is the correlation between the fitted "
            "mode's span and the true one -- the cosine of their largest principal angle -- so 1 is exact recovery "
            "of the pattern up to sign and scale; each panel is annotated with the subspace it scores")
# appended only when a PAIR_STUDIES column is drawn, which --studies decides: without one the row scores the
# slow mode in every panel and naming an exception sends the reader looking for a column that is not there
PAIR_CLAUSE = ", the slow mode everywhere except the pair study"


def footnote_height(fig, text=FOOTNOTE, fontsize=FOOTNOTE_FONTSIZE):
    """Fraction of the figure height the wrapped footnote needs, as a rect bottom for tight_layout.

    matplotlib wraps the text at draw time and reports no height for it, so the line count is estimated
    from the figure width at ~0.5 em per character. The reservation has to be in inches rather than a
    fixed fraction: the figure's height follows the row count and its width the number of studies, so a
    fraction that clears the footnote under one --studies selection buries the x labels under another.
    """
    lines = max(1, int(np.ceil(len(text) / (fig.get_figwidth() * 72 / (0.5 * fontsize)))))
    return (lines * 1.3 * fontsize / 72 + 0.12) / fig.get_figheight()


def row_legend(row_axes, **kwargs):
    """One legend for a whole row, pooled from its panels so no label is missed or repeated.

    Kept out of the layout: it hangs off the last panel, so tight_layout would otherwise narrow every
    column to make room for it inside the rect -- on top of the LEGEND_INCHES already reserved for it,
    which is what the strip is for.
    """
    entries = {}
    for ax in row_axes:
        handles, labels = ax.get_legend_handles_labels()
        entries.update(dict(zip(labels, handles)))
    if entries:
        row_axes[-1].legend(entries.values(), entries.keys(), **kwargs).set_in_layout(False)


def draw_panel(ax, spec, run, df, study, linestyle=None):
    """One run in one panel. linestyle None draws the reference: thin dotted medians, no legend entries."""
    metric, _ = spec.panel(study, run.cfg.lag)
    if metric not in df:
        return                               # a legacy run that never scored this metric: leave the row empty
    for method, label in spec.methods(run).items():
        if linestyle is None:
            plot_reference(ax, df, metric, method)
        else:
            plot_band(ax, df, metric, method, label=label, linestyle=linestyle)
    if spec.truth is not None:
        # one value per level by construction, so the median is that value; drawn per run, because the
        # truth is a property of the run's config and two runs need not share it
        truth = df.groupby("param_value")[spec.truth].median()
        if linestyle is None:
            ax.plot(truth.index, truth.values, color="k", **REFERENCE_STYLE)
        else:
            ax.plot(truth.index, truth.values, label=spec.truth_label, **TRUTH_STYLE)


def default_level(study, cfg):
    """The x value of the default system in a study's sweep, or None when the study has no single default.

    Every sweep bends one parameter away from the same starting point, so each column has one level that is
    the reference system itself. Marking it is what lets a column be read as a departure from the default
    rather than as an unanchored curve, and it is the level every other figure of the run describes.
    """
    if study == "total_snr":
        return 1 / sum(cfg.variances)        # the reference SNR is the variance shares' reciprocal (config.py)
    if study.startswith("partial_snr_"):
        return 1.0                           # factor 1 leaves the component at the equal budget
    if study.startswith("slow_timescale_"):
        return cfg.tau1_yr
    if study == "spatial_overlap":
        return build_reference(cfg).base_overlap
    if study == "forcing_overlap":
        return build_reference(cfg).base_b_overlap
    if study == "noise_overlap":
        return 0.0                           # the reference's noisy modes are orthogonal to w_1 (make_W)
    return None


def plot_sweeps(runs, path, studies=None):
    """Write the sweep figure and return the studies it drew, left to right.

    One row per metric (ROW_SPECS), one column per study, so a row reads as one quantity across the
    ablations. Studies absent from every run are dropped, and the order given is the order drawn.
    """
    reference, variants = (runs[0], runs[1:]) if len(runs) > 1 else (None, runs)
    studies = [s for s in (studies or DEFAULT_STUDIES) if any(s in set(r.results.study) for r in runs)]
    if not studies:
        return []
    n_rows, n_cols = len(ROW_SPECS), len(studies)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(COLUMN_W * n_cols + LEGEND_INCHES, ROW_H * n_rows),
                             squeeze=False)
    for j, study in enumerate(studies):
        frames = {run.name: run.results[run.results.study == study] for run in runs}
        for i, spec in enumerate(ROW_SPECS):
            ax = axes[i, j]
            if reference is not None and not frames[reference.name].empty:
                draw_panel(ax, spec, reference, frames[reference.name], study)
            for run, linestyle in zip(variants, VARIANT_LINESTYLES):
                if not frames[run.name].empty:
                    draw_panel(ax, spec, run, frames[run.name], study, linestyle=linestyle)
            ax.set_xscale("linear" if study in LINEAR_X else "log")   # per column: a property of the study
            ax.set_yscale(spec.yscale)                                # per row: a property of the metric
            ax.grid(alpha=0.3, linewidth=0.5)
            default = default_level(study, runs[0].cfg)                # runs[0]: the reference when comparing
            if default is not None:
                # axvline leaves the y limits alone, so this is safe before the annotation headroom below
                ax.axvline(default, label="default system", **DEFAULT_LINE_STYLE)
            _, detail = spec.panel(study, runs[0].cfg.lag)
            if detail is not None:
                if spec.yscale == "linear":
                    lo, hi = ax.get_ylim()
                    ax.set_ylim(lo, hi + ANNOTATION_HEADROOM * (hi - lo))
                ax.text(0.03, 0.96, detail, transform=ax.transAxes, ha="left", va="top",
                        fontsize=7, color="0.35")
            if spec.note:
                # tight_layout counts this text into the row gap, and row (b) carries no title for it to
                # overlap; runs are stacked with a newline so several of them grow it downward rather
                # than sideways into the neighbouring columns
                note = "\n".join(filter(None, (unstable_note(frames[r.name]) for r in runs)))
                ax.text(0.5, 1.02, note, transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=6.5, color="0.35")
        axes[0, j].set_title(SWEEP_STUDIES[study])
        axes[-1, j].set_xlabel(next(f.param_name.iloc[0] for f in frames.values() if not f.empty))
    for spec, ax in zip(ROW_SPECS, axes[:, 0]):
        ax.set_ylabel(spec.label)
    # one legend per kind of row, outside the right edge and top-aligned with the first row of its kind
    legend_rows = {}
    for i, spec in enumerate(ROW_SPECS):
        legend_rows.setdefault(spec.forced, i)
    for i in legend_rows.values():
        row_legend(axes[i], loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    if reference is not None:
        run_handles = [Line2D([], [], color="k", **REFERENCE_STYLE)] + [
            Line2D([], [], color="k", linestyle=ls) for _, ls in zip(variants, VARIANT_LINESTYLES)]
        run_labels = [f"{reference.label()} (reference)"] + [r.label(reference) for r in variants]
        fig.legend(run_handles, run_labels, loc="upper center", ncol=len(runs), frameon=False)
    else:
        fig.suptitle(variants[0].label())
    footnote = FOOTNOTE + (PAIR_CLAUSE if any(s in PAIR_STUDIES for s in studies) else "") + "."
    fig.text(0.5, 0.004, footnote, ha="center", fontsize=FOOTNOTE_FONTSIZE, wrap=True)
    header = HEADER_INCHES / fig.get_figheight()
    fig.tight_layout(rect=(0, footnote_height(fig, footnote), legend_rect(fig), 1 - header), h_pad=1.6)
    save(fig, path, tight=False, bbox="tight")
    return studies


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
    """One panel per method: the forced relative RMSE over the SNR x tau_1 grid, shaded and contoured.

    The shading and the lines are the same quantity at the same levels, so the labelled contours are the
    colorbar read back onto the panel, and the figure needs no key outside itself.
    """
    df = joint_results(run)
    if df.empty:
        return
    fig, axes = phase_axes(1, len(run.methods))
    for ax, method in zip(axes[0], run.methods):
        grid = phase_grid(df[df.method == method], PHASE_METRIC)
        centers = label_phase_axes(ax, grid, method)
        # explicit levels rather than vmin/vmax: every panel shares one scale by construction, and
        # extend="both" still colours the cells that fall outside the ladder
        filled = ax.contourf(*centers, grid.values, levels=RMSE_LEVELS, cmap=PHASE_CMAP, extend="both")
        lines = ax.contour(*centers, grid.values, levels=RMSE_LEVELS, colors="k", linewidths=1.0)
        ax.clabel(lines, fmt="%.2g", fontsize=6.5, inline=True)
        ax.set_xlabel(r"$\tau_1$ (yr)")
    axes[0, 0].set_ylabel("SNR")
    fig.colorbar(filled, ax=axes[0], ticks=RMSE_LEVELS, label=PHASE_LABEL, fraction=0.03, pad=0.02)
    fig.suptitle(f"{run.label()}\nmedian over realizations", y=PHASE_SUPTITLE_Y)
    save(fig, path, tight=False, bbox="tight")


def plot_phase_change(reference, variant, path):
    ref, var = joint_results(reference), joint_results(variant)
    methods = [m for m in variant.methods if m in reference.methods]
    if ref.empty or var.empty or not methods:
        return
    fig, axes = phase_axes(1, len(methods))
    # blue = variant better, i.e. lower RMSE. The delta has no fixed ladder to label against -- its range is
    # a property of the pair of runs -- so it stays a mesh with symmetric limits rather than labelled contours.
    deltas = {m: phase_grid(var[var.method == m], PHASE_METRIC) - phase_grid(ref[ref.method == m], PHASE_METRIC)
              for m in methods}
    limit = np.nanmax([np.abs(delta.values).max() for delta in deltas.values()]) or 1.0
    for ax, method in zip(axes[0], methods):
        mesh = ax.pcolormesh(deltas[method].values, cmap=PHASE_CHANGE_CMAP, vmin=-limit, vmax=limit)
        label_phase_axes(ax, deltas[method], method)
        ax.set_xlabel(r"$\tau_1$ (yr)")
    axes[0, 0].set_ylabel("SNR")
    fig.colorbar(mesh, ax=axes[0], label=f"change in {PHASE_LABEL}", fraction=0.03, pad=0.02)
    fig.suptitle(f"{variant.label(reference)} minus {reference.name}: change in median (blue: better)",
                 y=PHASE_SUPTITLE_Y)
    save(fig, path, tight=False, bbox="tight")


def forcing_change_datasets(runs):
    """{run name: dataset} for each run's own config, one realization (the forced response is noise free)."""
    return {run.name: reference_dataset(replace(run.cfg, n_realizations=1)) for run in runs}


def plot_forcing_change(reference, variant, path):
    """What changing the forcing's shape does: the forcing, the response it drives, and who notices.

    The point of the third panel is the contrast between a method that fits the dynamics and one that
    only regresses the data on the forcing. A deeper mid-century dip is a feature in time that the
    forcing and the forced response share, so LR can fit it directly; whether PullbackDMDc gains or
    loses against that is the question the figure is for, and it cannot be read off the phase diagrams.
    """
    datasets = forcing_change_datasets([reference, variant])
    total = {run.name: run.results[run.results.study == "total_snr"] for run in (reference, variant)}
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 3.6))
    # REFERENCE_STYLE already carries a linewidth, so neither style takes one at the call site
    styles = {reference.name: dict(REFERENCE_STYLE, color="0.35"),
              variant.name: dict(linestyle="-", color="C0", linewidth=1.6)}
    labels = {reference.name: reference.label(), variant.name: variant.label(reference)}

    for name, ds in datasets.items():
        record = record_years(ds.system)
        axes[0].plot(record, ds.y[ds.system.spinup:], label=labels[name], **styles[name])
        axes[1].plot(record, standardized(global_mean(ds.forced)), **styles[name])
    zero_line(axes[0])
    zero_line(axes[1])
    axes[0].set_title("(a) forcing $y$, centered on the record")
    axes[0].set_ylabel("forcing")
    axes[1].set_title("(b) forced response, global mean")
    axes[1].set_ylabel("standardized")
    for ax in axes[:2]:
        ax.set_xlabel("year")
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)

    for name, df in total.items():
        if df.empty:
            continue
        for method in (m for m in FORCING_CHANGE_METHODS if m in set(df.method)):
            if name == reference.name:
                plot_reference(axes[2], df, "forced_rel_rmse", method)
            else:
                plot_band(axes[2], df, "forced_rel_rmse", method, label=method)
    axes[2].set_xscale("log")
    axes[2].grid(alpha=0.3, linewidth=0.5)
    axes[2].set_xlabel("SNR")
    axes[2].set_ylabel("forced relative RMSE")
    axes[2].set_title("(c) total SNR sweep")
    axes[2].legend(loc="upper right", fontsize=7, frameon=False)

    # title above and note below the axes, as in plot_slow_mode_shapes: reserving a fraction of a figure
    # this short for a two-line title leaves a band of empty canvas that bbox="tight" will not crop
    fig.tight_layout()
    fig.suptitle(f"{labels[variant.name]} against {reference.name}: forcing, forced response and skill",
                 fontsize=10, y=1.06)
    fig.text(0.5, -0.08, "dotted: the reference run. (b) is standardized, so only the shapes are compared. "
             "(c) is the median over realizations, shaded: interquartile range.",
             ha="center", fontsize=7.5, color="0.35")
    save(fig, path, tight=False, bbox="tight")


def mode_shape_studies(run, studies=None):
    """The studies the shapes figure draws: the one-dimensional sweeps the run actually measured.

    joint_snr_timescale is excluded because its param_value is only one of its two axes, so a line per
    param_value would silently average over the other.
    """
    measured = set(run.modes.study)
    return [s for s in (studies or DEFAULT_STUDIES) if s != "joint_snr_timescale" and s in measured]


def median_pattern(frame):
    """The elementwise median over realizations of patterns already signed and normalized by the runner."""
    return np.median(np.stack(frame.pattern.to_numpy()), axis=0)


def median_residual(fits, truth):
    """The median fitted pattern's departure from the truth of its own level, elementwise.

    The truth is one vector per level, so differencing it out commutes with the median over realizations
    and this is also the median of the per-realization residuals.
    """
    return median_pattern(fits) - truth


def plot_slow_mode_shapes(run, path, studies=None):
    """The fitted slow mode's departure from the truth, $\hat w_1 - w_1$, per study.

    Row (c) of the sweeps figure scores this with one correlation per level, which says how wrong a fit
    is but never how. This says how: whether a degraded fit flattens the pattern, tilts it, or picks up
    a different mode entirely. The residual rather than the pattern, because the fits sit close to the
    truth: drawn as patterns the whole panel is spent on the shape they share and the deviation the
    figure exists to show is the thin gap between them. Differencing also absorbs the truth's own motion
    with the level in the two overlap studies -- every level is read against its own $w_1$ -- so one zero
    line replaces a truth curve per level.
    """
    if run.modes is None:
        return []
    studies = mode_shape_studies(run, studies)
    methods = [m for m in OPERATOR_METHODS if m in set(run.modes.method)]
    if not studies or not methods:
        return []
    lat = np.rad2deg(PHI)
    # COLUMN_W per column and ROW_H per row, the same grid plot_sweeps is on, so the columns of the two
    # figures line up when they are read together; the colorbars go under their column rather than beside
    # it, which would take the width back out of the panel and break that alignment, and the legend strip
    # is reserved here too although this figure has no legend, so that its columns keep the same pitch
    fig, axes = plt.subplots(len(methods), len(studies), squeeze=False,
                             figsize=(COLUMN_W * len(studies) + LEGEND_INCHES, ROW_H * len(methods)),
                             sharex=True, layout="constrained")
    reserve_legend_strip(fig)
    for j, study in enumerate(studies):
        frame = run.modes[run.modes.study == study]
        levels = sorted(set(frame.param_value))
        norm = (Normalize if study in LINEAR_X else LogNorm)(vmin=min(levels), vmax=max(levels))
        mappable = cm.ScalarMappable(norm=norm, cmap=MODE_SHAPE_CMAP)
        truth = {level: median_pattern(frame[(frame.param_value == level) & (frame.method == "truth")])
                 for level in levels}
        for i, method in enumerate(methods):
            ax = axes[i, j]
            for level in levels:
                fits = frame[(frame.param_value == level) & (frame.method == method)]
                if not fits.empty:
                    ax.plot(lat, median_residual(fits, truth[level]), color=mappable.to_rgba(level),
                            linewidth=1.4)
            zero_line(ax)  # an exact fit is the flat line; the panels are read as departures from it
            ax.grid(alpha=0.3, linewidth=0.5)
            # the residuals are O(1e-3), so plain tick labels run to "0.0125" and the extra width shifts
            # every column right of the matching sweeps column; a shared exponent keeps them two or three
            # characters wide, which is what lets the two figures' columns line up
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.yaxis.get_offset_text().set_fontsize(PANEL_TITLE)
        # one line, the same heading plot_sweeps gives this column; the level's name labels the colorbar,
        # which is where the sweeps figure puts it too (as the x label of the matching column)
        axes[0, j].set_title(SWEEP_STUDIES[study])
        bar = fig.colorbar(mappable, ax=axes[:, j].tolist(), ticks=levels, location="bottom",
                           fraction=0.06, pad=0.02)
        bar.ax.minorticks_off()
        bar.ax.set_xticklabels(level_ticklabels(levels, norm), fontsize=PANEL_TITLE)
        bar.set_label(frame.param_name.iloc[0], fontsize=PANEL_TITLE)
    for i, method in enumerate(methods):
        axes[i, 0].set_ylabel(OPERATOR_METHODS[method])
    for ax in axes[-1]:
        ax.set_xlabel("latitude (deg)", fontsize=PANEL_TITLE)
    # Each panel autoscales to its own residuals: they span orders of magnitude between methods, so one
    # shared scale leaves every panel but the worst an empty strip, and forcing each panel symmetric about
    # zero spends half of it on empty space whenever the residual sits to one side -- which is itself part
    # of the result. The cost is that heights are not comparable across panels; the footnote says so and the
    # tick labels on every panel make it visible.
    # title above and note below the axes rather than inside them: bbox="tight" grows the canvas to
    # include both, where a two-line suptitle would sit on top of the two-line column headings
    fig.suptitle(f"{run.label()}: slow-mode pattern error $\\hat w_1 - w_1$, median over realizations",
                 fontsize=10, y=1.04)
    fig.text(0.5, -0.04, "each level is differenced against the true $w_1$ of that level, which matters "
             "when a study moves the truth with the level, as the overlap studies do; grey: zero, an exact "
             "pattern. "
             "Every panel sets its own y scale, so read the shapes within a panel and the numbers, not the "
             "heights, between them. Patterns are unit norm and signed to agree with the truth, so they can "
             "be averaged over realizations. LIM and LIM-opt fit the same lag-step operator, so their modes "
             "coincide.", ha="center", va="top", fontsize=7.5, color="0.35", wrap=True)
    save(fig, path, tight=False, bbox="tight")
    return studies


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["baseline"], help="run names; the first is the reference")
    parser.add_argument("--studies", nargs="+", default=list(DEFAULT_STUDIES), choices=list(SWEEP_STUDIES),
                        metavar="STUDY", help=f"sweep columns, left to right (default: {' '.join(DEFAULT_STUDIES)})")
    args = parser.parse_args()

    runs = [load_run(name) for name in args.runs]
    out_dir = FIGURES_DIR / (runs[0].name if len(runs) == 1 else "compare_" + "_vs_".join(r.name for r in runs))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_sweeps(runs, out_dir / "results_sweeps.png", studies=args.studies)
    for run in runs:
        plot_phase(run, out_dir / f"results_phase_snr_timescale_{run.name}.png")
        plot_slow_mode_shapes(run, out_dir / f"results_slow_mode_shapes_{run.name}.png", studies=args.studies)
    if len(runs) == 2:
        plot_phase_change(*runs, out_dir / "results_phase_change.png")
        plot_forcing_change(*runs, out_dir / "results_forcing_change.png")


if __name__ == "__main__":
    main()

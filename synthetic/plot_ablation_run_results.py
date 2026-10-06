"""Figures for ablation runs (results/<run>/ from run_ablation_studies.py), written to figures/ablations/.

    python plot_ablation_run_results.py --runs baseline                # figures/ablations/baseline/
    python plot_ablation_run_results.py --runs baseline slow50         # figures/ablations/compare_baseline_vs_slow50/
    python plot_ablation_run_results.py --runs baseline --studies total_snr spatial_overlap

results_sweeps.png is one row per evaluation metric, (a) to (d), and one column per ablation study.
The columns are DEFAULT_STUDIES; --studies picks any other selection of SWEEP_STUDIES, left to right.

With several runs the first is the reference: thin dotted medians under the later runs' solid lines and bands.
With exactly two runs a third figure maps the change in the phase diagram (later run minus reference).
"""

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable, Optional

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
# every study the figure can show, with the title its column carries
SWEEP_STUDIES = {
    "total_snr": "total SNR",
    "partial_snr_slow": "partial SNR: slow",
    "partial_snr_pair": "partial SNR: pair",
    "partial_snr_complement": "partial SNR: complement",
    "slow_timescale_snr": r"slow timescale ($\mathrm{SNR}$ fixed)",
    "slow_timescale_modal_variance": r"slow timescale ($\sigma_1^2$ fixed)",
    "spatial_overlap": "spatial overlap",
}
# the columns of the default figure, left to right; --studies overrides it
DEFAULT_STUDIES = ("total_snr", "partial_snr_slow", "partial_snr_pair",
                   "partial_snr_complement", "slow_timescale_modal_variance")
PAIR_STUDIES = {"partial_snr_pair"}  # studies scored on the oscillating pair rather than the slow mode
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


def spectral_metric(study=None, lag=None):
    """Row (c): recovery of the slow mode's e-folding time, in every study.

    Not |lambda_1 - lambda_1|: tau = -1 / log|lambda| is steeply nonlinear and asymmetric near lambda = 1,
    so an absolute eigenvalue error neither ranks fits the way the decay time does nor keeps its meaning
    across levels. It compresses exactly the large errors -- at tau_1 = 20 yr a seemingly tiny 0.0047 in
    lambda is a factor of ~130 in tau. The log ratio is also lag-invariant, where lambda_1 ** lag is not.

    The quantity is the same in every column, so the symbol lives in the row label and no panel is
    annotated: the second return value is the per-panel annotation, and there is none.
    """
    return "slow_tau_log_ratio", None


def shape_metric(study, lag=None):
    """Row (d): the principal angle between a recovered mode subspace and the true one, in degrees.

    Every study scores the slow mode except PAIR_STUDIES, which score the plane of the oscillating pair,
    so this row is not one quantity and its shared label cannot name both. The second return value is the
    symbol annotated inside that panel; every panel in the row carries one, so no column is a silent
    exception to the row label.
    """
    if study in PAIR_STUDIES:
        return "pair_angle", r"$\angle(\hat w_2 \hat w_3,\, w_2 w_3)$"
    return "slow_angle", r"$\angle(\hat w_1, w_1)$"


def unstable_note(df):
    """Per-method share of fits whose slow eigenvalue does not decay (|lambda| >= 1), or '' when there are none.

    A trend the fit cannot attribute to the forcing has to go somewhere, and in a propagator it goes into a
    non-decaying slow mode. Such a fit has no e-folding time at all, so row (c) drops it: where the share
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
    forced: bool = False                     # forced-response row: every method, plus the oracle curve
    note: bool = False                       # carries the dropped-fits note above its panels

    def panel(self, study, lag):
        return (self.metric, None) if self.metric_for is None else self.metric_for(study, lag)

    def methods(self, run):
        """{method: legend label} for this row, in METHOD_ORDER."""
        if self.forced:
            return {m: m for m in run.methods}
        return {m: label for m, label in OPERATOR_METHODS.items() if m in run.methods}


ROW_SPECS = (
    RowSpec("(a) forced pattern correlation", "linear", metric="forced_corr", forced=True),
    RowSpec("(b) forced relative RMSE", "linear", metric="forced_rel_rmse", forced=True),
    RowSpec("(c) slow-mode decay time\n" + r"$|\log(\hat\tau_1 / \tau_1)|$", "log",
            metric_for=spectral_metric, note=True),
    RowSpec("(d) mode shape: principal angle (deg)", "linear", metric_for=shape_metric),
)
ANNOTATION_HEADROOM = 0.16  # fraction of the y-range left clear above the curves for a panel annotation


def row_legend(row_axes, **kwargs):
    """One legend for a whole row, pooled from its panels so no label is missed or repeated."""
    entries = {}
    for ax in row_axes:
        handles, labels = ax.get_legend_handles_labels()
        entries.update(dict(zip(labels, handles)))
    if entries:
        row_axes[-1].legend(entries.values(), entries.keys(), **kwargs)


def draw_panel(ax, spec, run, df, study, linestyle=None):
    """One run in one panel. linestyle None draws the reference: thin dotted medians, no legend entries."""
    metric, _ = spec.panel(study, run.cfg.lag)
    for method, label in spec.methods(run).items():
        if linestyle is None:
            plot_reference(ax, df, metric, method)
        else:
            plot_band(ax, df, metric, method, label=label, linestyle=linestyle)
    if not spec.forced:
        return
    oracle = df[df.method == "oracle"].sort_values("param_value")
    if linestyle is None:
        ax.plot(oracle.param_value, oracle[metric], color="0.5", **REFERENCE_STYLE)
    else:
        ax.plot(oracle.param_value, oracle[metric], label="oracle (true A, b)", **ORACLE_STYLE)


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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.5 * n_rows), squeeze=False)
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
            _, detail = spec.panel(study, runs[0].cfg.lag)
            if detail is not None:
                if spec.yscale == "linear":
                    lo, hi = ax.get_ylim()
                    ax.set_ylim(lo, hi + ANNOTATION_HEADROOM * (hi - lo))
                ax.text(0.03, 0.96, detail, transform=ax.transAxes, ha="left", va="top",
                        fontsize=7, color="0.35")
            if spec.note:
                # tight_layout counts this text into the row gap, and row (c) carries no title for it to
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
    fig.text(0.5, 0.004, "median over realizations, shaded: interquartile range. Rows (c) and (d) score the "
             "eigen-decomposition of the fitted lag-step propagator; LIM-opt's forced estimate uses the SVD of its "
             "powered propagator, not these modes. Row (d) is the largest principal angle between the fitted mode's "
             "span and the true one, so 0 deg is exact recovery of the pattern up to sign and scale; each panel is "
             "annotated with the subspace it scores, the slow mode everywhere except the pair study.",
             ha="center", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965), h_pad=1.6)
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
    parser.add_argument("--studies", nargs="+", default=list(DEFAULT_STUDIES), choices=list(SWEEP_STUDIES),
                        metavar="STUDY", help=f"sweep columns, left to right (default: {' '.join(DEFAULT_STUDIES)})")
    args = parser.parse_args()

    runs = [load_run(name) for name in args.runs]
    out_dir = FIGURES_DIR / (runs[0].name if len(runs) == 1 else "compare_" + "_vs_".join(r.name for r in runs))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_sweeps(runs, out_dir / "results_sweeps.png", studies=args.studies)
    for run in runs:
        plot_phase(run, out_dir / f"results_phase_snr_timescale_{run.name}.png")
    if len(runs) == 2:
        plot_phase_change(*runs, out_dir / "results_phase_change.png")


if __name__ == "__main__":
    main()

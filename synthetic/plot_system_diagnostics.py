"""System diagnostics for any config or sweep level, written to figures/diagnostics/system/<name>/.

    python plot_system_diagnostics.py                                         # default reference system
    python plot_system_diagnostics.py --set tau1_yr=50 slow_variance=0.5      # tweaked reference system
    python plot_system_diagnostics.py --study slow_timescale_snr --level 100  # one sweep dataset
    python plot_system_diagnostics.py --from-run baseline --plots modal_overview
    python plot_system_diagnostics.py --plots forced_response_ablations      # one column per ablation study
    python plot_system_diagnostics.py --plots forced_response_ablations --studies total_snr forcing_overlap
    python plot_system_diagnostics.py --plots forced_response_shape           # forced-response shape against the forcing

Every forcing shown is centered on the record (see README.md).

forced_response_ablations draws the columns of results_sweeps.png (plot_ablation_run_results.py), in the same
order and at the same width, so the two figures stack.
"""

import argparse
import ast
import pathlib
from dataclasses import replace
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize

from ablation_data import (
    DEFAULT_STUDIES,
    LINEAR_X,
    M,
    PARTIAL_SNR_COMPONENTS,
    PHI,
    SLOW_TIMESCALE_HOLDS,
    STUDIES,
    SWEEP_STUDIES,
    annual_years,
    build_reference,
    centered_forcing,
    load_forcing_file,
    config_budget,
    decay_time_yr,
    drive_modal,
    eigenvalue_yr,
    find_level,
    forcing_series,
    gauss_params,
    global_mean,
    make_dataset,
    record_years,
    spinup_years,
)
from config import DEFAULT, config_diff, load_config, slug
from plot_ablation_diagnostics import annual, lat_label, standardized
from plot_style import (
    COLUMN_W,
    LEGEND_INCHES,
    PAGE_W,
    PANEL_TITLE,
    ROW_H,
    forcing_panel,
    level_ticklabels,
    record_span,
    reserve_legend_strip,
    residual_panel,
    save,
    zero_line,
)

LAT = np.rad2deg(PHI)
FIGURES_DIR = pathlib.Path(__file__).resolve().parent / "figures" / "diagnostics" / "system"
MODAL_OVERVIEW_SHOWN = 10
N_COLS = 5
LONG_TAU1_YR = 100  # the longest slow timescale in the sweeps, so the longest spin-up
ABLATION_CMAP = "viridis"  # sequential: these lines are ordered by their level, not categories


def gauss_label(system):
    _, (bump_amp, bump_year, _), (dip_amp, dip_year, _) = gauss_params(system)
    terms = [f"+{bump_amp:.2g} at {bump_year:.0f}" if bump_amp else "",
             f"$-${dip_amp:.2g} at {dip_year:.0f}" if dip_amp else ""]
    return "exp + Gaussians" + (f" ({', '.join(filter(None, terms))})" if any(terms) else "")


def centered_curve(system, source, t_yr):
    """`source`'s forcing at t_yr, centered on the system's record."""
    return centered_forcing(replace(system, forcing_source=source), t_yr)


def forcing_curves(system, t_yr):
    """{source: (centered values at t_yr, label)} for the two analytic models; the file is file_curve."""
    return {
        "analytic": (centered_curve(system, "analytic", t_yr),
                     rf"exp: $c + a\,e^{{(t-2014)/{system.forcing_efold_yr:.3g}\,\mathrm{{yr}}}}$"),
        "analytic_gauss": (centered_curve(system, "analytic_gauss", t_yr), gauss_label(system)),
    }


def file_curve(system):
    """(monthly years, values centered on the record) of the system's forcing file."""
    t_month, _ = load_forcing_file(system.forcing_file, system.forcing_column)
    return t_month, centered_curve(system, "file", t_month)


def record_mask(t_yr, record):
    return (t_yr >= record[0]) & (t_yr <= record[-1])


def latitude_axes(fig, sharex=False):
    """The north-to-south grid of latitude panels as {grid index: ax}."""
    n_rows = M // N_COLS
    grid = fig.add_gridspec(n_rows, N_COLS)
    axes, first = {}, None
    for k, i in enumerate(range(M - 1, -1, -1)):
        ax = fig.add_subplot(grid[k // N_COLS, k % N_COLS], sharex=first if sharex else None)
        first = first or ax
        ax.set_title(lat_label(i), fontsize=PANEL_TITLE, pad=2)
        axes[i] = ax
    bottom = list(axes.values())[-N_COLS:]
    for ax in axes.values():
        if ax in bottom:
            ax.set_xlabel("year")
        elif sharex:
            ax.tick_params(labelbottom=False)  # plt.subplots(sharex=True) does this for us; add_subplot does not
    return axes


def plot_modal_overview(ds, out_path, n_shown=None, **_):
    """Global means (raw, forced, internal), spatial patterns, and the forcing that drove them."""
    n_shown = MODAL_OVERVIEW_SHOWN if n_shown is None else n_shown
    s = ds.system
    years, record = annual_years(s), record_years(s)
    forced_gm = annual(global_mean(ds.forced), axis=-1)
    raw_gm = annual(global_mean(ds.data[:n_shown]), axis=-1)
    internal_gm = annual(global_mean(ds.internal[:n_shown]), axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for k, member in enumerate(raw_gm):
        axes[0].plot(years, member, color="C0", linewidth=0.7, alpha=0.6,
                     label="raw signal" if k == 0 else None)
    for k, member in enumerate(internal_gm):
        axes[0].plot(years, member, color="C2", linewidth=0.7, alpha=0.5,
                     label="internal variability" if k == 0 else None)
    axes[0].plot(years, forced_gm, color="k", linewidth=2.5, zorder=4, label="forced response")
    zero_line(axes[0])
    axes[0].set_title(f"(a) global mean, {n_shown} of {len(ds.data)} realizations (annual means)")
    axes[0].set_xlabel("year")
    axes[0].legend()

    for j, name in enumerate((r"$w_1$ slow", r"$w_2$ pair", r"$w_3$ pair")):
        axes[1].plot(LAT, s.W[:, j], marker="o", label=name)
    axes[1].plot(LAT, s.b, marker="D", color="k", linestyle="--", label=r"$\hat b$ forcing")
    zero_line(axes[1])
    axes[1].set_xlabel("latitude (deg)")
    axes[1].set_title("(b) spatial patterns (unit norm)")
    axes[1].legend()

    axes[2].plot(record, ds.y[s.spinup:], color="C0", linewidth=1.2)
    zero_line(axes[2])
    axes[2].set_xlabel("year")
    axes[2].set_ylabel("forcing $y$, centered on the record")
    axes[2].set_title("(c) forcing timeseries")

    taus = decay_time_yr(s.eigvals)
    fig.suptitle(rf"$\tau_1$ = {taus[0]:.3g} yr, $\tau_p$ = {taus[1]:.3g} yr "
                 rf"(period {2 * np.pi / s.theta / 12:.3g} yr), SNR {ds.snr:.3g}")
    save(fig, out_path)


def plot_ensemble_super_spaghetti(ds, out_path, n_shown=None, label="", **_):
    """Every grid point: realizations (annual means) with the forced response on top; north to south, row by row."""
    fig = plt.figure(figsize=(PAGE_W, 1.25 * (M // N_COLS)))
    axes = latitude_axes(fig, sharex=True)
    data = ds.data if n_shown is None else ds.data[:n_shown]
    years = annual_years(ds.system)
    for i, ax in axes.items():
        ax.plot(years, annual(data[:, :, i], axis=-1).T, color="0.45", linewidth=0.4, alpha=0.25)
        ax.plot(years, annual(ds.forced[:, i], axis=-1), color="k", linewidth=1.8)
    title = f"{len(data)} of {len(ds.data)} realizations (annual means, grey) and forced response (black), SNR {ds.snr:.3g}"
    fig.suptitle(title + (f"\n{label}" if label else ""))
    save(fig, out_path)


def plot_spinup(ax, system, title):
    """Forcing and forced slow mode over the whole spin-up, as fractions of their rise over the record."""
    y = forcing_series(system)
    z1 = drive_modal(system, y)[:, 0]
    years = spinup_years(system)
    for series, name, style in ((z1, r"forced slow mode $z_1^{(f)}$", dict(color="C0")),
                                (y, "forcing $y$", dict(color="k", linestyle="--", zorder=3))):
        ax.plot(years, series / np.ptp(series[system.spinup:]), linewidth=1.2, label=name, **style)
    ax.axvspan(years[system.spinup], years[-1], color="0.9", zorder=0, label="record")
    zero_line(ax)
    ax.set_title(title)
    ax.set_xlabel("year")
    ax.set_ylabel("value / record range\n($y$ centered on the record)")
    ax.legend(loc="upper left")


def plot_forcing(ds, out_path, label="", **_):
    """The analytic forcing against the forcing file, the one in use in bold, and its spin-up: constant in the past."""
    s = ds.system
    t_month, data = file_curve(s)
    models = forcing_curves(s, t_month)
    record = record_years(s)
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, 7))

    curves = {"file": (data, f"file {pathlib.Path(s.forcing_file).name} [{s.forcing_column}], monthly"), **models}
    forcing_panel(axes[0, 0], t_month, curves, bold=s.forcing_source, bold_suffix=" (used)", width=1.0)
    axes[0, 0].set_title(f"(a) forcing_source = {s.forcing_source}; record shaded")
    residual_panel(axes[1, 0], t_month, data, models, record_mask(t_month, record))
    axes[1, 0].set_title("(b) analytic models minus file")
    axes[1, 0].set_xlabel("year")
    for ax in axes[:, 0]:
        record_span(ax, record)

    plot_spinup(axes[0, 1], s, rf"(c) this system: $\tau_1$ = {decay_time_yr(s.lam1):.3g} yr, "
                               f"spin-up {s.spinup / 12:.0f} yr")
    long = replace(s, lam1=eigenvalue_yr(LONG_TAU1_YR))
    plot_spinup(axes[1, 1], long, rf"(d) longest spin-up: $\tau_1$ = {LONG_TAU1_YR} yr, spin-up {long.spinup / 12:.0f} yr"
                                  " (the zero start decays first)")
    fig.suptitle(f"Forcing ({s.forcing_source}): analytic fit against the forcing file, and the constant past"
                 + (f"\n{label}" if label else ""))
    save(fig, out_path)


def level_datasets(cfg, study, levels):
    """{level: dataset} for the given levels of `study`, one realization each.

    One realization is enough: the forced response does not depend on the noise at all, and the data
    panel shows a single member by design rather than an ensemble.
    """
    one = replace(cfg, n_realizations=1)
    return {level: find_level(one, study, level)()[0] for level in levels}


def level_colors(levels, study):
    """(color per level, scalar mappable for the colorbar).

    Log unless the study's levels are cosines (LINEAR_X): the tau_1 and SNR level sets span decades, and
    a log norm cannot take the 0 the overlap sweeps start at and would misrepresent them anyway. It is
    the rule that sets the sweep figure's x scale, so a study's levels read the same way in both figures.
    """
    norm = (Normalize if study in LINEAR_X else LogNorm)(vmin=min(levels), vmax=max(levels))
    mappable = cm.ScalarMappable(norm=norm, cmap=ABLATION_CMAP)
    return [mappable.to_rgba(level) for level in levels], mappable


def level_colorbar(fig, ax, mappable, levels, label):
    """A colorbar under the column, ticked at the sweep's own levels, not at decades.

    Under rather than beside, as in plot_slow_mode_shapes: a vertical bar takes its width out of the
    panel, which would leave the column narrower than the sweeps column it has to line up with. A log
    norm otherwise labels the bar 10^0, 4x10^0, ... which names values the sweep never ran; the levels
    are the only meaningful ticks, and there are few enough of them to print.
    """
    # aspect: the default 20 is a length cap, and under a single row 0.06 of the panel's height is thin
    # enough that the cap leaves the bar well short of its column
    bar = fig.colorbar(mappable, ax=ax, ticks=list(levels), location="bottom", fraction=0.06, pad=0.02,
                       aspect=40)
    bar.ax.minorticks_off()
    # 1/30 is 0.0333, not 0.0333333; a level crowded by its neighbour keeps its tick but not its label
    bar.ax.set_xticklabels(level_ticklabels(levels, mappable.norm), fontsize=PANEL_TITLE)
    bar.set_label(label, fontsize=PANEL_TITLE)  # the name the matching sweeps column carries as its x label
    return bar


def realization_panel(ax, cfg, study, levels, line_colors):
    """One noise realization per level, global mean, with the forced response in black.

    These studies leave the system alone and only move the noise budget, so the forced response is the
    same curve at every level -- the panel asserts it -- and what the sweep changes is how far that fixed
    signal sits inside one realization, which is what this draws.
    """
    datasets = level_datasets(cfg, study, levels)
    forced = [ds.forced for ds in datasets.values()]
    assert all(np.allclose(f, forced[0]) for f in forced), f"{study} changed the forced response"
    for ds, color in zip(datasets.values(), line_colors):
        ax.plot(annual_years(ds.system), annual(global_mean(ds.data[0]), axis=-1),
                color=color, linewidth=0.9, alpha=0.85)
    fixed = next(iter(datasets.values()))
    ax.plot(annual_years(fixed.system), annual(global_mean(fixed.forced), axis=-1),
            color="k", linewidth=2.2, zorder=5)
    ax.set_xlabel("year")
    return "data (global mean)\nblack: the forced response"


def forced_panel(ax, cfg, study, levels, line_colors):
    """The forced response per level, global mean: these studies move lambda_1, so it moves with them."""
    for ds, color in zip(level_datasets(cfg, study, levels).values(), line_colors):
        ax.plot(annual_years(ds.system), annual(global_mean(ds.forced), axis=-1),
                color=color, linewidth=1.6)
    ax.set_xlabel("year")
    return "forced response (global mean)"


def pattern_panel(ax, cfg, study, levels, line_colors):
    """The slow mode w_1 against latitude per level: what these studies ablate is a shape.

    Each level's system comes from the study table itself (STUDIES[study].system), the same call the
    sweep's own builder makes, so these are its systems -- and none of them needs a dataset simulated.
    """
    reference = build_reference(cfg)
    if study == "forcing_overlap":
        # b-hat first and heavy, so the c = 1 level is seen landing on it rather than hiding it
        ax.plot(LAT, reference.system.b, color="k", linewidth=2.4, zorder=1)
    for level, color in zip(levels, line_colors):
        ax.plot(LAT, STUDIES[study].system(reference, level).W[:, 0], color=color, linewidth=1.6)
    ax.set_xlabel("latitude (deg)")
    return r"slow mode $w_1$" + (r", black: $\hat b$" if study == "forcing_overlap" else "")


def noise_pattern_panel(ax, cfg, study, levels, line_colors):
    """One noisy (complement) mode against latitude per level, with the slow mode w_1 in black.

    noise_overlap moves the noisy modes and leaves w_1 -- and, by its choice of signs, the forced
    response -- where they were, so the shape it ablates is theirs. All 17 tilt by the same angle; the one
    drawn is the first, whose sign balanced_signs fixes at +1, so its curves close on +w_1 rather than on
    -w_1. Its starting shape is a Haar-random draw and means nothing on its own: what the panel shows is
    how far toward the slow fingerprint the noise is tipped.
    """
    reference = build_reference(cfg)
    # w_1 first and heavy, as b-hat is in pattern_panel, so the c -> 1 levels are seen closing on it
    ax.plot(LAT, reference.system.W[:, 0], color="k", linewidth=2.4, zorder=1)
    for level, color in zip(levels, line_colors):
        ax.plot(LAT, STUDIES[study].system(reference, level).W[:, 3], color=color, linewidth=1.6)
    ax.set_xlabel("latitude (deg)")
    return r"a noisy mode $q_4$, black: $w_1$"


# One panel per study, keyed the way SWEEP_STUDIES is, so any column the sweep figure can draw this one
# can draw too. Four kinds cover the nine, and no two kinds show the same quantity: each panel's own
# docstring says why its studies get it.
ABLATION_PANELS = {
    "total_snr": realization_panel,
    **{f"partial_snr_{component}": realization_panel for component in PARTIAL_SNR_COMPONENTS},
    **{f"slow_timescale_{hold}": forced_panel for hold in SLOW_TIMESCALE_HOLDS},
    "spatial_overlap": pattern_panel,
    "forcing_overlap": pattern_panel,
    "noise_overlap": noise_pattern_panel,
}
# the fraction of the y range left clear above the curves for the annotation, as in plot_sweeps
ANNOTATION_HEADROOM = 0.22
ABLATION_FOOTNOTE = (
    "one realization per level, annual means, coloured by the level; the colorbar carries the levels the "
    "matching column of results_sweeps.png sweeps. A study that only rescales the noise budget leaves the "
    "forced response identical at every level (black, asserted), so its column draws the data instead; the "
    r"slow-timescale columns draw the forced response, which moves with $\tau_1$; the overlap columns are "
    r"spatial rather than time series, because what they ablate is a shape -- $w_1$ rotates toward $\hat b$ "
    "until the forcing drives the slow mode and nothing else, and the noisy modes tilt toward $w_1$ until "
    "only their timescale tells them apart from it."
)


def plot_forced_response_ablations(out_path, label="", cfg=DEFAULT, studies=None, **_):
    """What each ablation does to the data, one column per study; returns the studies drawn, left to right.

    The same studies in the same order as the sweep figure (DEFAULT_STUDIES, or the given selection) and
    the same column grid plot_sweeps and plot_slow_mode_shapes are on (COLUMN_W, ROW_H, LEGEND_INCHES in
    plot_style), so the three figures stack and read column by column: what the ablation changes in the
    data here, what it costs each method there.

    Each column gets the panel its study needs (ABLATION_PANELS), under the heading its sweeps column
    carries. The levels go on a colorbar beneath the panel, because the x axis is spent on the quantity
    itself, and what that quantity is goes inside the panel rather than on a y label: the ylabel is the
    sweeps figure's row label, and a column that carries one is narrower than the columns that do not.
    """
    studies = list(studies or DEFAULT_STUDIES)
    fig, axes = plt.subplots(1, len(studies), squeeze=False,
                             figsize=(COLUMN_W * len(studies) + LEGEND_INCHES, ROW_H), layout="constrained")
    reserve_legend_strip(fig)  # no legend of its own, but the strip keeps the columns on the sweeps' pitch
    reference = build_reference(cfg)
    for ax, study in zip(axes[0], studies):
        levels = list(STUDIES[study].levels(reference, cfg))
        line_colors, mappable = level_colors(levels, study)
        detail = ABLATION_PANELS[study](ax, cfg, study, levels, line_colors)
        zero_line(ax)
        ax.grid(alpha=0.3, linewidth=0.5)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + ANNOTATION_HEADROOM * (hi - lo))
        ax.text(0.03, 0.96, detail, transform=ax.transAxes, ha="left", va="top", fontsize=7, color="0.35")
        ax.set_title(SWEEP_STUDIES[study])
        level_colorbar(fig, ax, mappable, levels, STUDIES[study].param_name)
    axes[0, 0].set_ylabel("what the ablation changes")
    # title above and note below the axes rather than inside them, as in plot_slow_mode_shapes: bbox="tight"
    # grows the canvas to include both, where a suptitle inside it would sit on the column headings
    # one line, however long the label: the suptitle sits above the canvas at a fixed y, and a second line
    # grows downward from it onto the column headings
    fig.suptitle("The ablations: what each one does to the data" + (f" ({label})" if label else ""), y=1.04)
    fig.text(0.5, -0.06, ABLATION_FOOTNOTE, ha="center", va="top", fontsize=7.5, color="0.35", wrap=True)
    save(fig, out_path, tight=False, bbox="tight")
    return studies


def plot_forced_response_shape(ds, out_path, label="", **_):
    """Forced response against the forcing that drove it, both global means over the record and standardized,
    so only their shapes are compared (a slow mode lags the forcing and rounds its turns)."""
    s = ds.system
    record = record_years(s)
    forced, forcing = standardized(global_mean(ds.forced)), standardized(ds.y[s.spinup:])
    corr = np.corrcoef(forced, forcing)[0, 1]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(record, forcing, color="k", linewidth=1.6, linestyle="--", label="forcing $y$")
    ax.plot(record, forced, color="tab:blue", linewidth=1.8, label="forced response (global mean)")
    zero_line(ax)
    ax.set_xlabel("year")
    ax.set_ylabel("standardized")
    ax.legend(loc="upper left")
    ax.set_title(rf"Forced-response shape against the forcing, $\tau_1$ = {decay_time_yr(s.lam1):.3g} yr "
                 f"(correlation {corr:.4f})" + (f"\n{label}" if label else ""))
    save(fig, out_path)


# Each plotter takes (ds, out_path) and whatever of n_shown/label/cfg it needs, ignoring the rest;
# forced_response_ablations builds its own datasets, so it does not take ds at all.
DIAGNOSTICS = {
    "modal_overview": plot_modal_overview,
    "ensemble_super_spaghetti": plot_ensemble_super_spaghetti,
    "forcing": plot_forcing,
    "forced_response_ablations": lambda ds, out_path, **kw: plot_forced_response_ablations(out_path, **kw),
    "forced_response_shape": plot_forced_response_shape,
}


def reference_dataset(cfg):
    return make_dataset(build_reference(cfg).system, partial(config_budget, cfg=cfg),
                        n_realizations=cfg.n_realizations, seed=cfg.noise_seed)


def plot_diagnostics(ds, out_dir, plots=tuple(DIAGNOSTICS), n_shown=None, label="", cfg=DEFAULT, studies=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in plots:
        DIAGNOSTICS[name](ds, out_dir / f"{name}.png", n_shown=n_shown, label=label, cfg=cfg, studies=studies)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    parser.add_argument("--from-run", help="start from results/<run>/config.json")
    parser.add_argument("--study", help="plot one sweep dataset of this study instead of the reference")
    parser.add_argument("--level", help="the level within --study (a tuple 'snr,tau' for joint_snr_timescale)")
    parser.add_argument("--plots", nargs="*", choices=list(DIAGNOSTICS), default=list(DIAGNOSTICS))
    parser.add_argument("--studies", nargs="+", choices=list(SWEEP_STUDIES), metavar="STUDY",
                        help="forced_response_ablations columns, left to right, as in plot_ablation_run_results "
                             f"(default: {' '.join(DEFAULT_STUDIES)})")
    parser.add_argument("--n-shown", type=int, help=f"realizations drawn (default: {MODAL_OVERVIEW_SHOWN} in "
                                                     "modal_overview, all in ensemble_super_spaghetti)")
    parser.add_argument("--name", help="output folder under figures/diagnostics/system/")
    args = parser.parse_args()
    if (args.study is None) != (args.level is None):
        parser.error("--study and --level go together")

    cfg = load_config(args)
    label = config_diff(cfg)
    if args.study:
        level = ast.literal_eval(args.level)
        (ds,) = find_level(cfg, args.study, level)()
        label = ", ".join(filter(None, [label, f"{args.study} = {level}"]))
    else:
        ds = reference_dataset(cfg)
    name = args.name or args.from_run or slug(cfg)
    if args.study and not args.name:
        name += f"__{args.study}={args.level}".replace(",", "_").replace(" ", "")
    plot_diagnostics(ds, FIGURES_DIR / name, args.plots, args.n_shown, label, cfg, args.studies)


if __name__ == "__main__":
    main()

"""System diagnostics for any config or sweep level, written to figures/diagnostics/system/<name>/.

    python plot_system_diagnostics.py                                         # default reference system
    python plot_system_diagnostics.py --set tau1_yr=50 slow_variance=0.5      # tweaked reference system
    python plot_system_diagnostics.py --study slow_timescale_snr --level 100  # one sweep dataset
    python plot_system_diagnostics.py --from-run baseline --plots modal_overview
    python plot_system_diagnostics.py --plots forced_response_tau1            # forced responses at tau1 = 1, 20, 100 yr
    python plot_system_diagnostics.py --plots forced_response_shape           # forced-response shape against the forcing

Every forcing shown is centered on the record (see DATA_GENERATION.md).
"""

import argparse
import ast
import pathlib
from dataclasses import replace
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from ablation_data import (
    M,
    PHI,
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
    PAGE_W,
    PANEL_TITLE,
    TALL_ROW,
    forcing_panel,
    record_span,
    residual_panel,
    save,
    zero_line,
)

LAT = np.rad2deg(PHI)
FIGURES_DIR = pathlib.Path(__file__).resolve().parent / "figures" / "diagnostics" / "system"
MODAL_OVERVIEW_SHOWN = 10
N_COLS = 5
LONG_TAU1_YR = 100  # the longest slow timescale in the sweeps, so the longest spin-up
TAU1_COMPARED = (1, 20, 100)  # slow timescales (yr) of forced_response_tau1, levels of slow_timescale_snr
TAU1_COLORS = ("tab:orange", "k", "tab:blue")


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


def tau1_datasets(cfg):
    """{tau1_yr: dataset} at the TAU1_COMPARED levels of the slow_timescale_snr sweep (one realization: the forced
    response does not depend on the noise)."""
    one = replace(cfg, n_realizations=1)
    return {tau: find_level(one, "slow_timescale_snr", tau)()[0] for tau in TAU1_COMPARED}


def plot_forced_response_tau1(out_path, label="", cfg=DEFAULT, **_):
    """Only the forced responses of the slow_timescale_snr datasets at TAU1_COMPARED (annual means), tall panels."""
    datasets = tau1_datasets(cfg)
    fig = plt.figure(figsize=(PAGE_W, TALL_ROW * (M // N_COLS)))
    axes = latitude_axes(fig)
    for (tau, ds), color in zip(datasets.items(), TAU1_COLORS):
        years = annual_years(ds.system)
        for i, ax in axes.items():
            ax.plot(years, annual(ds.forced[:, i], axis=-1), color=color, linewidth=1.6,
                    label=rf"$\tau_1$ = {tau:g} yr ($V^{{(f)}}$ {ds.V_f:.3g})")
    handles, labels = axes[M - 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(datasets), frameon=False)
    fig.suptitle(r"Forced response at $\tau_1$ = " + ", ".join(f"{tau:g}" for tau in TAU1_COMPARED)
                 + " yr (slow_timescale_snr datasets, annual means)" + (f"\n{label}" if label else ""),
                 x=0.02, ha="left")
    save(fig, out_path)


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
# forced_response_tau1 builds its own datasets, so it does not take ds at all.
DIAGNOSTICS = {
    "modal_overview": plot_modal_overview,
    "ensemble_super_spaghetti": plot_ensemble_super_spaghetti,
    "forcing": plot_forcing,
    "forced_response_tau1": lambda ds, out_path, **kw: plot_forced_response_tau1(out_path, **kw),
    "forced_response_shape": plot_forced_response_shape,
}


def reference_dataset(cfg):
    return make_dataset(build_reference(cfg).system, partial(config_budget, cfg=cfg),
                        n_realizations=cfg.n_realizations, seed=cfg.noise_seed)


def plot_diagnostics(ds, out_dir, plots=tuple(DIAGNOSTICS), n_shown=None, label="", cfg=DEFAULT):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in plots:
        DIAGNOSTICS[name](ds, out_dir / f"{name}.png", n_shown=n_shown, label=label, cfg=cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    parser.add_argument("--from-run", help="start from results/<run>/config.json")
    parser.add_argument("--study", help="plot one sweep dataset of this study instead of the reference")
    parser.add_argument("--level", help="the level within --study (a tuple 'snr,tau' for joint_snr_timescale)")
    parser.add_argument("--plots", nargs="*", choices=list(DIAGNOSTICS), default=list(DIAGNOSTICS))
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
    plot_diagnostics(ds, FIGURES_DIR / name, args.plots, args.n_shown, label, cfg)


if __name__ == "__main__":
    main()

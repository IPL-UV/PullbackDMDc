"""System diagnostics for any config or sweep level, written to figures/diagnostics/system/<name>/.

    python plot_system.py                                         # default reference system
    python plot_system.py --set tau1_yr=50 slow_variance=0.5      # tweaked reference system
    python plot_system.py --study slow_timescale_snr --level 100  # one sweep dataset
    python plot_system.py --from-run baseline --plots modal_overview
"""

import argparse
import ast
import pathlib
from dataclasses import replace
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import plot_style  # noqa: F401  (sets the shared rcParams)
from ablations import (
    M,
    N,
    build_reference,
    co2_forcing_gauss_model,
    co2_forcing_model,
    load_forcing_file,
    config_budget,
    decay_time,
    eigenvalue,
    find_level,
    forcing_series,
    gauss_params,
    make_dataset,
    record_years,
    run_modal,
)
from config import DEFAULT, config_diff, from_json, slug, with_overrides
from plot_ablations import annual, lat_label
from test import YEARS, plot_modal_overview, save

SYNTHETIC_DIR = pathlib.Path(__file__).resolve().parent
FIGURES_DIR = SYNTHETIC_DIR / "figures" / "diagnostics" / "system"
RESULTS_DIR = SYNTHETIC_DIR / "results"
MODAL_OVERVIEW_SHOWN = 10
N_COLS = 5
LONG_TAU1_YR = 100  # the longest slow timescale in the sweeps, so the longest spin-up
FORCING_COLORS = {"file": "0.6", "analytic": "k", "analytic_gauss": "tab:blue"}


def gauss_label(system):
    _, (bump_amp, bump_year, _), (dip_amp, dip_year, _) = gauss_params(system)
    return f"exp + Gaussians (+{bump_amp:.3g} at {bump_year:.0f}, $-${dip_amp:.3g} at {dip_year:.0f})"


def forcing_curves(system, t_yr):
    """{forcing_source: (values at t_yr, label)} for the two analytic models; the file series is loaded separately."""
    return {
        "analytic": (co2_forcing_model(t_yr, system.forcing_efold_yr),
                     rf"exp: $c + a\,e^{{(t-2014)/{system.forcing_efold_yr:.3g}\,\mathrm{{yr}}}}$"),
        "analytic_gauss": (co2_forcing_gauss_model(t_yr, *gauss_params(system)), gauss_label(system)),
    }


def plot_ensemble_super_spaghetti(ds, out_path, n_shown=None, label=""):
    """Every grid point: realizations (annual means) with the forced response on top; north to south, row by row."""
    n_rows = M // N_COLS
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(13, 1.25 * n_rows), sharex=True)
    data = ds.data if n_shown is None else ds.data[:n_shown]
    for k, i in enumerate(range(M - 1, -1, -1)):
        ax = axes[k // N_COLS, k % N_COLS]
        ax.plot(YEARS, annual(data[:, :, i], axis=-1).T, color="0.45", linewidth=0.4, alpha=0.25)
        ax.plot(YEARS, annual(ds.forced[:, i], axis=-1), color="k", linewidth=1.8)
        ax.set_title(lat_label(i), fontsize=8, pad=2)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("year")
    title = f"{len(data)} of {len(ds.data)} realizations (annual means, grey) and forced response (black), SNR {ds.snr:.3g}"
    fig.suptitle(title + (f"\n{label}" if label else ""))
    save(fig, out_path)


def plot_modal(ds, out_path, n_shown=None, label=""):
    plot_modal_overview(ds, out_path, MODAL_OVERVIEW_SHOWN if n_shown is None else n_shown)


def spinup_years(system):
    return record_years(system)[0] + np.arange(-system.spinup, N) / 12


def plot_spinup(ax, system, title):
    """Forcing and forced slow mode over the whole spin-up, as fractions of their rise over the record."""
    y = forcing_series(system)
    z1 = run_modal(system.Lambda_R, np.outer(y, system.W_inv @ system.b))[:, 0]
    years = spinup_years(system)
    for v, name, style in ((z1, r"forced slow mode $z_1^{(f)}$", dict(color="C0")),
                           (y, "forcing $y$", dict(color="k", linestyle="--", zorder=3))):
        record = v[system.spinup:]
        ax.plot(years, (v - record[0]) / np.ptp(record), linewidth=1.2, label=name, **style)
    ax.axvspan(years[system.spinup], years[-1], color="0.9", zorder=0, label="record")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("year")
    ax.set_ylabel("change from record start\n(fraction of record range)")
    ax.legend(fontsize=7, loc="upper left")


def plot_forcing(ds, out_path, n_shown=None, label=""):
    """The analytic forcing against the forcing file, the one in use in bold, and its spin-up: constant in the past."""
    s = ds.system
    t_month, data = load_forcing_file(s.forcing_file, s.forcing_column)
    models = forcing_curves(s, t_month)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    ax = axes[0, 0]
    curves = {"file": (data, f"file {pathlib.Path(s.forcing_file).name} [{s.forcing_column}], monthly"), **models}
    for name, (values, text) in curves.items():
        used = name == s.forcing_source
        ax.plot(t_month, values, color=FORCING_COLORS[name], linewidth=2.6 if used else 1.0,
                label=text + (" (used)" if used else ""))
    record = record_years(s)
    for a in axes[:, 0]:
        a.axvspan(record[0], record[-1], color="0.92", zorder=0)
    ax.set_ylabel("W m$^{-2}$")
    ax.set_title(f"(a) forcing_source = {s.forcing_source}; record shaded", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    in_record = (t_month >= record[0]) & (t_month <= record[-1])
    for name, (values, text) in models.items():
        residual = values - data
        axes[1, 0].plot(t_month, residual, color=FORCING_COLORS[name], linewidth=1,
                        label=f"{name}: RMSE {np.sqrt((residual ** 2).mean()):.3f} "
                              f"(record {np.sqrt((residual[in_record] ** 2).mean()):.3f}) W m$^{{-2}}$")
    axes[1, 0].axhline(0, color="0.5", linewidth=0.6)
    axes[1, 0].legend(fontsize=7, loc="lower left")
    axes[1, 0].set_title("(b) analytic models minus file", fontsize=9)
    axes[1, 0].set_xlabel("year")
    axes[1, 0].set_ylabel("W m$^{-2}$")
    plot_spinup(axes[0, 1], s, rf"(c) this system: $\tau_1$ = {decay_time(s.lam1) / 12:.3g} yr, "
                               f"spin-up {s.spinup / 12:.0f} yr")
    long = replace(s, lam1=eigenvalue(12 * LONG_TAU1_YR))
    plot_spinup(axes[1, 1], long, rf"(d) longest spin-up: $\tau_1$ = {LONG_TAU1_YR} yr, spin-up {long.spinup / 12:.0f} yr"
                                  " (the zero start decays first)")
    fig.suptitle(f"Forcing ({s.forcing_source}): analytic fit against the forcing file, and the constant past"
                 + (f"\n{label}" if label else ""))
    fig.tight_layout()
    save(fig, out_path)


DIAGNOSTICS = {
    "modal_overview": plot_modal,
    "ensemble_super_spaghetti": plot_ensemble_super_spaghetti,
    "forcing": plot_forcing,
}


def reference_dataset(cfg):
    return make_dataset(build_reference(cfg).system, partial(config_budget, cfg=cfg),
                        n_realizations=cfg.n_realizations, seed=cfg.noise_seed)


def plot_diagnostics(ds, out_dir, plots=tuple(DIAGNOSTICS), n_shown=None, label=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in plots:
        DIAGNOSTICS[name](ds, out_dir / f"{name}.png", n_shown=n_shown, label=label)


def load_config(args):
    cfg = from_json(RESULTS_DIR / args.from_run / "config.json") if args.from_run else DEFAULT
    return with_overrides(cfg, args.set)


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
    plot_diagnostics(ds, FIGURES_DIR / name, args.plots, args.n_shown, label)


if __name__ == "__main__":
    main()

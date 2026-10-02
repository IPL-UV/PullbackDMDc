"""Compare the forcing inputs: the true forcing (file), the exp model and the exp + Gaussians model.

    python compare_forcings.py                          # default system
    python compare_forcings.py --set tau1_yr=50         # the forced responses of a tweaked system
    python compare_forcings.py --from-run baseline --member 3
    python compare_forcings.py --set gauss_dip_amp=0                     # exp + bump only
    python compare_forcings.py --set gauss_bump_year=1940 gauss_bump_width_yr=8 gauss_efold_yr=50

Shape of the exp + Gaussians curve (defaults: the joint fit to AR6 CO2; amplitudes in W m^-2 next to the exp's
a = 1.964, amplitude 0 removes a Gaussian): gauss_efold_yr, gauss_bump_amp, gauss_bump_year, gauss_bump_width_yr,
gauss_dip_amp, gauss_dip_year, gauss_dip_width_yr. The exp curve's shape is forcing_efold_yr.

Writes to figures/diagnostics/data/:
    compare_forcings.png          the three forcings, their misfits, and their shapes over the record
    compare_forced_responses.png  one system under each forcing: forced responses and one ensemble member
With overrides the config slug is appended to both names, so the default figures are never overwritten.
"""

import argparse
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

import plot_style  # noqa: F401  (sets the shared rcParams)
from ablations import (
    M,
    build_reference,
    co2_forcing_gauss_model,
    file_forcing,
    gauss_params,
    gaussian,
    load_forcing_file,
    record_years,
)
from config import config_diff, slug
from plot_ablations import FIGURES_DIR, annual, lat_label
from plot_system import FORCING_COLORS, N_COLS, forcing_curves, load_config, reference_dataset
from test import YEARS, save

SOURCES = ("file", "analytic", "analytic_gauss")
NAMES = {"file": "true (file)", "analytic": "exp", "analytic_gauss": "exp + Gaussians"}


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def standardized(v):
    return (v - v.mean()) / v.std()


def plot_forcing_comparison(cfg, out_path):
    """Returns {model: (RMSE all years, RMSE record, RMSE of the standardized record shape)} against the file."""
    system = build_reference(cfg).system
    t, true = load_forcing_file(cfg.forcing_file, cfg.forcing_column)
    models = forcing_curves(system, t)
    record = record_years(system)
    in_record = (t >= record[0]) & (t <= record[-1])
    true_record = file_forcing(record, cfg.forcing_file, cfg.forcing_column)
    record_models = forcing_curves(system, record)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax in axes.ravel()[:3]:
        ax.axvspan(record[0], record[-1], color="0.93", zorder=0)

    ax = axes[0, 0]
    ax.plot(t, true, color=FORCING_COLORS["file"], linewidth=3, label=f"{NAMES['file']}: {cfg.forcing_column}")
    for name, (values, text) in models.items():
        ax.plot(t, values, color=FORCING_COLORS[name], linewidth=1.2, label=text)
    ax.set_title("(a) forcing (record shaded)", fontsize=9)
    ax.set_ylabel("W m$^{-2}$")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[0, 1]
    efold, bump_params, dip_params = gauss_params(system)
    exp_part = co2_forcing_gauss_model(t, efold, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    bump, dip = gaussian(t, *bump_params), gaussian(t, *dip_params)
    ax.plot(t, true - exp_part, color=FORCING_COLORS["file"], linewidth=2.5,
            label=f"true minus the model's exp part (e-fold {efold:.3g} yr)")
    ax.plot(t, bump - dip, color=FORCING_COLORS["analytic_gauss"], linewidth=1.4, label="Gaussian correction")
    ax.plot(t, bump, color=FORCING_COLORS["analytic_gauss"], linewidth=0.7, linestyle="--",
            label=f"+{bump_params[0]:.3f} at {bump_params[1]:.0f} ($\\sigma$ {bump_params[2]:.1f} yr)")
    ax.plot(t, -dip, color=FORCING_COLORS["analytic_gauss"], linewidth=0.7, linestyle=":",
            label=f"$-${dip_params[0]:.3f} at {dip_params[1]:.0f} ($\\sigma$ {dip_params[2]:.1f} yr)")
    ax.axhline(0, color="0.5", linewidth=0.6)
    ax.set_title("(b) what the Gaussians capture", fontsize=9)
    ax.set_ylabel("W m$^{-2}$")
    ax.legend(fontsize=7, loc="lower left")

    table = {}
    ax = axes[1, 0]
    for name, (values, _) in models.items():
        shape = rmse(standardized(record_models[name][0]), standardized(true_record))
        table[name] = (rmse(values, true), rmse(values[in_record], true[in_record]), shape)
        ax.plot(t, values - true, color=FORCING_COLORS[name], linewidth=1,
                label=f"{NAMES[name]}: RMSE {table[name][0]:.4f} (record {table[name][1]:.4f}) W m$^{{-2}}$")
    ax.axhline(0, color="0.5", linewidth=0.6)
    ax.set_title("(c) model minus true", fontsize=9)
    ax.set_xlabel("year")
    ax.set_ylabel("W m$^{-2}$")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1, 1]
    ax.plot(record, standardized(true_record), color=FORCING_COLORS["file"], linewidth=3, label=NAMES["file"])
    for name, (values, _) in record_models.items():
        ax.plot(record, standardized(values), color=FORCING_COLORS[name], linewidth=1.2,
                label=f"{NAMES[name]}: shape RMSE {table[name][2]:.3f}")
    ax.set_title("(d) what the system sees: record forcing, centered and scaled to unit std", fontsize=9)
    ax.set_xlabel("year")
    ax.set_ylabel("standardized forcing")
    ax.legend(fontsize=7, loc="upper left")

    label = config_diff(cfg)
    fig.suptitle("Forcing inputs: true, exp and exp + Gaussians" + (f"\n{label}" if label else ""))
    fig.tight_layout()
    save(fig, out_path)
    return table


def forced_datasets(cfg, member=0):
    """{forcing_source: dataset} of the same system under each forcing, with realizations 0..member."""
    return {source: reference_dataset(replace(cfg, forcing_source=source, n_realizations=member + 1))
            for source in SOURCES}


def plot_forced_response_comparison(cfg, out_path, member=0):
    datasets = forced_datasets(cfg, member)
    n_rows = M // N_COLS
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(13, 1.25 * n_rows), sharex=True)
    widths = {"file": 2.6, "analytic": 1.4, "analytic_gauss": 1.4}
    for k, i in enumerate(range(M - 1, -1, -1)):
        ax = axes[k // N_COLS, k % N_COLS]
        for source, ds in datasets.items():
            ax.plot(YEARS, annual(ds.data[member, :, i], axis=-1), color=FORCING_COLORS[source], linewidth=0.6,
                    alpha=0.45)
        for source, ds in datasets.items():
            ax.plot(YEARS, annual(ds.forced[:, i], axis=-1), color=FORCING_COLORS[source], linewidth=widths[source],
                    label=NAMES[source])
        ax.set_title(lat_label(i), fontsize=8, pad=2)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("year")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(SOURCES), fontsize=8, frameon=False)
    snr = datasets["analytic"].snr
    label = config_diff(cfg)
    fig.suptitle(f"Forced response under each forcing (thick) and ensemble member {member} (thin, annual means), "
                 f"SNR {snr:.3g}" + (f"\n{label}" if label else ""), x=0.02, ha="left")
    fig.tight_layout()
    save(fig, out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    parser.add_argument("--from-run", help="start from results/<run>/config.json")
    parser.add_argument("--member", type=int, default=0, help="ensemble member shown in the forced-response plot")
    args = parser.parse_args()

    cfg = load_config(args)
    suffix = "" if not config_diff(cfg) else f"__{slug(cfg)}"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table = plot_forcing_comparison(cfg, FIGURES_DIR / f"compare_forcings{suffix}.png")
    plot_forced_response_comparison(cfg, FIGURES_DIR / f"compare_forced_responses{suffix}.png", args.member)
    print(f"{'model':<16s} {'RMSE all':>9s} {'RMSE record':>12s} {'shape RMSE':>11s}   (vs {cfg.forcing_file} "
          f"[{cfg.forcing_column}])")
    for name, (whole, rec, shape) in table.items():
        print(f"{NAMES[name]:<16s} {whole:9.4f} {rec:12.4f} {shape:11.4f}")


if __name__ == "__main__":
    main()

"""Compare the forcing inputs: the true forcing (file), the exp model and the exp + Gaussians model.

    python compare_forcings.py                                     # default config
    python compare_forcings.py --from-run baseline
    python compare_forcings.py --set gauss_dip_amp=1                # a much deeper dip
    python compare_forcings.py --set gauss_bump_amp=0.05 gauss_bump_year=1920 gauss_bump_width_yr=20

Shape of the exp + Gaussians curve (amplitudes in W m^-2 next to the exp's a = 1.96, amplitude 0 removes a
Gaussian): gauss_efold_yr, gauss_bump_amp, gauss_bump_year, gauss_bump_width_yr, gauss_dip_amp, gauss_dip_year,
gauss_dip_width_yr. The exp curve's shape is forcing_efold_yr. Every curve is centered on the record, as the forcing
that generates the data.

Writes figures/diagnostics/data/compare_forcings.png: the three forcings, their misfits, and their shapes over the
record. With overrides the config slug is appended to the name, so the default figure is never overwritten. The
forced responses are in plot_system.py (forced_response_tau1, forcing_response_check).
"""

import argparse
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

import plot_style  # noqa: F401  (sets the shared rcParams)
from ablations import build_reference, gauss_params, gaussian, record_years
from config import config_diff, slug
from plot_ablations import FIGURES_DIR
from plot_system import FORCING_COLORS, centered_curve, file_curve, forcing_curves, load_config
from test import save

SOURCES = ("file", "analytic", "analytic_gauss")
NAMES = {"file": "true (file)", "analytic": "exp", "analytic_gauss": "exp + Gaussians"}


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def standardized(v):
    return (v - v.mean()) / v.std()


def plot_forcing_comparison(cfg, out_path):
    """Returns {model: (RMSE all years, RMSE record, RMSE of the standardized record shape)} against the file."""
    system = build_reference(cfg).system
    t, true = file_curve(system)
    models = forcing_curves(system, t)
    record = record_years(system)
    in_record = (t >= record[0]) & (t <= record[-1])
    true_record = centered_curve(system, "file", record)
    record_models = forcing_curves(system, record)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax in axes.ravel()[:3]:
        ax.axvspan(record[0], record[-1], color="0.93", zorder=0)

    ax = axes[0, 0]
    ax.plot(t, true, color=FORCING_COLORS["file"], linewidth=3, label=f"{NAMES['file']}: {cfg.forcing_column}")
    for name, (values, text) in models.items():
        ax.plot(t, values, color=FORCING_COLORS[name], linewidth=1.2, label=text)
    ax.axhline(0, color="0.5", linewidth=0.6)
    ax.set_title("(a) forcing, centered on the record (shaded)", fontsize=9)
    ax.set_ylabel("W m$^{-2}$, centered on the record")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[0, 1]
    efold, bump_params, dip_params = gauss_params(system)
    exp_part = centered_curve(replace(system, gauss_bump_amp=0.0, gauss_dip_amp=0.0), "analytic_gauss", t)
    bump, dip = (gaussian(t, *g) - gaussian(record, *g).mean() for g in (bump_params, dip_params))
    ax.plot(t, true - exp_part, color=FORCING_COLORS["file"], linewidth=2.5,
            label=f"true minus the model's exp part (e-fold {efold:.3g} yr)")
    ax.plot(t, bump - dip, color=FORCING_COLORS["analytic_gauss"], linewidth=1.4, label="Gaussian correction")
    for sign, part, (amp, year, width), style in (("+", bump, bump_params, "--"), ("$-$", -dip, dip_params, ":")):
        if amp:  # amplitude 0 removes that Gaussian
            ax.plot(t, part, color=FORCING_COLORS["analytic_gauss"], linewidth=0.7, linestyle=style,
                    label=f"{sign}{amp:.2g} at {year:.0f} ($\\sigma$ {width:.0f} yr)")
    ax.axhline(0, color="0.5", linewidth=0.6)
    ax.set_title("(b) what the Gaussians capture (each part centered on the record)", fontsize=9)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    parser.add_argument("--from-run", help="start from results/<run>/config.json")
    args = parser.parse_args()

    cfg = load_config(args)
    suffix = "" if not config_diff(cfg) else f"__{slug(cfg)}"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table = plot_forcing_comparison(cfg, FIGURES_DIR / f"compare_forcings{suffix}.png")
    print(f"{'model':<16s} {'RMSE all':>9s} {'RMSE record':>12s} {'shape RMSE':>11s}   (vs {cfg.forcing_file} "
          f"[{cfg.forcing_column}])")
    for name, (whole, rec, shape) in table.items():
        print(f"{NAMES[name]:<16s} {whole:9.4f} {rec:12.4f} {shape:11.4f}")


if __name__ == "__main__":
    main()

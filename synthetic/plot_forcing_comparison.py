"""Compare the forcing inputs: the true forcing (file), the exp model and the exp + Gaussians model.

    python plot_forcing_comparison.py                                # default config
    python plot_forcing_comparison.py --from-run baseline
    python plot_forcing_comparison.py --set gauss_dip_amp=1          # a much deeper dip
    python plot_forcing_comparison.py --set gauss_bump_amp=0.05 gauss_bump_year=1920

Writes figures/diagnostics/data/compare_forcings.png; with overrides the config slug is appended, so the default
figure is never overwritten. The curves' shape parameters are documented in config.py, and the forced responses
are in plot_system_diagnostics.py (forced_response_tau1, forced_response_shape).
"""

import argparse
from dataclasses import replace

import matplotlib.pyplot as plt

from ablation_data import build_reference, gauss_params, gaussian, record_years
from config import config_diff, load_config, slug
from plot_ablation_diagnostics import FIGURES_DIR, standardized
from plot_style import (FORCING_COLORS, PAGE_W, forcing_panel, record_span, residual_panel, rmse,
                        save, zero_line)
from plot_system_diagnostics import centered_curve, file_curve, forcing_curves, record_mask

NAMES = {"file": "true (file)", "analytic": "exp", "analytic_gauss": "exp + Gaussians"}


def plot_forcing_comparison(cfg, out_path):
    """Returns {model: (RMSE all years, RMSE record, RMSE of the standardized record shape)} against the file."""
    system = build_reference(cfg).system
    t_month, true = file_curve(system)
    models = forcing_curves(system, t_month)
    record = record_years(system)
    in_record = record_mask(t_month, record)
    true_record = centered_curve(system, "file", record)
    record_models = forcing_curves(system, record)
    table = {name: (rmse(values, true), rmse(values[in_record], true[in_record]),
                    rmse(standardized(record_models[name][0]), standardized(true_record)))
             for name, (values, _) in models.items()}

    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, 7.5))
    for ax in axes.ravel()[:3]:
        record_span(ax, record)

    curves = {"file": (true, f"{NAMES['file']}: {cfg.forcing_column}"), **models}
    forcing_panel(axes[0, 0], t_month, curves, bold="file", bold_width=3)
    axes[0, 0].set_title("(a) forcing, centered on the record (shaded)")

    ax = axes[0, 1]
    efold, bump_params, dip_params = gauss_params(system)
    exp_part = centered_curve(replace(system, gauss_bump_amp=0.0, gauss_dip_amp=0.0), "analytic_gauss", t_month)
    bump, dip = (gaussian(t_month, *params) - gaussian(record, *params).mean()
                 for params in (bump_params, dip_params))
    ax.plot(t_month, true - exp_part, color=FORCING_COLORS["file"], linewidth=2.5,
            label=f"true minus the model's exp part (e-fold {efold:.3g} yr)")
    ax.plot(t_month, bump - dip, color=FORCING_COLORS["analytic_gauss"], linewidth=1.4, label="Gaussian correction")
    for sign, part, (amp, year, width), style in (("+", bump, bump_params, "--"), ("$-$", -dip, dip_params, ":")):
        if amp:  # amplitude 0 removes that Gaussian
            ax.plot(t_month, part, color=FORCING_COLORS["analytic_gauss"], linewidth=0.7, linestyle=style,
                    label=f"{sign}{amp:.2g} at {year:.0f} ($\\sigma$ {width:.0f} yr)")
    zero_line(ax)
    ax.set_title("(b) what the Gaussians capture (each part centered on the record)")
    ax.set_ylabel("W m$^{-2}$")
    ax.legend(loc="lower left")

    residual_panel(axes[1, 0], t_month, true, models, in_record, names=NAMES, precision=4)
    axes[1, 0].set_title("(c) model minus true")
    axes[1, 0].set_xlabel("year")

    ax = axes[1, 1]
    ax.plot(record, standardized(true_record), color=FORCING_COLORS["file"], linewidth=3, label=NAMES["file"])
    for name, (values, _) in record_models.items():
        ax.plot(record, standardized(values), color=FORCING_COLORS[name], linewidth=1.2,
                label=f"{NAMES[name]}: shape RMSE {table[name][2]:.3f}")
    ax.set_title("(d) what the system sees: record forcing, centered and scaled to unit std")
    ax.set_xlabel("year")
    ax.set_ylabel("standardized forcing")
    ax.legend(loc="upper left")

    label = config_diff(cfg)
    fig.suptitle("Forcing inputs: true, exp and exp + Gaussians" + (f"\n{label}" if label else ""))
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

"""System diagnostics for any config or sweep level, written to figures/system/<name>/.

    python plot_system.py                                         # default reference system
    python plot_system.py --set tau1_yr=50 slow_variance=0.5      # tweaked reference system
    python plot_system.py --study slow_timescale_snr --level 100  # one sweep dataset
    python plot_system.py --from-run baseline --plots modal_overview
"""

import argparse
import ast
import pathlib
from functools import partial

import matplotlib.pyplot as plt

from ablations import M, build_reference, config_budget, find_level, make_dataset
from config import DEFAULT, config_diff, from_json, slug, with_overrides
from plot_ablations import annual, lat_label
from test import YEARS, plot_modal_overview, save

SYNTHETIC_DIR = pathlib.Path(__file__).resolve().parent
FIGURES_DIR = SYNTHETIC_DIR / "figures" / "system"
RESULTS_DIR = SYNTHETIC_DIR / "results"
MODAL_OVERVIEW_SHOWN = 10


def plot_ensemble_super_spaghetti(ds, out_path, n_shown=None, label=""):
    """Every grid point: realizations (annual means) with the forced response on top; north to south."""
    n_rows = M // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 1.25 * n_rows), sharex=True)
    data = ds.data if n_shown is None else ds.data[:n_shown]
    for k, i in enumerate(range(M - 1, -1, -1)):
        ax = axes[k % n_rows, k // n_rows]
        ax.plot(YEARS, annual(data[:, :, i], axis=-1).T, color="0.45", linewidth=0.4, alpha=0.25)
        ax.plot(YEARS, annual(ds.forced[:, i], axis=-1), color="k", linewidth=1.8)
        ax.set_ylabel(lat_label(i), rotation=0, ha="right", va="center")
    for ax in axes[-1]:
        ax.set_xlabel("year")
    title = f"{len(data)} of {len(ds.data)} realizations (annual means, grey) and forced response (black), SNR {ds.snr:.3g}"
    fig.suptitle(title + (f"\n{label}" if label else ""))
    save(fig, out_path)


def plot_modal(ds, out_path, n_shown=None, label=""):
    plot_modal_overview(ds, out_path, MODAL_OVERVIEW_SHOWN if n_shown is None else n_shown)


DIAGNOSTICS = {
    "modal_overview": plot_modal,
    "ensemble_super_spaghetti": plot_ensemble_super_spaghetti,
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
    parser.add_argument("--name", help="output folder under figures/system/")
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

"""Fit every method on every ablation dataset and score the forced-response and operator recovery.

    python run_ablations.py                                       # results/baseline/
    python run_ablations.py --name slow50 --set tau1_yr=50 slow_variance=0.5
    python run_ablations.py --name lag3 --from-run baseline --set lag=3 --studies total_snr

Each run writes results/<name>/config.json and results/<name>/ablations.csv (one row per study, level,
realization and method; the oracle forced response, true A and b over the same history, is one row per level
with method "oracle"), plus the system diagnostics of its reference system in figures/ablations/<name>/.
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

from ablations import decay_time, study_levels
from config import DEFAULT, config_diff, to_json
from methods import make_methods
from plot_system import RESULTS_DIR, load_config, plot_diagnostics, reference_dataset
from test import oracle_forced, plane_cosines, rms, slow_index

FIGURES_DIR = pathlib.Path(__file__).resolve().parent / "figures" / "ablations"


def slow_eig_err(eigvals, lam1):
    real = np.abs(eigvals.imag) < 1e-12
    return np.abs(eigvals[slow_index(eigvals, lam1)] - lam1) if real.any() else np.nan


def pair_angle(eigvals, eigvecs, pair_eig, W_pair):
    """Largest principal angle (degrees) between span(w2, w3) and the plane of the eigenvector nearest the pair."""
    complex_ = np.where(eigvals.imag > 1e-12)[0]
    if len(complex_) == 0:
        return np.nan
    v = eigvecs[:, complex_[np.argmin(np.abs(eigvals[complex_] - pair_eig))]]
    cosines = plane_cosines(W_pair, np.column_stack([v.real, v.imag]))
    return np.rad2deg(np.arccos(np.clip(cosines.min(), -1, 1)))


def score(ds, forced_est, A=None, lag=1):
    """A is the fitted propagator over `lag` steps, so its eigenvalues are compared with the true ones ** lag."""
    row = dict(
        forced_corr=np.corrcoef(forced_est.ravel(), ds.forced.ravel())[0, 1],
        forced_rel_rmse=rms(forced_est - ds.forced) / rms(ds.forced),
        slow_eig_err=np.nan, slow_tau_rel_err=np.nan, pair_angle=np.nan,
    )
    if A is not None:
        eigvals, eigvecs = np.linalg.eig(A)
        lam1 = ds.system.lam1 ** lag
        row["slow_eig_err"] = slow_eig_err(eigvals, lam1)
        if not np.isnan(row["slow_eig_err"]):
            tau = lag * decay_time(eigvals[slow_index(eigvals, lam1)].real)
            tau1 = decay_time(ds.system.lam1)
            row["slow_tau_rel_err"] = np.abs(tau - tau1) / tau1
        row["pair_angle"] = pair_angle(eigvals, eigvecs, ds.system.eigvals[1] ** lag, ds.system.W[:, 1:3])
    return row


def run_level(study, make, cfg=DEFAULT):
    (ds,) = make()
    meta = dict(study=study, param_name=ds.param_name, param_value=ds.param_value,
                snr=ds.snr, tau1_yr=decay_time(ds.system.lam1) / 12)
    rows = [dict(meta, realization=-1, method="oracle", **score(ds, oracle_forced(ds)))]
    long_forcings, short_forcings, history = ds.forcings()
    methods = make_methods(cfg)
    for r, data in enumerate(ds.data):
        for name, fit in methods.items():
            rows.append(dict(meta, realization=r, method=name,
                             **score(ds, *fit(data, long_forcings, short_forcings, history), lag=cfg.lag)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="baseline", help="run name: results/<name>/")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides (see config.py)")
    parser.add_argument("--from-run", help="start from results/<run>/config.json, then apply --set")
    parser.add_argument("--studies", nargs="*", help="study names to run (default: all)")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    cfg = load_config(args)
    levels = [(s, make) for s, _, make in study_levels(cfg) if not args.studies or s in args.studies]
    if not levels:
        parser.error(f"no levels selected; studies are {sorted({s for s, _, _ in study_levels(cfg)})}")
    print(f"run {args.name!r}: {len(levels)} levels, {cfg.n_realizations} realizations, "
          f"config: {config_diff(cfg) or 'default'}")
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(run_level)(s, make, cfg) for s, make in levels)

    out_dir = RESULTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    to_json(cfg, out_dir / "config.json")
    pd.DataFrame([row for rows in results for row in rows]).to_csv(out_dir / "ablations.csv", index=False)
    print(f"saved {out_dir}/config.json, ablations.csv")
    plot_diagnostics(reference_dataset(cfg), FIGURES_DIR / args.name, label=config_diff(cfg), cfg=cfg)


if __name__ == "__main__":
    main()

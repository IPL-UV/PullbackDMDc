"""Fit every method on every ablation dataset and score the forced-response and operator recovery.

    python run_ablation_studies.py                                       # results/baseline/
    python run_ablation_studies.py --name slow50 --set tau1_yr=50 slow_variance=0.5
    python run_ablation_studies.py --name lag3 --from-run baseline --set lag=3 --studies total_snr

Each run writes results/<name>/config.json and results/<name>/ablations.csv (one row per study, level,
realization and method; the oracle forced response, true A and b over the same history, is one row per level
with method "oracle"), plus the system diagnostics of its reference system in figures/ablations/<name>/.
"""

import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

from ablation_data import M, N, decay_time, decay_time_yr, drive_modal, study_levels
from config import DEFAULT, RESULTS_DIR, config_diff, load_config, to_json
from methods import make_methods
from plot_ablation_run_results import FIGURES_DIR
from plot_system_diagnostics import plot_diagnostics, reference_dataset


def rms(a):
    return np.sqrt((a**2).mean())


def plane_cosines(U, V):
    return np.linalg.svd(np.linalg.qr(U)[0].T @ np.linalg.qr(V)[0], compute_uv=False)


def subspace_angle(U, V):
    """Largest principal angle (degrees) between the column spans of U and V.

    Only the spans enter, so the angle is invariant to the sign, phase and scale an eigensolver happens
    to give its vectors. One-dimensional spans have a single principal angle, arccos |<u, v>| / (|u| |v|),
    which is what the slow mode is scored with; the oscillating pair spans a plane and uses the largest of
    the two angles.
    """
    return np.rad2deg(np.arccos(np.clip(plane_cosines(U, V).min(), -1, 1)))


def real_eigs(eigvals):
    return np.where(np.abs(eigvals.imag) < 1e-12)[0]


def slow_index(eigvals, lam1):
    real = real_eigs(eigvals)
    return real[np.argmin(np.abs(eigvals[real] - lam1))]


def oracle_forced(ds):
    """The forced response the true A and B give over the same history: the floor set by truncating it."""
    s = ds.system
    long_forcings, _, history = ds.forcings()
    return drive_modal(s, long_forcings[:, 0])[history:] @ s.W.T


def pair_angle(eigvals, eigvecs, pair_eig, W_pair):
    """Largest principal angle (degrees) between span(w2, w3) and the plane of the eigenvector nearest the pair."""
    oscillating = np.where(eigvals.imag > 1e-12)[0]
    if len(oscillating) == 0:
        return np.nan
    v = eigvecs[:, oscillating[np.argmin(np.abs(eigvals[oscillating] - pair_eig))]]
    return subspace_angle(W_pair, np.column_stack([v.real, v.imag]))


def slow_angle(eigvecs, index, w1):
    """Principal angle (degrees) between the fitted slow eigenvector and the true slow mode w1.

    The eigenvector of a real eigenvalue of a real propagator is real, so its imaginary part is roundoff.
    """
    return subspace_angle(w1[:, None], eigvecs[:, index].real[:, None])


def score(ds, forced_est, A=None, lag=1):
    """A is the fitted propagator over `lag` steps, so its eigenvalues are compared with the true ones ** lag."""
    row = dict(
        forced_corr=np.corrcoef(forced_est.ravel(), ds.forced.ravel())[0, 1],
        forced_rel_rmse=rms(forced_est - ds.forced) / rms(ds.forced),
        slow_eig_err=np.nan, slow_tau_rel_err=np.nan, slow_tau_log_ratio=np.nan,
        slow_unstable=np.nan, slow_angle=np.nan, pair_angle=np.nan,
    )
    if A is None:
        return row
    eigvals, eigvecs = np.linalg.eig(A)
    lam1 = ds.system.lam1 ** lag
    row["pair_angle"] = pair_angle(eigvals, eigvecs, ds.system.eigvals[1] ** lag, ds.system.W[:, 1:3])
    if not len(real_eigs(eigvals)):
        return row
    index = slow_index(eigvals, lam1)
    lam_hat = eigvals[index].real
    row["slow_eig_err"] = np.abs(eigvals[index] - lam1)
    row["slow_angle"] = slow_angle(eigvecs, index, ds.system.W[:, 0])
    # A fit with |lambda| >= 1 does not decay, so it has no decay time: count it instead of letting a
    # negative tau pass as just another large error. lambda = 0 (tau = 0) is the same degenerate case.
    row["slow_unstable"] = float(np.abs(lam_hat) >= 1)
    if 0 < np.abs(lam_hat) < 1:
        tau, tau1 = lag * decay_time(lam_hat), decay_time(ds.system.lam1)
        row["slow_tau_rel_err"] = np.abs(tau - tau1) / tau1
        # the relative error saturates at 1 for underestimates but is unbounded for overestimates; the
        # log ratio puts the two on the same scale
        row["slow_tau_log_ratio"] = np.abs(np.log(tau / tau1))
    return row


def run_level(study, make, cfg=DEFAULT):
    (ds,) = make()
    meta = dict(study=study, param_name=ds.param_name, param_value=ds.param_value,
                snr=ds.snr, tau1_yr=decay_time_yr(ds.system.lam1))
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
    # N and M are structural, so they are not in the config: record them here, or runs made before a
    # structural change silently compare against runs made after it.
    (out_dir / "provenance.json").write_text(json.dumps(
        {"record_months": N, "state_dim": M, "b_scaling": "||y0 (I - A)^-1 B||_F = 1"}, indent=2) + "\n")
    pd.DataFrame([row for rows in results for row in rows]).to_csv(out_dir / "ablations.csv", index=False)
    print(f"saved {out_dir}/config.json, provenance.json, ablations.csv")
    plot_diagnostics(reference_dataset(cfg), FIGURES_DIR / args.name, label=config_diff(cfg), cfg=cfg)


if __name__ == "__main__":
    main()

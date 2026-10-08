"""Fit every method on every ablation dataset and score the forced-response and operator recovery.

    python run_ablation_studies.py                                       # results/baseline/
    python run_ablation_studies.py --name slow50 --set tau1_yr=50 slow_variance=0.5
    python run_ablation_studies.py --name lag3 --from-run baseline --set lag=3 --studies total_snr

Each run writes results/<name>/config.json and results/<name>/ablations.csv (one row per study, level,
realization and method), plus the system diagnostics of its reference system in figures/ablations/<name>/.
"""

import argparse
import importlib.metadata
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

from ablation_data import M, N, SPINUP_DECAY_TIMES, decay_time, decay_time_yr, study_levels
from config import DEFAULT, RESULTS_DIR, config_diff, load_config, to_json
from methods import make_methods
from plot_ablation_run_results import FIGURES_DIR
from plot_system_diagnostics import plot_diagnostics, reference_dataset


# The system diagnostics kept alongside each run. `forcing` and `ensemble_super_spaghetti` describe the
# forcing and the noise realizations, which are properties of the config rather than of the fits, and they
# are identical in every run that shares a config -- figures/diagnostics/system/<name>/ is where they belong,
# and plot_system_diagnostics.py's own main still writes all five there.
RUN_DIAGNOSTICS = ("modal_overview", "forced_response_ablations", "forced_response_shape")


def package_versions():
    """Versions of everything a run's numbers depend on, for provenance.json.

    The seeds pin the draws only within one numpy generator stream, and the fits are LAPACK calls, so
    the environment is as much a part of a run's identity as its config. Missing packages are recorded
    as None rather than skipped, so the key set does not change with what happens to be installed.
    """
    versions = {}
    # distribution names, not import names: scikit-learn installs the `sklearn` module
    for name in ("numpy", "scipy", "scikit-learn", "joblib", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def rms(a):
    return np.sqrt((a**2).mean())


def centered_rms(a):
    """RMS of an (N, M) field about its own time mean: sqrt(V / M) with V = sum_i Var_t a_i.

    The denominator of a relative error has to be the variance the SNR is built from
    (ablation_data.forced_variance), or the score is deflated wherever the forced response carries a
    large constant offset: by a factor 2.5 at tau_1 = 100 yr, where it would turn the growing offset
    into an apparent gain in skill. The numerator stays uncentered, so a constant bias in the estimate
    is still penalized.
    """
    return rms(a - a.mean(axis=0))


def plane_cosines(U, V):
    return np.linalg.svd(np.linalg.qr(U)[0].T @ np.linalg.qr(V)[0], compute_uv=False)


def subspace_corr(U, V):
    """Correlation between the column spans of U and V: their smallest principal cosine, in [0, 1].

    Only the spans enter, so the score is invariant to the sign, phase and scale an eigensolver happens
    to give its vectors. One-dimensional spans have a single principal cosine, |<u, v>| / (|u| |v|) --
    the pattern correlation of the two modes up to that invariance -- which is what the slow mode is
    scored with; the oscillating pair spans a plane and is scored on its worst-aligned direction, so 1
    is recovery of the whole plane rather than of one lucky direction in it.
    """
    return float(np.clip(plane_cosines(U, V).min(), 0, 1))


def subspace_angle(U, V):
    """Largest principal angle (degrees) between the column spans of U and V: arccos of subspace_corr."""
    return np.rad2deg(np.arccos(subspace_corr(U, V)))


def real_eigs(eigvals):
    return np.where(np.abs(eigvals.imag) < 1e-12)[0]


def slow_index(eigvals, lam1):
    real = real_eigs(eigvals)
    return real[np.argmin(np.abs(eigvals[real] - lam1))]


def pair_subspace(eigvals, eigvecs, pair_eig, W_pair):
    """(true, fitted) planes of the oscillating pair, or None when the fit has no complex pair.

    The fitted plane is spanned by the real and imaginary parts of the eigenvector nearest the pair.
    Returning the pair of subspaces, rather than one score, keeps the correlation and the angle --
    the same quantity in two units -- from being computed in two places.
    """
    oscillating = np.where(eigvals.imag > 1e-12)[0]
    if len(oscillating) == 0:
        return None
    v = eigvecs[:, oscillating[np.argmin(np.abs(eigvals[oscillating] - pair_eig))]]
    return W_pair, np.column_stack([v.real, v.imag])


def slow_subspace(eigvecs, index, w1):
    """(true, fitted) spans of the slow mode, as single-column matrices.

    The eigenvector of a real eigenvalue of a real propagator is real, so its imaginary part is roundoff.
    """
    return w1[:, None], eigvecs[:, index].real[:, None]


def fitted_slow_mode(A, lam1, w1):
    """The fitted slow eigenvector, unit norm and signed to agree with w1, or None when there is none.

    The sign and scale of an eigenvector are arbitrary, so the patterns could not be averaged across
    realizations without fixing them first; the scalar scores in `score` are invariant to both and do
    not need this. Recomputing the eigendecomposition rather than returning it from `score` keeps that
    function's signature, and an eig of a 20x20 matrix is nothing against the fit that produced A.
    """
    if A is None:
        return None
    eigvals, eigvecs = np.linalg.eig(A)
    if not len(real_eigs(eigvals)):
        return None
    _, fitted = slow_subspace(eigvecs, slow_index(eigvals, lam1), w1)
    v = fitted[:, 0]
    return v / np.linalg.norm(v) * (np.sign(v @ w1) or 1.0)


def save_slow_modes(modes, path):
    """The fitted slow-mode patterns, one row per (study, level, realization, method), plus the truth.

    A file of its own rather than more columns in ablations.csv, because a row here is a vector of M
    numbers, not a scalar. The truth is stored per level, not once per run: it is the one thing that
    moves with the level in spatial_overlap and forcing_overlap. Strings go in as fixed-width unicode
    so the archive reads back without allow_pickle.
    """
    frame = pd.DataFrame(modes)
    arrays = {"pattern": np.stack(frame.pop("pattern").to_numpy())}
    for column in frame:
        values = frame[column].to_numpy()
        # astype(str) sizes the dtype to the longest string present; a fixed width silently truncates
        arrays[column] = values.astype(str) if values.dtype == object else values
    np.savez_compressed(path, **arrays)


def score(ds, forced_est, A=None, lag=1):
    """A is the fitted propagator over `lag` steps, so its eigenvalues are compared with the true ones ** lag."""
    row = dict(
        forced_corr=np.corrcoef(forced_est.ravel(), ds.forced.ravel())[0, 1],
        forced_rel_rmse=rms(forced_est - ds.forced) / centered_rms(ds.forced),
        slow_eig_err=np.nan, slow_tau_yr=np.nan, slow_tau_rel_err=np.nan, slow_tau_log_ratio=np.nan,
        slow_unstable=np.nan, slow_corr=np.nan, slow_angle=np.nan, pair_corr=np.nan, pair_angle=np.nan,
    )
    if A is None:
        return row
    eigvals, eigvecs = np.linalg.eig(A)
    lam1 = ds.system.lam1 ** lag
    pair = pair_subspace(eigvals, eigvecs, ds.system.eigvals[1] ** lag, ds.system.W[:, 1:3])
    if pair is not None:
        row["pair_corr"], row["pair_angle"] = subspace_corr(*pair), subspace_angle(*pair)
    if not len(real_eigs(eigvals)):
        return row
    index = slow_index(eigvals, lam1)
    lam_hat = eigvals[index].real
    row["slow_eig_err"] = np.abs(eigvals[index] - lam1)
    slow = slow_subspace(eigvecs, index, ds.system.W[:, 0])
    row["slow_corr"], row["slow_angle"] = subspace_corr(*slow), subspace_angle(*slow)
    # A fit with |lambda| >= 1 does not decay, so it has no decay time: count it instead of letting a
    # negative tau pass as just another large error. lambda = 0 (tau = 0) is the same degenerate case.
    row["slow_unstable"] = float(np.abs(lam_hat) >= 1)
    if 0 < np.abs(lam_hat) < 1:
        tau, tau1 = lag * decay_time(lam_hat), decay_time(ds.system.lam1)
        # the figure plots the estimate itself against the true tau_1, so it is also reported as it is,
        # in years; lag * decay_time(lambda_hat) is the estimate of the one-step decay time, so the
        # column does not move when the lag does
        row["slow_tau_yr"] = lag * decay_time_yr(lam_hat)
        row["slow_tau_rel_err"] = np.abs(tau - tau1) / tau1
        # the relative error saturates at 1 for underestimates but is unbounded for overestimates; the
        # log ratio puts the two on the same scale
        row["slow_tau_log_ratio"] = np.abs(np.log(tau / tau1))
    return row


def run_level(study, make, cfg=DEFAULT):
    """(scored rows, slow-mode patterns) for one level: the CSV rows and the vectors save_slow_modes writes."""
    (ds,) = make()
    meta = dict(study=study, param_name=ds.param_name, param_value=ds.param_value,
                snr=ds.snr, tau1_yr=decay_time_yr(ds.system.lam1))
    mode_meta = {k: meta[k] for k in ("study", "param_name", "param_value", "tau1_yr")}
    w1 = ds.system.W[:, 0]
    rows = []
    modes = [dict(mode_meta, realization=-1, method="truth", pattern=w1 / np.linalg.norm(w1))]
    long_forcings, short_forcings, spinup = ds.forcings()
    methods = make_methods(cfg)
    for r, data in enumerate(ds.data):
        for name, fit in methods.items():
            forced_est, A = fit(data, long_forcings, short_forcings, spinup)
            rows.append(dict(meta, realization=r, method=name, **score(ds, forced_est, A, lag=cfg.lag)))
            pattern = fitted_slow_mode(A, ds.system.lam1 ** cfg.lag, w1)
            if pattern is not None:
                modes.append(dict(mode_meta, realization=r, method=name, pattern=pattern))
    return rows, modes


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
    # N, M, the B scaling and the scoring conventions are structural, so they are not in the config:
    # record them here, or runs made before a structural change silently compare against runs made
    # after it. forced_rel_rmse's denominator was uncentered before Oct 2026, which deflated it by up
    # to 2.5x at long tau_1 (see centered_rms), so a run without this key is not comparable.
    #
    # The seeds are in config.json too, but they are repeated here because they are what makes the run
    # reproducible and this is the file you read to find out whether two runs are comparable. `versions`
    # is the part config.json cannot carry: the draws come from numpy's generator stream and the fits
    # from LAPACK, so a run is reproducible given the same config AND the same environment.
    (out_dir / "provenance.json").write_text(json.dumps(
        {"record_months": N, "state_dim": M, "b_scaling": "||y0 (I - A)^-1 B||_F = 1",
         "spinup_decay_times": SPINUP_DECAY_TIMES,
         "forced_rel_rmse": "rms(est - true) / centered_rms(true), denominator = sqrt(V_f / M)",
         "seeds": {name: getattr(cfg, name) for name in ("pattern_seed", "eigenvalue_seed", "noise_seed")},
         "versions": package_versions()},
        indent=2) + "\n")
    pd.DataFrame([row for rows, _ in results for row in rows]).to_csv(out_dir / "ablations.csv", index=False)
    save_slow_modes([mode for _, modes in results for mode in modes], out_dir / "slow_modes.npz")
    print(f"saved {out_dir}/config.json, provenance.json, ablations.csv, slow_modes.npz")
    plot_diagnostics(reference_dataset(cfg), FIGURES_DIR / args.name, RUN_DIAGNOSTICS,
                     label=config_diff(cfg), cfg=cfg)


if __name__ == "__main__":
    main()

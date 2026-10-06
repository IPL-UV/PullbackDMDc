"""Helpers shared by the test modules."""

import numpy as np

from ablation_data import M, REFERENCE, build_reference, drive_modal, equal_budget, make_dataset
from config import DEFAULT, with_overrides
from methods import fit_pullback
from run_ablation_studies import oracle_forced, rms, slow_index  # noqa: F401  (re-exported)

SMALL = dict(n_realizations=2)


def reference_data(**kwargs):
    """A dataset on the default reference system at the equal budget."""
    return make_dataset(REFERENCE, equal_budget, **{"n_realizations": 1, **kwargs})


def record_dataset(cfg, **kwargs):
    """(reference, dataset) for a config, for the forcing tests that compare y against a model curve."""
    ref = build_reference(cfg)
    return ref, make_dataset(ref.system, equal_budget, **{"n_realizations": 1, **kwargs})


def overridden(overrides, source="analytic_gauss"):
    return with_overrides(DEFAULT, [f"forcing_source={source}", *overrides])


def assert_unit_b_scale(system):
    """||y0 (I - A)^-1 B||_F = 1."""
    gain = np.linalg.solve(np.eye(M) - system.A, system.B)
    assert abs(abs(system.y0) * np.linalg.norm(gain) - 1) < 1e-9


def noise_free_white_forcing(n_steps=20000):
    """(system, forcing, data) with no internal noise and a white forcing: the exactly recoverable case."""
    s = REFERENCE
    rng = np.random.default_rng(0)
    y = rng.standard_normal(s.spinup + n_steps)
    z = drive_modal(s, y, z0=s.W_inv @ rng.standard_normal(M))
    return s, y, z[s.spinup:] @ s.W.T


def recovery_metrics(ds):
    """PullbackDMDc skill per realization.

    Deliberately independent of run_ablation_studies.score: test_runner_results_match_recovery_diagnostics
    cross-checks the two implementations against each other.
    """
    long_forcings, short_forcings, history = ds.forcings()
    rows = []
    for data, internal in zip(ds.data, ds.internal):
        system = ds.system
        model = fit_pullback(data, long_forcings, short_forcings, history)
        forced_est = model.predict()
        internal_est = data - forced_est
        rows.append(dict(
            forced_corr=np.corrcoef(forced_est.ravel(), ds.forced.ravel())[0, 1],
            forced_err=rms(forced_est - ds.forced) / rms(ds.forced),
            internal_err=rms(internal_est - internal) / rms(internal),
            mirror=np.abs((forced_est - ds.forced) + (internal_est - internal)).max(),
            slow_eig=model.eigvals[slow_index(model.eigvals, system.lam1)].real,
            forced_mse=((forced_est - ds.forced) ** 2).mean(axis=0),
            internal_mse=((internal_est - internal) ** 2).mean(axis=0),
            forced_est=forced_est,
        ))
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}

"""Helpers shared by the test modules."""

import numpy as np

from ablations import M, T
from config import DEFAULT, with_overrides
from methods import compute_eofs, fit_pullback
from run_ablation_studies import centered_rms, rms, true_eigs  # noqa: F401  (re-exported)
from synthetic_generator import generate_ensemble

# a short spin-up keeps a test ensemble cheap: 5 decay times still leave the start-up transient at e^-5
CHEAP = with_overrides(DEFAULT, ["spin_up_factor=5", "n_realizations=2"])


def cheap_ensemble(n_members=1, **kwargs):
    return generate_ensemble(n_members, **{"tau1": DEFAULT.tau1, "spin_up_factor": 5, "rng": 0, **kwargs})


def noise_free_white_forcing(n_steps=20000):
    """(ens, forcing, data, spinup): the reference propagator driven, with no noise, by a white forcing.

    The exactly recoverable case: a white forcing excites every mode, where the smooth forcing of the
    ensemble leaves the propagator poorly determined.
    """
    ens = cheap_ensemble(spin_up_factor=40)  # long enough for the random initial state to decay
    rng = np.random.default_rng(0)
    spinup = len(ens["y_full"]) - T
    y = rng.standard_normal(spinup + n_steps)
    x, data = rng.standard_normal(M), np.empty((len(y), M))
    for t, yt in enumerate(y):
        x = ens["A"] @ x + ens["B"] * yt
        data[t] = x
    return ens, y, data[spinup:], spinup


def recovery_metrics(ens, lag):
    """PullbackDMDc skill per member.

    Deliberately independent of run_ablation_studies.score: test_runner_results_match_recovery_metrics
    cross-checks the two implementations against each other.
    """
    spinup = len(ens["y_full"]) - T
    rows = []
    for data in ens["data"]:
        model = fit_pullback(data, ens["y_full"][:, None], ens["y"][:, None], spinup, compute_eofs(data), lag=lag)
        forced_est = model.predict()
        rows.append(dict(
            forced_corr=np.corrcoef((forced_est - forced_est.mean(0)).ravel(),
                                    (ens["forced"] - ens["forced"].mean(0)).ravel())[0, 1],
            forced_err=rms(forced_est - ens["forced"]) / centered_rms(ens["forced"]),
            # the slow mode is the largest real eigenvalue, selected here without run_ablation_studies.slow_index
            slow_eig=model.eigvals[np.abs(model.eigvals.imag) < 1e-12].real.max(),
        ))
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}

from dataclasses import replace

import numpy as np

from ablation_data import M, REFERENCE, decay_time, equal_budget, find_level, make_dataset
from config import DEFAULT, with_overrides
from run_ablation_studies import run_level, score
from support import recovery_metrics


OPERATOR_KEYS = ("slow_eig_err", "slow_tau_rel_err", "slow_tau_log_ratio", "slow_unstable",
                 "slow_angle", "pair_angle")


def test_score_matches_known_solution():
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    row = score(ds, ds.forced, ds.system.A)
    assert abs(row["forced_corr"] - 1) < 1e-12 and row["forced_rel_rmse"] < 1e-12
    assert row["slow_eig_err"] < 1e-10 and row["slow_tau_rel_err"] < 1e-8 and row["slow_tau_log_ratio"] < 1e-8
    assert row["slow_unstable"] == 0
    assert row["slow_angle"] < 1e-6 and row["pair_angle"] < 1e-5, row
    no_operator = score(ds, ds.forced)
    assert all(np.isnan(no_operator[k]) for k in OPERATOR_KEYS)
    row = score(ds, ds.forced, np.linalg.matrix_power(ds.system.A, 3), lag=3)
    assert row["slow_eig_err"] < 1e-10 and row["slow_tau_rel_err"] < 1e-8
    assert row["slow_angle"] < 1e-6 and row["pair_angle"] < 1e-5, row


def propagator(system, W=None, lam1=None):
    """The true system's propagator with its slow eigenvector and/or slow eigenvalue replaced.

    Built from the real block-diagonal Lambda_R, so the oscillating pair stays a conjugate pair and A stays real.
    """
    W = system.W if W is None else W
    Lambda = system.Lambda_R.copy()
    if lam1 is not None:
        Lambda[0, 0] = lam1
    return W @ Lambda @ np.linalg.inv(W)


def tilted_slow_mode(system, tilt=0.3):
    """W with w1 tilted into w4, so the recovered slow mode sits at a comfortably non-zero angle from w1."""
    W = system.W.copy()
    W[:, 0] = W[:, 0] + tilt * W[:, 3]
    return W


def test_slow_angle_is_invariant_to_sign_and_scale():
    """The principal angle sees only the span, so rescaling or flipping the true mode cannot move it."""
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    A = propagator(ds.system, W=tilted_slow_mode(ds.system))
    angle = score(ds, ds.forced, A)["slow_angle"]
    assert 1 < angle < 89, angle
    for factor in (-3.0, 0.1):
        rescaled = replace(ds.system, W=ds.system.W * np.array([factor] + [1.0] * (M - 1)))
        assert abs(score(replace(ds, system=rescaled), ds.forced, A)["slow_angle"] - angle) < 1e-9


def test_non_decaying_fit_is_counted_not_scored():
    """A propagator with |lambda_1| > 1 has no decay time: the tau metrics are nan and the fit is flagged."""
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    row = score(ds, ds.forced, propagator(ds.system, lam1=1.01))
    assert row["slow_unstable"] == 1
    assert np.isnan(row["slow_tau_rel_err"]) and np.isnan(row["slow_tau_log_ratio"])
    assert row["slow_angle"] < 1e-6, row["slow_angle"]  # the shape is still recovered


def test_tau_log_ratio_is_symmetric_in_over_and_underestimates():
    """Halving and doubling tau_1 are the same error on the log ratio, unlike the relative error."""
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    tau1 = decay_time(ds.system.lam1)
    rows = [score(ds, ds.forced, propagator(ds.system, lam1=np.exp(-1 / (factor * tau1))))
            for factor in (0.5, 2.0)]
    assert abs(rows[0]["slow_tau_log_ratio"] - rows[1]["slow_tau_log_ratio"]) < 1e-9
    assert np.allclose([r["slow_tau_rel_err"] for r in rows], [0.5, 1.0], atol=1e-9)


def test_runner_results_match_recovery_diagnostics():
    cfg = with_overrides(DEFAULT, ["n_realizations=5"])
    rows = run_level("total_snr", find_level(cfg, "total_snr", 1 / 3), cfg)
    ours = [r for r in rows if r["method"] == "PullbackDMDc"]
    ref = recovery_metrics(make_dataset(REFERENCE, equal_budget, n_realizations=5))
    assert np.allclose([r["forced_corr"] for r in ours], ref["forced_corr"], atol=1e-12)
    assert np.allclose([r["forced_rel_rmse"] for r in ours], ref["forced_err"], atol=1e-12)
    slow_eig = np.array([r["slow_eig_err"] for r in ours])
    assert np.allclose(slow_eig, np.abs(ref["slow_eig"] - REFERENCE.lam1), atol=1e-12)
    oracle = [r for r in rows if r["method"] == "oracle"]
    assert len(oracle) == 1 and oracle[0]["forced_rel_rmse"] < 0.05

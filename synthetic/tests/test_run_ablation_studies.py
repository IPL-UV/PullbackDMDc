import pathlib
import tempfile
from dataclasses import replace

import numpy as np

from ablation_data import M, REFERENCE, decay_time, equal_budget, find_level, make_dataset
from config import DEFAULT, with_overrides
from run_ablation_studies import centered_rms, fitted_slow_mode, run_level, save_slow_modes, score
from support import recovery_metrics


OPERATOR_KEYS = ("slow_eig_err", "slow_tau_yr", "slow_tau_rel_err", "slow_tau_log_ratio",
                 "slow_unstable", "slow_corr", "slow_angle", "pair_corr", "pair_angle")


def test_score_matches_known_solution():
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    row = score(ds, ds.forced, ds.system.A)
    assert abs(row["forced_corr"] - 1) < 1e-12 and row["forced_rel_rmse"] < 1e-12
    assert row["slow_eig_err"] < 1e-10 and row["slow_tau_rel_err"] < 1e-8 and row["slow_tau_log_ratio"] < 1e-8
    assert abs(row["slow_tau_yr"] - DEFAULT.tau1_yr) < 1e-6, row["slow_tau_yr"]   # reported in years
    assert row["slow_unstable"] == 0
    assert row["slow_angle"] < 1e-6 and row["pair_angle"] < 1e-5, row
    assert 1 - row["slow_corr"] < 1e-12 and 1 - row["pair_corr"] < 1e-9, row
    no_operator = score(ds, ds.forced)
    assert all(np.isnan(no_operator[k]) for k in OPERATOR_KEYS)
    row = score(ds, ds.forced, np.linalg.matrix_power(ds.system.A, 3), lag=3)
    assert row["slow_eig_err"] < 1e-10 and row["slow_tau_rel_err"] < 1e-8
    assert abs(row["slow_tau_yr"] - DEFAULT.tau1_yr) < 1e-6, row["slow_tau_yr"]   # and lag-invariant
    assert row["slow_angle"] < 1e-6 and row["pair_angle"] < 1e-5, row


def test_rel_rmse_denominator_is_the_snr_variance():
    """The relative RMSE is scored against the same V_f the SNR is built from.

    centered_rms(f) ** 2 * M is sum_i Var_t f_i exactly, so row (a) of the sweeps figure and the SNR
    of its x axis measure the forced response the same way. The uncentered rms(f) would instead carry
    the forced response's constant offset, which grows with tau_1 and deflates the score by a factor
    2.5 at tau_1 = 100 yr.
    """
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    assert abs(centered_rms(ds.forced) ** 2 * M / ds.V_f - 1) < 1e-12
    # a constant offset added to the forced response must leave the denominator untouched...
    shifted = replace(ds, forced=ds.forced + 3.0)
    assert abs(centered_rms(shifted.forced) / centered_rms(ds.forced) - 1) < 1e-12
    # ...and must still be charged to the estimate through the uncentered numerator
    assert score(ds, ds.forced + 3.0)["forced_rel_rmse"] > 1


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


def test_slow_mode_shape_score_is_invariant_to_sign_and_scale():
    """The score sees only the span, so rescaling or flipping the true mode cannot move it.

    The correlation the figure plots and the angle are the same quantity in two units, so the test pins
    both and their relation.
    """
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    A = propagator(ds.system, W=tilted_slow_mode(ds.system))
    row = score(ds, ds.forced, A)
    angle, corr = row["slow_angle"], row["slow_corr"]
    assert 1 < angle < 89, angle
    assert abs(corr - np.cos(np.deg2rad(angle))) < 1e-12, row
    for factor in (-3.0, 0.1):
        rescaled = replace(ds.system, W=ds.system.W * np.array([factor] + [1.0] * (M - 1)))
        moved = score(replace(ds, system=rescaled), ds.forced, A)
        assert abs(moved["slow_angle"] - angle) < 1e-9 and abs(moved["slow_corr"] - corr) < 1e-12


def test_non_decaying_fit_is_counted_not_scored():
    """A propagator with |lambda_1| > 1 has no decay time: the tau metrics are nan and the fit is flagged."""
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    row = score(ds, ds.forced, propagator(ds.system, lam1=1.01))
    assert row["slow_unstable"] == 1
    assert np.isnan(row["slow_tau_yr"]) and np.isnan(row["slow_tau_rel_err"])
    assert np.isnan(row["slow_tau_log_ratio"])
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
    rows, _ = run_level("total_snr", find_level(cfg, "total_snr", 1 / 3), cfg)
    ours = [r for r in rows if r["method"] == "PullbackDMDc"]
    ref = recovery_metrics(make_dataset(REFERENCE, equal_budget, n_realizations=5))
    assert np.allclose([r["forced_corr"] for r in ours], ref["forced_corr"], atol=1e-12)
    assert np.allclose([r["forced_rel_rmse"] for r in ours], ref["forced_err"], atol=1e-12)
    slow_eig = np.array([r["slow_eig_err"] for r in ours])
    assert np.allclose(slow_eig, np.abs(ref["slow_eig"] - REFERENCE.lam1), atol=1e-12)


def test_run_level_persists_slow_mode_patterns():
    """The patterns saved for the shapes figure are unit norm, signed to the truth, and carry the truth."""
    cfg = with_overrides(DEFAULT, ["n_realizations=3"])
    rows, modes = run_level("forcing_overlap", find_level(cfg, "forcing_overlap", 0.9), cfg)
    w1 = REFERENCE.W[:, 0]  # only the direction of the truth row is compared, so any level's w1 serves

    truth = [m for m in modes if m["method"] == "truth"]
    assert len(truth) == 1, "exactly one truth row per level, since the study moves w_1"
    assert abs(np.linalg.norm(truth[0]["pattern"]) - 1) < 1e-12
    assert abs(abs(truth[0]["pattern"] @ w1) - 1) > 1e-6, "level 0.9 is not the reference w_1"

    fits = [m for m in modes if m["method"] != "truth"]
    assert {m["method"] for m in fits} == {"PullbackDMDc", "LIM", "LIM-opt"}, "LR has no propagator"
    for m in fits:
        assert abs(np.linalg.norm(m["pattern"]) - 1) < 1e-12
        assert m["pattern"] @ truth[0]["pattern"] >= 0, "patterns are signed to agree with the truth"
        assert m["study"] == "forcing_overlap" and m["param_value"] == 0.9
    # every row of the CSV that has a propagator also has a pattern, so the two files stay aligned
    assert len(fits) == len([r for r in rows if r["method"] in {"PullbackDMDc", "LIM", "LIM-opt"}])


def test_save_slow_modes_round_trips_without_pickle():
    """Strings must be stored at full width: a fixed dtype silently truncated the LaTeX param_name."""
    name = r"$\cos\angle(w_1,\hat b)$ a deliberately long label"
    modes = [dict(study="forcing_overlap", param_name=name, param_value=0.9, tau1_yr=20.0,
                  realization=r, method="PullbackDMDc", pattern=np.arange(20.0) + r) for r in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "slow_modes.npz"
        save_slow_modes(modes, path)
        with np.load(path) as archive:  # no allow_pickle: strings are unicode arrays, not objects
            assert archive["param_name"][0] == name, "param_name was truncated"
            assert archive["pattern"].shape == (3, 20)
            assert np.array_equal(archive["pattern"][2], np.arange(20.0) + 2)


def test_fitted_slow_mode_without_a_propagator():
    """LR returns no operator, so it contributes no pattern rather than a nan one."""
    assert fitted_slow_mode(None, 0.9, REFERENCE.W[:, 0]) is None

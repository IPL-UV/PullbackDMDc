import pathlib
import tempfile

import numpy as np

from ablations import M, make_level
from config import DEFAULT
from run_ablation_studies import (centered_rms, decay_time, fitted_slow_mode, run_level, save_slow_modes, score,
                                  slow_index, true_eigs)
from support import CHEAP, cheap_ensemble, recovery_metrics


OPERATOR_KEYS = ("slow_eig", "slow_eig_err", "slow_tau_yr", "slow_tau_rel_err", "slow_tau_log_ratio",
                 "slow_unstable", "slow_corr", "slow_angle", "pair_corr", "pair_angle")


def test_score_matches_known_solution():
    ens = cheap_ensemble()
    row = score(ens, ens["forced"], eig(ens["A"]))
    assert abs(row["forced_corr"] - 1) < 1e-12 and row["forced_rel_rmse"] < 1e-12
    assert row["slow_eig_err"] < 1e-10 and row["slow_tau_rel_err"] < 1e-8 and row["slow_tau_log_ratio"] < 1e-8
    assert abs(row["slow_tau_yr"] - DEFAULT.tau1) < 1e-6, row["slow_tau_yr"]   # reported in years
    assert row["slow_unstable"] == 0
    assert row["slow_angle"] < 1e-6 and row["pair_angle"] < 1e-5, row
    assert 1 - row["slow_corr"] < 1e-12 and 1 - row["pair_corr"] < 1e-9, row
    no_operator = score(ens, ens["forced"])
    assert all(np.isnan(no_operator[k]) for k in OPERATOR_KEYS)
    row = score(ens, ens["forced"], eig(np.linalg.matrix_power(ens["A"], 3)), lag=3)
    assert row["slow_eig_err"] < 1e-10 and row["slow_tau_rel_err"] < 1e-8
    assert abs(row["slow_tau_yr"] - DEFAULT.tau1) < 1e-6, row["slow_tau_yr"]   # and lag-invariant
    assert row["slow_angle"] < 1e-6 and row["pair_angle"] < 1e-5, row


def test_rel_rmse_denominator_is_the_snr_variance():
    """The relative RMSE is scored against the same V_f the SNR is built from.

    centered_rms(f) ** 2 * M is sum_i Var_t f_i exactly, so row (a) of the sweeps figure and the SNR
    of its x axis measure the forced response the same way. The uncentered rms(f) would instead carry
    the forced response's constant offset, which grows with tau_1 and deflates the score by a factor
    2.5 at tau_1 = 100 yr.
    """
    ens = cheap_ensemble()
    assert abs(centered_rms(ens["forced"]) ** 2 * M / ens["Vf"] - 1) < 1e-12
    # a constant offset added to the forced response must leave the denominator untouched...
    assert abs(centered_rms(ens["forced"] + 3.0) / centered_rms(ens["forced"]) - 1) < 1e-12
    # ...and must still be charged to the estimate through the uncentered numerator
    assert score(ens, ens["forced"] + 3.0)["forced_rel_rmse"] > 1


def test_forced_r2_is_one_minus_rel_rmse_squared():
    """R^2 = 1 - SSE/SST: 1 for an exact estimate, below 0 once the error exceeds the forced variance."""
    ens = cheap_ensemble()
    assert abs(score(ens, ens["forced"])["forced_r2"] - 1) < 1e-12
    noisy = ens["forced"] + 0.3 * (ens["data"][0] - ens["forced"])
    row = score(ens, noisy)
    assert abs(row["forced_r2"] - (1 - row["forced_rel_rmse"] ** 2)) < 1e-12 and row["forced_r2"] < 1
    assert score(ens, ens["forced"] + 3.0)["forced_r2"] < 0  # an offset counts, unlike in forced_corr


def test_forced_corr_is_blind_to_offsets_per_latitude():
    """forced_corr centers each latitude over time first, so a constant or per-latitude offset leaves it at 1."""
    ens = cheap_ensemble()
    offsets = np.linspace(-2, 2, M)
    for shifted in (ens["forced"] + 3.0, ens["forced"] + offsets):
        row = score(ens, shifted)
        assert abs(row["forced_corr"] - 1) < 1e-12 and row["forced_rel_rmse"] > 1, row
    # a per-latitude offset is not a uniform one: a plain corrcoef of the raveled fields would see it
    assert np.corrcoef((ens["forced"] + offsets).ravel(), ens["forced"].ravel())[0, 1] < 1 - 1e-3


def eig(A):
    """(eigvals, eigenvectors) of a grid-space propagator: what a method hands to score."""
    return np.linalg.eig(A)


def propagator(ens, W=None, lam1=None):
    """The true system's propagator with its slow eigenvector and/or slow eigenvalue replaced.

    Built from the real block-diagonal Lambda_R, so the oscillating pair stays a conjugate pair and A stays real.
    """
    W = ens["W"] if W is None else W
    Lambda = ens["Lambda_R"].copy()
    if lam1 is not None:
        Lambda[0, 0] = lam1
    return W @ Lambda @ np.linalg.inv(W)


def tilted_slow_mode(ens, tilt=0.3):
    """W with w1 tilted into w4, so the recovered slow mode sits at a comfortably non-zero angle from w1."""
    W = ens["W"].copy()
    W[:, 0] = W[:, 0] + tilt * W[:, 3]
    return W


def test_slow_mode_shape_score_is_invariant_to_sign_and_scale():
    """The score sees only the span, so rescaling or flipping the true mode cannot move it.

    The correlation the figure plots and the angle are the same quantity in two units, so the test pins
    both and their relation.
    """
    ens = cheap_ensemble()
    A = propagator(ens, W=tilted_slow_mode(ens))
    row = score(ens, ens["forced"], eig(A))
    angle, corr = row["slow_angle"], row["slow_corr"]
    assert 1 < angle < 89, angle
    assert abs(corr - np.cos(np.deg2rad(angle))) < 1e-12, row
    for factor in (-3.0, 0.1):
        rescaled = {**ens, "W": ens["W"] * np.array([factor] + [1.0] * (M - 1))}
        moved = score(rescaled, ens["forced"], eig(A))
        assert abs(moved["slow_angle"] - angle) < 1e-9 and abs(moved["slow_corr"] - corr) < 1e-12


def test_non_decaying_fit_is_counted_not_scored():
    """A slow eigenvalue >= 1 has no decay time: the fit is flagged and no slow metric but the eigenvalue is scored."""
    ens = cheap_ensemble()
    for lam1 in (1.01, 1.05):
        row = score(ens, ens["forced"], eig(propagator(ens, lam1=lam1)))
        assert row["slow_unstable"] == 1
        assert abs(row["slow_eig"] - lam1) < 1e-9
        assert all(np.isnan(row[k]) for k in ("slow_tau_yr", "slow_tau_rel_err", "slow_tau_log_ratio",
                                              "slow_corr", "slow_angle")), row
        assert fitted_slow_mode(eig(propagator(ens, lam1=lam1)), ens["W"][:, 0]) is None


def test_slow_index_is_the_largest_real_eigenvalue():
    """The selection never looks at the truth: the largest real eigenvalue, whatever lambda_1 is."""
    assert slow_index(np.array([0.2, 0.9 + 0.3j, 0.9 - 0.3j, 0.5, -0.95])) == 3   # complex ones are skipped
    assert slow_index(np.array([1.2, 0.99, 0.1])) == 0                             # even when it does not decay
    assert slow_index(np.array([-0.3, -0.8, 0.5 + 0.1j, 0.5 - 0.1j])) is None      # no positive real eigenvalue
    assert slow_index(np.array([0.0, -0.1])) is None
    assert slow_index(np.array([0.5 + 0.1j, 0.5 - 0.1j])) is None                  # no real eigenvalue
    # a fit whose slow mode is not the slowest real one is scored on the largest, not on the one nearest lambda_1
    ens = cheap_ensemble()
    eigvals = np.r_[0.999, np.linalg.eigvals(ens["A"])]
    assert eigvals[slow_index(eigvals)] == 0.999


def test_tau_log_ratio_is_symmetric_in_over_and_underestimates():
    """Halving and doubling tau_1 are the same error on the log ratio, unlike the relative error."""
    ens = cheap_ensemble()
    tau1 = decay_time(true_eigs(ens)[0])
    rows = [score(ens, ens["forced"], eig(propagator(ens, lam1=np.exp(-1 / (factor * tau1)))))
            for factor in (0.5, 2.0)]
    assert abs(rows[0]["slow_tau_log_ratio"] - rows[1]["slow_tau_log_ratio"]) < 1e-9
    assert np.allclose([r["slow_tau_rel_err"] for r in rows], [0.5, 1.0], atol=1e-9)


def test_runner_results_match_recovery_metrics():
    rows, _ = run_level("total_snr", CHEAP.snr, CHEAP)
    ours = [r for r in rows if r["method"] == "PullbackDMDc"]
    ens = make_level(CHEAP, "total_snr", CHEAP.snr)
    ref = recovery_metrics(ens, CHEAP.lag)
    assert np.allclose([r["forced_corr"] for r in ours], ref["forced_corr"], atol=1e-12)
    assert np.allclose([r["forced_rel_rmse"] for r in ours], ref["forced_err"], atol=1e-12)
    slow_eig = np.array([r["slow_eig_err"] for r in ours])
    assert np.allclose(slow_eig, np.abs(ref["slow_eig"] - true_eigs(ens)[0] ** CHEAP.lag), atol=1e-12)


def test_run_level_persists_slow_mode_patterns():
    """The patterns saved for the shapes figure are unit norm, signed to the truth, and carry the truth."""
    rows, modes = run_level("obs_noise", 0.5, CHEAP)

    truth = [m for m in modes if m["method"] == "truth"]
    assert len(truth) == 1, "exactly one truth row per level"
    assert abs(np.linalg.norm(truth[0]["pattern"]) - 1) < 1e-12

    fits = [m for m in modes if m["method"] != "truth"]
    assert {m["method"] for m in fits} == {"PullbackDMDc", "LIM"}, "LR and LIM-opt have no propagator"
    for m in fits:
        assert abs(np.linalg.norm(m["pattern"]) - 1) < 1e-12
        assert m["pattern"] @ truth[0]["pattern"] >= 0, "patterns are signed to agree with the truth"
        assert m["study"] == "obs_noise" and m["param_value"] == 0.5
    # every row of the CSV with a decaying slow mode also has a pattern, so the two files stay aligned
    assert len(fits) == len([r for r in rows if r["method"] in {"PullbackDMDc", "LIM"} and r["slow_unstable"] == 0])


def test_save_slow_modes_round_trips_without_pickle():
    """Strings must be stored at full width: a fixed dtype silently truncated the LaTeX param_name."""
    name = r"$\cos\angle(w_1,\hat b)$ a deliberately long label"
    modes = [dict(study="obs_noise", param_name=name, param_value=0.9, tau1_yr=20.0,
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
    assert fitted_slow_mode(None, np.ones(M)) is None

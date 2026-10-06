import numpy as np

from ablation_data import REFERENCE, equal_budget, find_level, make_dataset
from config import DEFAULT, with_overrides
from run_ablation_studies import run_level, score
from support import recovery_metrics


def test_score_matches_known_solution():
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    row = score(ds, ds.forced, ds.system.A)
    assert abs(row["forced_corr"] - 1) < 1e-12 and row["forced_rel_rmse"] < 1e-12
    assert row["slow_eig_err"] < 1e-10 and row["slow_tau_rel_err"] < 1e-8
    assert row["pair_angle"] < 1e-5, row["pair_angle"]
    no_operator = score(ds, ds.forced)
    assert all(np.isnan(no_operator[k]) for k in ("slow_eig_err", "slow_tau_rel_err", "pair_angle"))
    row = score(ds, ds.forced, np.linalg.matrix_power(ds.system.A, 3), lag=3)
    assert row["slow_eig_err"] < 1e-10 and row["slow_tau_rel_err"] < 1e-8 and row["pair_angle"] < 1e-5, row


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

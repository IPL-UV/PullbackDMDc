"""Tests for the config, the ablation harness and the diagnostics. Run with `python test_run_ablations.py`."""

import pathlib
import tempfile
from functools import partial
from types import SimpleNamespace

import numpy as np

from ablations import (
    FORCING_AMPLITUDE,
    M,
    N,
    BASE_OVERLAP,
    PARTIAL_SNR_COMPONENTS,
    REFERENCE,
    SLOW_TIMESCALE_HOLDS,
    TOTAL_SNRS,
    SLOW_TIMESCALES_YR,
    build_reference,
    decay_time,
    eigenvalue,
    empirical_snr,
    equal_budget,
    find_level,
    forced_variance,
    joint_sweep,
    make_dataset,
    modal_coordinates,
    partial_snr_sweep,
    run_modal,
    slow_timescale_sweep,
    study_levels,
    total_snr_sweep,
)
from config import DEFAULT, Config, config_diff, from_json, slug, to_json, with_overrides
from methods import fit_lim, fit_lim_opt, fit_pullback_dmdc, make_methods
from plot_system import plot_ensemble_super_spaghetti, reference_dataset
from run_ablations import run_level, score
from test import recovery_metrics

SMALL = dict(n_realizations=2)


def test_default_reference_unchanged():
    ref = build_reference(DEFAULT)
    assert np.abs(ref.system.A - REFERENCE.A).max() == 0
    assert ref.system.forcing_amplitude == FORCING_AMPLITUDE and ref.base_overlap == BASE_OVERLAP
    assert abs(ref.snr - 1 / 3) < 1e-15 and ref.budget == equal_budget(M)
    assert build_reference(Config()) is ref  # cached


def test_overrides():
    cfg = with_overrides(DEFAULT, ["tau1_yr=50", "total_snrs=[0.1, 1]", "methods=['LIM']"])
    assert cfg.tau1_yr == 50 and cfg.total_snrs == (0.1, 1) and cfg.methods == ("LIM",)
    assert config_diff(cfg) == "tau1_yr=50, total_snrs=(0.1, 1), methods=(LIM,)"
    assert slug(DEFAULT) == "default" and "/" not in slug(cfg)
    hash(cfg)
    for bad in (["tau_1=50"], ["tau1_yr"]):
        try:
            with_overrides(DEFAULT, bad)
        except (KeyError, ValueError):
            continue
        raise AssertionError(f"{bad} should raise")
    ref = build_reference(cfg)
    assert ref.system.lam1 == eigenvalue(600)
    assert abs(forced_variance(make_dataset(ref.system, equal_budget, n_realizations=1).forced) - cfg.forced_variance) < 1e-9


def test_timescale_overrides():
    cfg = with_overrides(DEFAULT, ["tau_p_yr=5", "period_p_yr=8", "complement_eig_range=(0.2, 0.3)"])
    s = build_reference(cfg).system
    assert abs(decay_time(s.rho) - 60) < 1e-9 and abs(2 * np.pi / s.theta - 96) < 1e-9
    assert np.all((s.lam_c >= 0.2) & (s.lam_c <= 0.3))
    assert np.abs(np.sort_complex(np.linalg.eigvals(s.A)) - np.sort_complex(s.eigvals)).max() < 1e-10


def test_variance_overrides():
    cfg = with_overrides(DEFAULT, ["slow_variance=2", "n_realizations=500"])
    ref = build_reference(cfg)
    assert abs(ref.snr - 1 / 4) < 1e-12
    ds = reference_dataset(cfg)
    s, V_f = ds.system, ds.V_f
    assert abs(s.s1_sq - 2 * V_f) < 1e-12 and abs(s.sp_sq - V_f / 2) < 1e-12 and abs(s.sc_sq - V_f / (M - 3)) < 1e-12
    empirical = (modal_coordinates(ds, ds.internal) ** 2).mean(axis=(0, 1))
    assert np.abs(empirical / s.modal_variances - 1).max() < 0.1, empirical / s.modal_variances
    (slow4,) = partial_snr_sweep("slow", factors=[4], cfg=cfg, **SMALL)
    assert abs(slow4.system.s1_sq - 8 * slow4.V_f) < 1e-9
    for ds, snr in zip(total_snr_sweep(cfg=cfg, **SMALL), cfg.total_snrs):
        assert abs(ds.snr / snr - 1) < 1e-12
        assert abs(ds.system.s1_sq / ds.system.sp_sq - 4) < 1e-12  # 2 V_f against V_f / 2: proportions kept


def test_history_override():
    cfg = with_overrides(DEFAULT, ["history=2400", "tau1_yr=1"])
    (ds,) = total_snr_sweep(snrs=[1], cfg=cfg, **SMALL)
    assert ds.system.history == 2400 and ds.system.spinup >= 2400
    long_forcings, short_forcings, history = ds.forcings()
    assert history == 2400 and long_forcings.shape == (2400 + N, 1) and short_forcings.shape == (N, 1)
    # the history is a method setting: the realizations and the forced response must not move
    (base,) = total_snr_sweep(snrs=[1], cfg=with_overrides(DEFAULT, ["tau1_yr=1"]), **SMALL)
    assert np.all(ds.internal == base.internal) and np.abs(ds.forced - base.forced).max() < 1e-12
    assert np.all(ds.y[-N:] == base.y[-N:])


def test_config_json_roundtrip():
    cfg = with_overrides(DEFAULT, ["tau1_yr=50", "mode_overlaps=[0.1, 0.2]", "lag=3"])
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "config.json"
        to_json(cfg, path)
        assert from_json(path) == cfg


def test_score_truth():
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    row = score(ds, ds.forced, ds.system.A)
    assert abs(row["forced_corr"] - 1) < 1e-12 and row["forced_rel_rmse"] < 1e-12
    assert row["slow_eig_err"] < 1e-10 and row["slow_tau_rel_err"] < 1e-8
    assert row["pair_angle"] < 1e-5, row["pair_angle"]
    no_operator = score(ds, ds.forced)
    assert all(np.isnan(no_operator[k]) for k in ("slow_eig_err", "slow_tau_rel_err", "pair_angle"))
    row = score(ds, ds.forced, np.linalg.matrix_power(ds.system.A, 3), lag=3)
    assert row["slow_eig_err"] < 1e-10 and row["slow_tau_rel_err"] < 1e-8 and row["pair_angle"] < 1e-5, row


def noise_free_white_forcing(n_steps=20000):
    # same setting as test.check_dmdc_recovery: white forcing excites every mode, no noise
    s = REFERENCE
    rng = np.random.default_rng(0)
    y = rng.standard_normal(s.spinup + n_steps)
    z = run_modal(s.Lambda_R, np.outer(y, s.W_inv @ s.b), z0=s.W_inv @ rng.standard_normal(M))
    return s, y, z[s.spinup:] @ s.W.T


def test_pullback_noise_free():
    s, y, data = noise_free_white_forcing()
    forced_est, A = fit_pullback_dmdc(data, y[:, None], y[s.spinup:, None], s.spinup)
    row = score(SimpleNamespace(forced=data, system=s), forced_est, A)
    assert row["slow_eig_err"] < 1e-8 and row["pair_angle"] < 1e-4, row
    assert row["forced_corr"] > 1 - 1e-8, row


def test_lag_reaches_pullback():
    # lag 3 on a lag-1 system: x(t+3) = A^3 x(t) + (forcing terms), so the fitted A is close to A^3
    s, y, data = noise_free_white_forcing()
    cfg = with_overrides(DEFAULT, ["lag=3"])
    fit = make_methods(cfg)["PullbackDMDc"]
    assert fit.keywords["lag"] == 3
    _, A = fit(data, y[:, None], y[s.spinup:, None], s.spinup)
    eigvals = np.linalg.eigvals(A)
    assert np.abs(eigvals - s.lam1 ** 3).min() < 1e-2, eigvals
    assert np.abs(eigvals - s.lam1).min() > 1e-3  # not the lag-1 operator
    lags = {name: getattr(fit, "keywords", {}).get("lag") for name, fit in make_methods(cfg).items()}
    assert lags == {"PullbackDMDc": 3, "LIM": 3, "LIM-opt": 3, "LR": None}, lags


def test_matches_recovery_metrics():
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


def test_joint_sweep():
    sweep = joint_sweep(**SMALL)
    assert len(sweep) == len(TOTAL_SNRS) * len(SLOW_TIMESCALES_YR)
    grid = [(snr, tau) for snr in TOTAL_SNRS for tau in SLOW_TIMESCALES_YR]
    for ds, (snr, tau) in zip(sweep, grid):
        assert abs(ds.snr / snr - 1) < 1e-12 and ds.param_value == snr
        assert abs(decay_time(ds.system.lam1) / 12 - tau) < 1e-9
    # the SNR = 1/3 row is the slow-timescale sweep at fixed SNR
    row = [ds for ds in sweep if ds.param_value == 1 / 3]
    for a, b in zip(row, slow_timescale_sweep(hold="snr", **SMALL)):
        assert np.abs(a.data - b.data).max() < 1e-10
    ds = joint_sweep(snrs=[1], taus_yr=[5], n_realizations=200)[0]
    assert abs(empirical_snr(ds).mean() - 1) < 0.1, empirical_snr(ds).mean()


def test_methods():
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    args = (ds.data[0], *ds.forcings())
    methods = make_methods(DEFAULT)
    assert list(methods) == list(DEFAULT.methods)
    for name, fit in methods.items():
        forced_est, A = fit(*args)
        assert forced_est.shape == (N, M), name
        assert (A is None) == (name == "LR"), name
        assert A is None or A.shape == (M, M), name
    # LIM and LIM-opt fit the same lag-1 regression, so their propagators must coincide
    _, A_lim = fit_lim(ds.data[0])
    _, A_lim_opt = fit_lim_opt(ds.data[0])
    assert np.abs(A_lim - A_lim_opt).max() < 1e-8
    assert list(make_methods(with_overrides(DEFAULT, ["methods=['LR', 'LIM']"]))) == ["LR", "LIM"]


def test_study_levels():
    levels = study_levels()
    expected = (len(TOTAL_SNRS) + len(PARTIAL_SNR_COMPONENTS) * len(DEFAULT.partial_snr_factors)
                + len(SLOW_TIMESCALE_HOLDS) * len(SLOW_TIMESCALES_YR) + len(build_reference().mode_overlaps)
                + len(TOTAL_SNRS) * len(SLOW_TIMESCALES_YR))
    assert len(levels) == expected
    for study, _, make in levels[:: len(levels) // 8]:
        (ds,) = make(n_realizations=1)
        assert ds.study == study
    assert len(study_levels(with_overrides(DEFAULT, ["total_snrs=[1]", "slow_timescales_yr=[5, 50]"]))) == (
        1 + 15 + 4 + 6 + 2)


def test_find_level():
    (ds,) = find_level(DEFAULT, "slow_timescale_snr", 50)(n_realizations=1)
    assert abs(decay_time(ds.system.lam1) / 12 - 50) < 1e-9
    (ds,) = find_level(DEFAULT, "joint_snr_timescale", (0.1, 100))(n_realizations=1)
    assert abs(ds.snr - 0.1) < 1e-12 and abs(decay_time(ds.system.lam1) / 12 - 100) < 1e-9
    (ds,) = find_level(DEFAULT, "spatial_overlap", BASE_OVERLAP)(n_realizations=1)
    assert np.abs(ds.system.W - REFERENCE.W).max() < 1e-12
    for study, level in (("total_snrr", 1), ("total_snr", 7)):
        try:
            find_level(DEFAULT, study, level)
        except KeyError:
            continue
        raise AssertionError(f"{study}={level} should raise")


def test_super_spaghetti():
    ds = reference_dataset(with_overrides(DEFAULT, ["n_realizations=3"]))
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "s.png"
        plot_ensemble_super_spaghetti(ds, path, label="test")
        assert path.stat().st_size > 0


def main():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"  passed  {name}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()

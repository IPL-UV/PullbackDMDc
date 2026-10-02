"""Tests for the config, the ablation harness and the diagnostics. Run with `python test_run_ablations.py`."""

import pathlib
import tempfile
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ablations import (
    AR6_ERF_PATH,
    B_HAT,
    FILE_TREND_YEARS,
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
    CO2_FIT,
    CO2_GAUSS_FIT,
    co2_forcing_gauss_model,
    co2_forcing_model,
    co2_monthly,
    decay_time,
    eigenvalue,
    empirical_snr,
    exp_gauss_model,
    gaussian,
    equal_budget,
    file_forcing,
    find_level,
    forced_variance,
    forcing_values,
    joint_sweep,
    load_forcing_file,
    make_dataset,
    modal_coordinates,
    partial_snr_sweep,
    run_modal,
    slow_timescale_sweep,
    study_levels,
    total_snr_sweep,
)
from data_preparation.interpolate_full_forcing import interpolate
from scipy.optimize import curve_fit
from synthetic_system import CO2_ZONAL_FORCING, co2_forcing_pattern, lat_grid
from config import DEFAULT, Config, config_diff, from_json, slug, to_json, with_overrides
from methods import fit_lim, fit_lim_opt, fit_pullback_dmdc, make_methods
from compare_forcings import forced_datasets, plot_forced_response_comparison, plot_forcing_comparison
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


def test_forcing_fit():
    annual = pd.read_csv(AR6_ERF_PATH)
    t, y = annual.year.to_numpy() + 6.5 / 12, annual.co2.to_numpy()
    model = lambda t, c, a, k: c + a * np.exp(k * (t - 2014) / 100)
    p, _ = curve_fit(model, t, y, p0=[0, 2, 2], maxfev=100000)
    assert np.allclose(p, [CO2_FIT["c"], CO2_FIT["a"], CO2_FIT["k"]], rtol=1e-6, atol=0), p
    assert abs(DEFAULT.forcing_efold_yr - 100 / CO2_FIT["k"]) < 1e-12
    t_month, co2 = co2_monthly()
    residual = co2_forcing_model(t_month) - co2
    record = (t_month >= 1915) & (t_month < 2015)
    assert np.sqrt((residual**2).mean()) < 0.04 and np.sqrt((residual[record] ** 2).mean()) < 0.06
    # the record forcing is the model, up to the centering and the amplitude
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    F = co2_forcing_model(1915 + np.arange(N) / 12)
    expected = FORCING_AMPLITUDE * (F - F.mean())
    assert np.abs(ds.y[REFERENCE.spinup:] - expected).max() < 1e-12 * np.abs(expected).max()


def test_forcing_past_constant():
    past = co2_forcing_model(np.linspace(-3000, 1600, 5000))
    assert np.abs(past - CO2_FIT["c"]).max() / (co2_forcing_model(2014) - CO2_FIT["c"]) < 1e-3
    # the longest spin-up (tau_1 = 100 yr): the forced response sits still for millennia before the record
    (ds,) = find_level(DEFAULT, "slow_timescale_snr", 100)(n_realizations=1)
    s = ds.system
    z1 = run_modal(s.Lambda_R, np.outer(ds.y, s.W_inv @ s.b))[:, 0]  # forced slow mode, from the zero start
    assert s.spinup >= 4000 * 12, s.spinup
    # once the zero start has decayed (10 tau_1 = 1000 yr) it sits still: spin-up years 1000-3000 (~1085 BC - 915 AD)
    settled = z1[1000 * 12: 3000 * 12]
    assert np.ptp(settled) < 1e-3 * np.ptp(z1[s.spinup:]), (np.ptp(settled), np.ptp(z1[s.spinup:]))


def test_forcing_efold_override():
    cfg = with_overrides(DEFAULT, ["forcing_efold_yr=30"])
    fast = make_dataset(build_reference(cfg).system, equal_budget, n_realizations=1)
    base = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    assert abs(fast.V_f - cfg.forced_variance) < 1e-9
    # a shorter e-folding time puts more of the record's rise into its last decades
    late = lambda y: (y[-120:].mean() - y[-240:-120].mean()) / np.ptp(y)
    assert late(fast.y[fast.system.spinup:]) > late(base.y[REFERENCE.spinup:])


def test_co2_forcing_pattern():
    b = B_HAT
    assert np.all(b > 0) and np.allclose(b, b[::-1], atol=1e-12)
    assert np.argmax(b) in (M // 2 - 1, M // 2)
    ratio = CO2_ZONAL_FORCING[0] / CO2_ZONAL_FORCING[-1]
    assert abs(ratio / (2.50 / 1.54) - 1) < 0.02, ratio
    raw = co2_forcing_pattern(lat_grid(M))
    assert np.allclose(b, raw / np.linalg.norm(raw), atol=1e-15)


FILE = with_overrides(DEFAULT, ["forcing_source=file"])


def ar6_monthly(column="co2"):
    """Independent of ablations.py: the real pipeline's monthly interpolation of the AR6 file."""
    monthly = interpolate(pd.read_csv(AR6_ERF_PATH)[["year", column]])
    return monthly


def test_forcing_source_default():
    assert DEFAULT.forcing_source == "analytic" and REFERENCE.forcing_source == "analytic"
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    F = co2_forcing_model(1915 + np.arange(N) / 12)
    assert np.abs(ds.y[REFERENCE.spinup:] - FORCING_AMPLITUDE * (F - F.mean())).max() < 1e-12


def test_file_forcing_record():
    # the record is the monthly AR6 CO2 ERF for Jan 1915 - Dec 2014, up to the centering and the amplitude
    monthly = ar6_monthly()
    co2 = monthly.loc[monthly.time.between("1915-01-01", "2014-12-01"), "co2"].to_numpy()
    assert len(co2) == N
    ref = build_reference(FILE)
    ds = make_dataset(ref.system, equal_budget, n_realizations=1)
    y = ds.y[ref.system.spinup:]
    expected = ref.system.forcing_amplitude * (co2 - co2.mean())
    assert np.abs(y - expected).max() < 1e-12 * np.abs(expected).max()
    assert abs(ds.V_f - FILE.forced_variance) < 1e-9
    assert np.max(np.abs(y - make_dataset(REFERENCE, equal_budget, n_realizations=1).y[REFERENCE.spinup:])) > 1e-3


def test_file_forcing_outside_data():
    t, values = load_forcing_file(FILE.forcing_file, "co2")
    early = file_forcing([-3000.0, 0.0, 1000.0, 1700.0, t[0]], FILE.forcing_file, "co2")
    assert np.all(early == values[0]) and abs(values[0]) < 1e-3
    slope = (values[-1] - np.interp(t[-1] - FILE_TREND_YEARS, t, values)) / FILE_TREND_YEARS
    assert abs(slope - 0.0336) < 3e-3, slope
    late = file_forcing([t[-1], t[-1] + 5, t[-1] + 10], FILE.forcing_file, "co2")
    assert late[0] == values[-1] and np.allclose(np.diff(late), 5 * slope, rtol=1e-12)
    # a record past the data uses the trend and stays continuous at the join
    cfg = with_overrides(FILE, ["record_end_year=2030"])
    ds = make_dataset(build_reference(cfg).system, equal_budget, n_realizations=1)
    y = ds.y[ds.system.spinup:]
    assert np.abs(np.diff(y)).max() < 1.5 * np.abs(np.diff(y[:-200])).max()
    assert np.allclose(np.diff(y[-120:]), np.diff(y[-120:])[0], rtol=1e-9)


def test_file_forcing_formats_and_columns():
    annual = load_forcing_file(FILE.forcing_file, "co2")
    with tempfile.TemporaryDirectory() as tmp:
        monthly_path = pathlib.Path(tmp) / "monthly.csv"
        ar6_monthly().to_csv(monthly_path, index=False)  # time, co2: the real pipeline's monthly format
        monthly = load_forcing_file(str(monthly_path), "co2")
        # same months; values agree to the last bit except where pandas' CSV float parsing is off by an ulp
        assert np.array_equal(monthly[0], annual[0]) and np.allclose(monthly[1], annual[1], rtol=1e-12, atol=0)
        ds_monthly = make_dataset(build_reference(with_overrides(FILE, [f"forcing_file={monthly_path}"])).system,
                                  equal_budget, n_realizations=1)
    ds_annual = make_dataset(build_reference(FILE).system, equal_budget, n_realizations=1)
    assert np.allclose(ds_monthly.y[-N:], ds_annual.y[-N:], rtol=1e-12, atol=1e-15)
    absolute = load_forcing_file(str(AR6_ERF_PATH), "co2")
    assert np.array_equal(absolute[1], annual[1])
    total = load_forcing_file(FILE.forcing_file, "total")
    assert len(total[1]) == len(annual[1]) and np.abs(total[1] - annual[1]).max() > 0.1


def test_forcing_source_errors():
    def raises(overrides, error, text):
        try:
            build_reference(with_overrides(DEFAULT, overrides))
        except error as e:
            assert text in str(e), str(e)
            return
        raise AssertionError(f"{overrides} should raise {error.__name__}")
    raises(["forcing_source=data"], ValueError, "analytic, analytic_gauss, file")
    raises(["forcing_source=file", "forcing_file=/nonexistent/forcing.csv"], FileNotFoundError, "/nonexistent")
    raises(["forcing_source=file", "forcing_column=nope"], KeyError, "available")
    with tempfile.TemporaryDirectory() as tmp:
        bad = pathlib.Path(tmp) / "bad.csv"
        pd.DataFrame({"when": [1, 2], "co2": [0.1, 0.2]}).to_csv(bad, index=False)
        raises(["forcing_source=file", f"forcing_file={bad}"], ValueError, "`year`")


GAUSS = with_overrides(DEFAULT, ["forcing_source=analytic_gauss"])


def test_gauss_fit():
    annual = pd.read_csv(AR6_ERF_PATH)
    t, y = annual.year.to_numpy() + 6.5 / 12, annual.co2.to_numpy()
    p0 = [0.02, 1.9, 1.67, 0.05, 1935, 10, 0.08, 1965, 8]
    bounds = ([-1, 0, 0.1, 0, 1850, 3, 0, 1850, 3], [1, 10, 10, 1, 2019, 80, 1, 2019, 80])
    p, _ = curve_fit(exp_gauss_model, t, y, p0=p0, bounds=bounds, maxfev=200000)
    assert np.allclose(p, list(CO2_GAUSS_FIT.values()), rtol=1e-5, atol=0), p
    fit = CO2_GAUSS_FIT
    assert fit["A_pos"] > 0 and fit["A_neg"] > 0 and fit["mu_pos"] < fit["mu_neg"]
    t_month, co2 = co2_monthly()
    record = (t_month >= 1915) & (t_month < 2015)
    for mask in (np.ones_like(record), record):
        gauss = np.sqrt(((co2_forcing_gauss_model(t_month) - co2)[mask] ** 2).mean())
        exp = np.sqrt(((co2_forcing_model(t_month) - co2)[mask] ** 2).mean())
        assert gauss < 0.012 and gauss < exp, (gauss, exp)


def test_gauss_past_constant():
    past = co2_forcing_gauss_model(np.linspace(-3000, 1500, 5000))
    c = CO2_GAUSS_FIT["c"]
    assert np.abs(past - c).max() / (co2_forcing_gauss_model(2014) - c) < 1e-3
    (ds,) = find_level(GAUSS, "slow_timescale_snr", 100)(n_realizations=1)
    s = ds.system
    assert s.forcing_source == "analytic_gauss"
    z1 = run_modal(s.Lambda_R, np.outer(ds.y, s.W_inv @ s.b))[:, 0]
    settled = z1[1000 * 12: 3000 * 12]
    assert np.ptp(settled) < 1e-3 * np.ptp(z1[s.spinup:]), (np.ptp(settled), np.ptp(z1[s.spinup:]))


def test_gauss_source():
    ref = build_reference(GAUSS)
    ds = make_dataset(ref.system, equal_budget, n_realizations=1)
    F = co2_forcing_gauss_model(1915 + np.arange(N) / 12)
    expected = ref.system.forcing_amplitude * (F - F.mean())
    assert np.abs(ds.y[ref.system.spinup:] - expected).max() < 1e-12 * np.abs(expected).max()
    assert abs(ds.V_f - GAUSS.forced_variance) < 1e-9
    base = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    assert np.abs(ds.y[-N:] - base.y[-N:]).max() > 1e-3
    efold = make_dataset(build_reference(with_overrides(GAUSS, ["forcing_efold_yr=30"])).system, equal_budget,
                         n_realizations=1)
    assert np.array_equal(efold.y, ds.y)


def test_gauss_defaults_match_fit():
    fit = CO2_GAUSS_FIT
    assert abs(DEFAULT.gauss_efold_yr - 100 / fit["k"]) < 1e-12
    assert (DEFAULT.gauss_bump_amp, DEFAULT.gauss_bump_year, DEFAULT.gauss_bump_width_yr) == (
        fit["A_pos"], fit["mu_pos"], fit["sigma_pos"])
    assert (DEFAULT.gauss_dip_amp, DEFAULT.gauss_dip_year, DEFAULT.gauss_dip_width_yr) == (
        fit["A_neg"], fit["mu_neg"], fit["sigma_neg"])
    t = np.linspace(1500, 2030, 4000)
    expected = exp_gauss_model(t, **fit)
    assert np.allclose(co2_forcing_gauss_model(t), expected, rtol=1e-12, atol=0)
    assert np.allclose(forcing_values(build_reference(GAUSS).system, t), expected, rtol=1e-12, atol=0)


def gauss_record(overrides, source="analytic_gauss"):
    ref = build_reference(with_overrides(DEFAULT, [f"forcing_source={source}", *overrides]))
    ds = make_dataset(ref.system, equal_budget, n_realizations=1)
    return ref, ds


def test_gauss_overrides():
    t = np.linspace(1750, 2020, 5000)
    no_dip = build_reference(with_overrides(GAUSS, ["gauss_dip_amp=0"])).system
    bump = gaussian(t, DEFAULT.gauss_bump_amp, DEFAULT.gauss_bump_year, DEFAULT.gauss_bump_width_yr)
    exp_only = co2_forcing_gauss_model(t, DEFAULT.gauss_efold_yr, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    assert np.allclose(forcing_values(no_dip, t), exp_only + bump, rtol=1e-14, atol=1e-15)
    moved = build_reference(with_overrides(GAUSS, ["gauss_dip_amp=0", "gauss_bump_year=1940"])).system
    peak = t[np.argmax(forcing_values(moved, t) - exp_only)]
    assert abs(peak - 1940) < 0.1, peak
    _, base = gauss_record([])
    for overrides in (["gauss_efold_yr=40"], ["gauss_dip_amp=0"], ["gauss_bump_year=1940", "gauss_bump_width_yr=8"]):
        ref, ds = gauss_record(overrides)
        assert abs(ds.V_f - DEFAULT.forced_variance) < 1e-9, overrides
        assert np.abs(ds.y[-N:] - base.y[-N:]).max() > 1e-3, overrides
    # the gauss_* fields only shape analytic_gauss
    _, plain = gauss_record([], source="analytic")
    _, tweaked = gauss_record(["gauss_dip_amp=0", "gauss_efold_yr=30"], source="analytic")
    assert np.array_equal(plain.y, tweaked.y)


def test_gauss_validation():
    for bad in (["gauss_bump_amp=-0.1"], ["gauss_dip_amp=-1"], ["gauss_bump_width_yr=0"],
                ["gauss_dip_width_yr=-3"], ["gauss_efold_yr=0"]):
        try:
            build_reference(with_overrides(GAUSS, bad))
        except ValueError as e:
            assert bad[0].split("=")[0] in str(e), str(e)
        else:
            raise AssertionError(f"{bad} should raise")
        build_reference(with_overrides(DEFAULT, bad))  # ignored by the other sources


def test_compare_forcings():
    with tempfile.TemporaryDirectory() as tmp:
        table = plot_forcing_comparison(DEFAULT, pathlib.Path(tmp) / "f.png")
        plot_forced_response_comparison(DEFAULT, pathlib.Path(tmp) / "r.png", member=1)
        assert all((pathlib.Path(tmp) / name).stat().st_size > 0 for name in ("f.png", "r.png"))
    assert table["analytic_gauss"][0] < table["analytic"][0] and table["analytic_gauss"][2] < table["analytic"][2]
    with tempfile.TemporaryDirectory() as tmp:
        no_dip = plot_forcing_comparison(with_overrides(DEFAULT, ["gauss_dip_amp=0"]), pathlib.Path(tmp) / "f.png")
        plot_forced_response_comparison(with_overrides(DEFAULT, ["gauss_dip_amp=0"]), pathlib.Path(tmp) / "r.png")
        assert all((pathlib.Path(tmp) / name).stat().st_size > 0 for name in ("f.png", "r.png"))
    assert no_dip["analytic_gauss"][0] > table["analytic_gauss"][0] and no_dip["analytic"] == table["analytic"]
    datasets = forced_datasets(DEFAULT, member=1)
    internal = [ds.internal[1] for ds in datasets.values()]
    assert all(np.allclose(x, internal[0], rtol=1e-10, atol=1e-12) for x in internal[1:])
    forced = [ds.forced for ds in datasets.values()]
    assert all(np.abs(a - b).max() > 1e-2 for i, a in enumerate(forced) for b in forced[i + 1:])


def test_config_json_roundtrip():
    cfg = with_overrides(DEFAULT, ["tau1_yr=50", "mode_overlaps=[0.1, 0.2]", "lag=3"])
    file_cfg = with_overrides(DEFAULT, ["forcing_source=file", "forcing_file=/some/where/forcing.csv",
                                        "forcing_column=total"])
    assert (file_cfg.forcing_source, file_cfg.forcing_file, file_cfg.forcing_column) == (
        "file", "/some/where/forcing.csv", "total")
    with tempfile.TemporaryDirectory() as tmp:
        for c in (cfg, file_cfg):
            path = pathlib.Path(tmp) / "config.json"
            to_json(c, path)
            assert from_json(path) == c


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

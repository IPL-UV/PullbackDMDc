"""Forcing models, sweep construction and the study matrix."""

import pathlib
import tempfile
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ablation_data import (
    AR6_ERF_PATH,
    FILE_TREND_YEARS,
    M,
    N,
    BASE_OVERLAP,
    PARTIAL_SNR_COMPONENTS,
    REFERENCE,
    SLOW_TIMESCALE_HOLDS,
    TOTAL_SNRS,
    SLOW_TIMESCALES_YR,
    build_reference,
    centered_forcing,
    CO2_FIT,
    CO2_GAUSS_FIT,
    co2_forcing_gauss_model,
    co2_forcing_model,
    co2_monthly,
    decay_time,
    decay_time_yr,
    drive_modal,
    empirical_snr,
    exp_gauss_model,
    gaussian,
    equal_budget,
    file_forcing,
    find_level,
    forced_response,
    forced_variance,
    forcing_values,
    joint_sweep,
    load_forcing_file,
    make_dataset,
    modal_coordinates,
    partial_snr_sweep,
    record_years,
    spinup_years,
    slow_timescale_sweep,
    study_levels,
    total_snr_sweep,
)
from data_preparation.interpolate_full_forcing import interpolate
from config import DEFAULT, Config, with_overrides
from plot_system_diagnostics import file_curve, forcing_curves, reference_dataset
from support import SMALL, assert_unit_b_scale, overridden, record_dataset

RECORD_START_YEAR = DEFAULT.record_end_year - N // 12 + 1  # 1850
EXP = with_overrides(DEFAULT, ["forcing_source=analytic"])
# the least-squares fits; the defaults are their rounded (exp) or chosen (exp + dip) values
EXP_FIT = replace(EXP, forcing_efold_yr=100 / CO2_FIT["k"])
GAUSS_FIT = replace(DEFAULT, forcing_source="analytic_gauss", gauss_efold_yr=100 / CO2_GAUSS_FIT["k"],
                    gauss_bump_amp=CO2_GAUSS_FIT["A_pos"], gauss_bump_year=CO2_GAUSS_FIT["mu_pos"],
                    gauss_bump_width_yr=CO2_GAUSS_FIT["sigma_pos"], gauss_dip_amp=CO2_GAUSS_FIT["A_neg"],
                    gauss_dip_year=CO2_GAUSS_FIT["mu_neg"], gauss_dip_width_yr=CO2_GAUSS_FIT["sigma_neg"])


def test_default_reference_unchanged():
    ref = build_reference(DEFAULT)
    assert np.abs(ref.system.A - REFERENCE.A).max() == 0
    assert ref.base_overlap == BASE_OVERLAP
    assert abs(ref.snr - 1 / 3) < 1e-15
    assert ref.budget == equal_budget(forced_variance(forced_response(ref.system)[1]))
    assert build_reference(Config()) is ref  # cached


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


def test_forcing_fit():
    annual = pd.read_csv(AR6_ERF_PATH)
    t, y = annual.year.to_numpy() + 6.5 / 12, annual.co2.to_numpy()
    model = lambda t, c, a, k: c + a * np.exp(k * (t - 2014) / 100)
    p, _ = curve_fit(model, t, y, p0=[0, 2, 2], maxfev=100000)
    assert np.allclose(p, [CO2_FIT["c"], CO2_FIT["a"], CO2_FIT["k"]], rtol=1e-6, atol=0), p
    assert DEFAULT.forcing_efold_yr == 60 and abs(DEFAULT.forcing_efold_yr - EXP_FIT.forcing_efold_yr) < 2.5
    t_month, co2 = co2_monthly()
    residual = co2_forcing_model(t_month, EXP_FIT.forcing_efold_yr) - co2
    record = (t_month >= RECORD_START_YEAR) & (t_month < 2015)
    assert np.sqrt((residual**2).mean()) < 0.04 and np.sqrt((residual[record] ** 2).mean()) < 0.06
    # the record forcing is the model, up to the centering and the amplitude
    ref = build_reference(EXP)
    ds = make_dataset(ref.system, equal_budget, n_realizations=1)
    F = co2_forcing_model(RECORD_START_YEAR + np.arange(N) / 12, EXP.forcing_efold_yr)
    expected = F - F.mean()
    assert np.abs(ds.y[ref.system.spinup:] - expected).max() < 1e-12 * np.abs(expected).max()


def test_forcing_past_constant():
    past = co2_forcing_model(np.linspace(-3000, 1590, 5000))
    assert np.abs(past - CO2_FIT["c"]).max() / (co2_forcing_model(2014) - CO2_FIT["c"]) < 1e-3
    # the longest spin-up (tau_1 = 100 yr): the forced response sits still for millennia before the record
    (ds,) = find_level(EXP, "slow_timescale_snr", 100)(n_realizations=1)
    assert ds.system.forcing_source == "analytic"
    s = ds.system
    z1 = drive_modal(s, ds.y)[:, 0]  # forced slow mode, from the zero start
    assert s.spinup >= 4000 * 12, s.spinup
    # once the zero start has decayed (10 tau_1 = 1000 yr) it sits still: spin-up years 1000-3000 (~1150 BC - 850 AD)
    settled = z1[1000 * 12: 3000 * 12]
    assert np.ptp(settled) < 1e-3 * np.ptp(z1[s.spinup:]), (np.ptp(settled), np.ptp(z1[s.spinup:]))


def test_forcing_efold_override():
    cfg = with_overrides(EXP, ["forcing_efold_yr=30"])
    fast = make_dataset(build_reference(cfg).system, equal_budget, n_realizations=1)
    base = make_dataset(build_reference(EXP).system, equal_budget, n_realizations=1)
    assert_unit_b_scale(fast.system)
    # a shorter e-folding time puts more of the record's rise into its last decades
    late = lambda y: (y[-120:].mean() - y[-240:-120].mean()) / np.ptp(y)
    assert late(fast.y[fast.system.spinup:]) > late(base.y[base.system.spinup:])


FILE = with_overrides(DEFAULT, ["forcing_source=file"])


def ar6_monthly(column="co2"):
    """Independent of ablation_data.py: the real pipeline's monthly interpolation of the AR6 file."""
    monthly = interpolate(pd.read_csv(AR6_ERF_PATH)[["year", column]])
    return monthly


def test_forcing_source_default():
    assert DEFAULT.forcing_source == "analytic_gauss" and REFERENCE.forcing_source == "analytic_gauss"
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    F = co2_forcing_gauss_model(RECORD_START_YEAR + np.arange(N) / 12)
    assert np.abs(ds.y[REFERENCE.spinup:] - (F - F.mean())).max() < 1e-12


def test_file_forcing_record():
    # the record is the monthly AR6 CO2 ERF for Jan 1850 - Dec 2014, up to the centering and the amplitude
    monthly = ar6_monthly()
    co2 = monthly.loc[monthly.time.between("1850-01-01", "2014-12-01"), "co2"].to_numpy()
    assert len(co2) == N
    ref = build_reference(FILE)
    ds = make_dataset(ref.system, equal_budget, n_realizations=1)
    y = ds.y[ref.system.spinup:]
    expected = co2 - co2.mean()
    assert np.abs(y - expected).max() < 1e-12 * np.abs(expected).max()
    assert_unit_b_scale(ds.system)
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


# analytic_gauss is already the default, so GAUSS == DEFAULT; spelled out for the tests that mean the model
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
    record = (t_month >= RECORD_START_YEAR) & (t_month < 2015)
    for mask in (np.ones_like(record), record):
        fitted = co2_forcing_gauss_model(t_month, 100 / fit["k"], (fit["A_pos"], fit["mu_pos"], fit["sigma_pos"]),
                                         (fit["A_neg"], fit["mu_neg"], fit["sigma_neg"]))
        gauss = np.sqrt(((fitted - co2)[mask] ** 2).mean())
        exp = np.sqrt(((co2_forcing_model(t_month, EXP_FIT.forcing_efold_yr) - co2)[mask] ** 2).mean())
        assert gauss < 0.012 and gauss < exp, (gauss, exp)


def test_gauss_past_constant():
    past = co2_forcing_gauss_model(np.linspace(-3000, 1500, 5000))
    c = CO2_GAUSS_FIT["c"]
    assert np.abs(past - c).max() / (co2_forcing_gauss_model(2014) - c) < 1e-3
    (ds,) = find_level(GAUSS, "slow_timescale_snr", 100)(n_realizations=1)
    s = ds.system
    assert s.forcing_source == "analytic_gauss"
    z1 = drive_modal(s, ds.y)[:, 0]
    settled = z1[1000 * 12: 3000 * 12]
    assert np.ptp(settled) < 1e-3 * np.ptp(z1[s.spinup:]), (np.ptp(settled), np.ptp(z1[s.spinup:]))


def test_gauss_source():
    # the record forcing itself is test_forcing_source_default's; here: the B scale, that it differs from
    # the exp model, and that forcing_efold_yr is ignored
    _, ds = record_dataset(GAUSS)
    assert_unit_b_scale(ds.system)
    base = make_dataset(build_reference(EXP).system, equal_budget, n_realizations=1)
    assert np.abs(ds.y[-N:] - base.y[-N:]).max() > 1e-3
    efold = make_dataset(build_reference(with_overrides(GAUSS, ["forcing_efold_yr=30"])).system, equal_budget,
                         n_realizations=1)
    assert np.array_equal(efold.y, ds.y)


def test_gauss_defaults():
    fit = CO2_GAUSS_FIT
    # rounded fit: e-folding, years and widths within 2.5 yr of the fit; the amplitudes are chosen (one dip)
    assert (DEFAULT.gauss_efold_yr, DEFAULT.gauss_bump_amp, DEFAULT.gauss_dip_amp) == (65, 0, 0.25)
    for value, fitted in ((DEFAULT.gauss_efold_yr, 100 / fit["k"]), (DEFAULT.gauss_bump_year, fit["mu_pos"]),
                          (DEFAULT.gauss_bump_width_yr, fit["sigma_pos"]), (DEFAULT.gauss_dip_year, fit["mu_neg"]),
                          (DEFAULT.gauss_dip_width_yr, fit["sigma_neg"])):
        assert abs(value - fitted) <= 2.5 and value % 5 == 0, (value, fitted)
    t = np.linspace(1500, 2030, 4000)
    expected = exp_gauss_model(t, **fit)
    assert np.allclose(forcing_values(build_reference(GAUSS_FIT).system, t), expected, rtol=1e-12, atol=0)
    dip_only = exp_gauss_model(t, fit["c"], fit["a"], 100 / 65, 0, 1920, 20, 0.25, 1965, 15)
    assert np.allclose(co2_forcing_gauss_model(t), dip_only, rtol=1e-12, atol=1e-14)


def test_gauss_overrides():
    t = np.linspace(1750, 2020, 5000)
    no_dip = build_reference(with_overrides(GAUSS, ["gauss_dip_amp=0"])).system
    bump = gaussian(t, DEFAULT.gauss_bump_amp, DEFAULT.gauss_bump_year, DEFAULT.gauss_bump_width_yr)
    exp_only = co2_forcing_gauss_model(t, DEFAULT.gauss_efold_yr, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    assert np.allclose(forcing_values(no_dip, t), exp_only + bump, rtol=1e-14, atol=1e-15)
    moved = build_reference(with_overrides(GAUSS, ["gauss_dip_amp=0", "gauss_bump_amp=0.05",
                                                   "gauss_bump_year=1940"])).system
    peak = t[np.argmax(forcing_values(moved, t) - exp_only)]
    assert abs(peak - 1940) < 0.1, peak
    _, base = record_dataset(overridden([]))
    for overrides in (["gauss_efold_yr=40"], ["gauss_dip_amp=0"],
                      ["gauss_bump_amp=0.05", "gauss_bump_year=1940", "gauss_bump_width_yr=8"]):
        ref, ds = record_dataset(overridden(overrides))
        assert_unit_b_scale(ds.system)
        assert np.abs(ds.y[-N:] - base.y[-N:]).max() > 1e-3, overrides
    # the gauss_* fields only shape analytic_gauss
    _, plain = record_dataset(overridden([], source="analytic"))
    _, tweaked = record_dataset(overridden(["gauss_dip_amp=0", "gauss_efold_yr=30"], source="analytic"))
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
        build_reference(with_overrides(EXP, bad))  # ignored by the other sources


def test_forcing_centered_everywhere():
    """Generation, the methods' input and the plotted curves all use the forcing centered on the record."""
    for cfg in (DEFAULT, EXP, FILE):
        levels = study_levels(cfg)
        if cfg is not DEFAULT:
            levels = [level for level in levels if level[0] in ("total_snr", "slow_timescale_snr")]
        for study, value, make in levels:
            (ds,) = make(n_realizations=1)
            s = ds.system
            record = ds.y[s.spinup:]
            assert abs(record.mean()) < 1e-12 * np.ptp(record), (cfg.forcing_source, study, value)
            long_forcings, short_forcings, transition_time = ds.forcings()
            assert transition_time == s.spinup and np.array_equal(short_forcings[:, 0], record)
            assert np.array_equal(long_forcings[:, 0], ds.y)
            assert np.allclose(ds.y, centered_forcing(s, spinup_years(s)), rtol=0, atol=1e-14)
        s = build_reference(cfg).system
        record = record_years(s)
        for values, _ in forcing_curves(s, record).values():
            assert abs(values.mean()) < 1e-12
        t, values = file_curve(s)
        in_record = (t >= record[0]) & (t <= record[-1])
        assert in_record.sum() == N and abs(values[in_record].mean()) < 1e-12


def test_joint_sweep():
    sweep = joint_sweep(**SMALL)
    assert len(sweep) == len(TOTAL_SNRS) * len(SLOW_TIMESCALES_YR)
    grid = [(snr, tau) for snr in TOTAL_SNRS for tau in SLOW_TIMESCALES_YR]
    for ds, (snr, tau) in zip(sweep, grid):
        assert abs(ds.snr / snr - 1) < 1e-12 and ds.param_value == snr
        assert abs(decay_time_yr(ds.system.lam1) - tau) < 1e-9
    # the SNR = 1/3 row is the slow-timescale sweep at fixed SNR
    row = [ds for ds in sweep if ds.param_value == 1 / 3]
    for a, b in zip(row, slow_timescale_sweep(hold="snr", **SMALL)):
        assert np.abs(a.data - b.data).max() < 1e-10
    ds = joint_sweep(snrs=[1], taus_yr=[5], n_realizations=200)[0]
    assert abs(empirical_snr(ds).mean() - 1) < 0.1, empirical_snr(ds).mean()


def test_study_levels():
    levels = study_levels()
    expected = (len(TOTAL_SNRS) + len(PARTIAL_SNR_COMPONENTS) * len(DEFAULT.partial_snr_factors)
                + len(SLOW_TIMESCALE_HOLDS) * len(SLOW_TIMESCALES_YR) + len(build_reference().mode_overlaps)
                + len(build_reference().b_overlaps) + len(build_reference().noise_overlaps)
                + len(TOTAL_SNRS) * len(SLOW_TIMESCALES_YR))
    assert len(levels) == expected
    for study, _, make in levels[:: len(levels) // 8]:
        (ds,) = make(n_realizations=1)
        assert ds.study == study
    #  total_snr + partial_snr x3 + slow_timescale x2 + spatial_overlap + forcing_overlap + noise_overlap + joint
    assert len(study_levels(with_overrides(DEFAULT, ["total_snrs=[1]", "slow_timescales_yr=[5, 50]"]))) == (
        1 + 15 + 4 + 6 + 7 + 6 + 2)


def test_find_level():
    (ds,) = find_level(DEFAULT, "slow_timescale_snr", 50)(n_realizations=1)
    assert abs(decay_time_yr(ds.system.lam1) - 50) < 1e-9
    (ds,) = find_level(DEFAULT, "joint_snr_timescale", (0.1, 100))(n_realizations=1)
    assert abs(ds.snr - 0.1) < 1e-12 and abs(decay_time_yr(ds.system.lam1) - 100) < 1e-9
    (ds,) = find_level(DEFAULT, "spatial_overlap", BASE_OVERLAP)(n_realizations=1)
    assert np.abs(ds.system.W - REFERENCE.W).max() < 1e-12
    for study, level in (("total_snrr", 1), ("total_snr", 7)):
        try:
            find_level(DEFAULT, study, level)
        except KeyError:
            continue
        raise AssertionError(f"{study}={level} should raise")

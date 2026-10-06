from types import SimpleNamespace

import numpy as np

from ablation_data import M, N, REFERENCE, equal_budget, make_dataset
from config import DEFAULT, with_overrides
from methods import fit_lim, fit_lim_opt, fit_pullback_dmdc, make_methods
from run_ablation_studies import score
from support import noise_free_white_forcing


def test_fit_methods_shapes_and_scores():
    ds = make_dataset(REFERENCE, equal_budget, n_realizations=1)
    args = (ds.data[0], *ds.forcings())
    methods = make_methods(DEFAULT)
    assert list(methods) == list(DEFAULT.methods)
    for name, fit in methods.items():
        forced_est, A = fit(*args)
        assert forced_est.shape == (N, M), name
        assert (A is None) == (name == "LR"), name
        assert A is None or A.shape == (M, M), name
    _, A_lim = fit_lim(ds.data[0])
    _, A_lim_opt = fit_lim_opt(ds.data[0])
    assert np.abs(A_lim - A_lim_opt).max() < 1e-8
    assert list(make_methods(with_overrides(DEFAULT, ["methods=['LR', 'LIM']"]))) == ["LR", "LIM"]


def test_pullback_recovery_with_noise_free_forcing():
    s, y, data = noise_free_white_forcing()
    forced_est, A = fit_pullback_dmdc(data, y[:, None], y[s.spinup:, None], s.spinup)
    row = score(SimpleNamespace(forced=data, system=s), forced_est, A)
    assert row["slow_eig_err"] < 1e-8 and row["pair_angle"] < 1e-4, row
    assert row["forced_corr"] > 1 - 1e-8, row


def test_lag_reaches_pullback_estimators():
    s, y, data = noise_free_white_forcing()
    cfg = with_overrides(DEFAULT, ["lag=3"])
    fit = make_methods(cfg)["PullbackDMDc"]
    assert fit.keywords["lag"] == 3
    _, A = fit(data, y[:, None], y[s.spinup:, None], s.spinup)
    eigvals = np.linalg.eigvals(A)
    assert np.abs(eigvals - s.lam1 ** 3).min() < 1e-2, eigvals
    assert np.abs(eigvals - s.lam1).min() > 1e-3
    lags = {name: getattr(fit, "keywords", {}).get("lag") for name, fit in make_methods(cfg).items()}
    assert lags == {"PullbackDMDc": 3, "LIM": 3, "LIM-opt": 3, "LR": None}, lags

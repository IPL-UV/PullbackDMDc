import numpy as np

from ablations import M, N
from config import DEFAULT, with_overrides
from methods import fit_lim, fit_lim_opt, fit_pullback_dmdc, make_methods
from run_ablation_studies import score
from support import cheap_ensemble, noise_free_white_forcing, true_eigs


def test_fit_methods_shapes_and_scores():
    ens = cheap_ensemble()
    data, spinup = ens["data"][0], len(ens["y_full"]) - N
    args = (data, ens["y_full"][:, None], ens["y"][:, None], spinup)
    methods = make_methods(DEFAULT)
    assert list(methods) == list(DEFAULT.methods)
    for name, fit in methods.items():
        forced_est, A = fit(*args)
        assert forced_est.shape == (N, M), name
        assert (A is None) == (name == "LR"), name
        assert A is None or A.shape == (M, M), name
    _, A_lim = fit_lim(data)
    _, A_lim_opt = fit_lim_opt(data)
    assert np.abs(A_lim - A_lim_opt).max() < 1e-8
    assert list(make_methods(with_overrides(DEFAULT, ["methods=['LR', 'LIM']"]))) == ["LR", "LIM"]


def test_pullback_recovery_with_noise_free_forcing():
    ens, y, data, spinup = noise_free_white_forcing()
    forced_est, A = fit_pullback_dmdc(data, y[:, None], y[spinup:, None], spinup)
    row = score({**ens, "forced": data}, forced_est, A)
    assert row["slow_eig_err"] < 1e-8 and row["pair_angle"] < 1e-4, row
    assert row["forced_corr"] > 1 - 1e-8, row


def test_lag_reaches_pullback_estimators():
    ens, y, data, spinup = noise_free_white_forcing()
    lam1, _ = true_eigs(ens)
    fit = make_methods(DEFAULT)["PullbackDMDc"]
    assert DEFAULT.lag == 3 and fit.keywords["lag"] == 3
    _, A = fit(data, y[:, None], y[spinup:, None], spinup)
    eigvals = np.linalg.eigvals(A)
    assert np.abs(eigvals - lam1 ** 3).min() < 1e-2, eigvals
    assert np.abs(eigvals - lam1).min() > 1e-3
    lags = {name: getattr(fit, "keywords", {}).get("lag")
            for name, fit in make_methods(with_overrides(DEFAULT, ["lag=1"])).items()}
    assert lags == {"PullbackDMDc": 1, "LIM": 1, "LIM-opt": 1, "LR": None}, lags

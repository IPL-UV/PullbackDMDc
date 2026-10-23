import numpy as np

from ablations import M, T
from config import DEFAULT, with_overrides
from methods import compute_eofs, fit_pullback_dmdc, make_methods
from run_ablation_studies import score
from support import cheap_ensemble, noise_free_white_forcing, true_eigs


def test_fit_methods_shapes():
    ens = cheap_ensemble()
    data, spinup = ens["data"][0], len(ens["y_full"]) - T
    args = (data, ens["y_full"][:, None], ens["y"][:, None], spinup, compute_eofs(data))
    methods = make_methods(DEFAULT)
    assert list(methods) == list(DEFAULT.methods)
    for name, fit in methods.items():
        forced_est, eig = fit(*args)
        assert forced_est.shape == (T, M), name
        assert (eig is None) == (name in {"LR", "LIM-opt"}), name
        if eig is not None:
            assert eig[0].shape == (M,) and eig[1].shape == (M, M), name
    assert list(make_methods(with_overrides(DEFAULT, ["methods=['LR', 'LIM']"]))) == ["LR", "LIM"]


def test_methods_keep_the_data_mean():
    """Each class centers the data with the shared data_mean and adds it back in predict(), as on real data."""
    ens = cheap_ensemble()
    data, spinup = ens["data"][0] + 5.0, len(ens["y_full"]) - T   # a member far from zero mean
    eofs = compute_eofs(data)
    assert np.allclose(eofs["data_mean"], data.mean(0))
    lr, _ = make_methods(DEFAULT)["LR"](data, ens["y_full"][:, None], ens["y"][:, None], spinup, eofs)
    assert np.allclose(lr.mean(0), data.mean(0))  # LR's forced estimate carries the data mean


def test_pullback_recovery_with_noise_free_forcing():
    """Noise free, the dynamics are recovered; the forced response nearly so.

    Not exactly: the class centers the data with its record mean m (as on real data), and the centered data obey
    x(t) = A x(t-lag) + B y(t) - (I - A^lag) m, a constant the regression has no intercept for.
    """
    ens, y, data, spinup = noise_free_white_forcing()
    forced_est, eig = fit_pullback_dmdc(data, y[:, None], y[spinup:, None], spinup, compute_eofs(data))
    row = score({**ens, "forced": data}, forced_est, eig)
    assert row["slow_eig_err"] < 1e-6 and row["pair_angle"] < 1e-2, row
    assert row["forced_corr"] > 0.99, row


def test_lag_reaches_pullback_estimators():
    ens, y, data, spinup = noise_free_white_forcing()
    lam1, _ = true_eigs(ens)
    fit = make_methods(with_overrides(DEFAULT, ["lag=3"]))["PullbackDMDc"]
    assert fit.keywords["lag"] == 3
    eigvals, _ = fit(data, y[:, None], y[spinup:, None], spinup, compute_eofs(data))[1]
    assert np.abs(eigvals - lam1 ** 3).min() < 1e-2, eigvals
    assert np.abs(eigvals - lam1).min() > 1e-3
    lags = {name: getattr(fit, "keywords", {}).get("lag")
            for name, fit in make_methods(with_overrides(DEFAULT, ["lag=1"])).items()}
    assert lags == {"PullbackDMDc": 1, "LIM": 1, "LIM-opt": 1, "LR": None}, lags

"""Forced-response estimators fitted on the synthetic data: identity EOFs, one realization at a time.

make_methods(cfg) binds cfg.lag and cfg.optlag. Each fit returns (forced_est (N, M), A) with A the fitted
propagator over cfg.lag steps in the column convention x(t+lag) = A x(t), or None for methods without dynamics.
"""

import pathlib
import sys
from functools import partial

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ablation_data import M
from config import DEFAULT
from utils.pullback_dmdc import PullbackDMDc
from utils.lim import LIM
from utils.lim_opt import LIM_opt
from utils.lr import LR


def identity_eofs(data):
    return {"data_mean": np.zeros(M), "eofs": np.eye(M), "pcs": data}


def fit_pullback(data, long_forcings, short_forcings, spinup, lag=1):
    model = PullbackDMDc(truncation=M, lag=lag, transition_time=spinup)
    model.fit(data, short_forcings=short_forcings, long_forcings=long_forcings,
              precomputed_eofs=identity_eofs(data))
    model.compute_modes()
    return model


def fit_pullback_dmdc(data, long_forcings, short_forcings, spinup, lag=1):
    model = fit_pullback(data, long_forcings, short_forcings, spinup, lag=lag)
    return model.predict(), model.A


def fit_lim(data, *_, lag=1):
    model = LIM(truncation=M, lag=lag)
    model.fit(data, precomputed_eofs=identity_eofs(data))
    return model.predict(), model.A


def fit_lim_opt(data, *_, lag=1, optlag=DEFAULT.optlag):
    svals = data.std(axis=0, ddof=1)
    model = LIM_opt(truncation=M, lag=lag, optlag=optlag)
    model.fit(data, precomputed_eofs={**identity_eofs(data), "svals": svals})
    # M is a row-convention regression on the whitened pcs x / svals
    return model.predict(), svals[:, None] * model.M.T / svals[None, :]


def fit_lr(data, _long_forcings, short_forcings, *_):
    model = LR(truncation=M)
    model.fit(data, short_forcings, precomputed_eofs=identity_eofs(data))
    return model.predict(), None


def make_methods(cfg=DEFAULT):
    available = {
        "PullbackDMDc": partial(fit_pullback_dmdc, lag=cfg.lag),
        "LIM": partial(fit_lim, lag=cfg.lag),
        "LIM-opt": partial(fit_lim_opt, lag=cfg.lag, optlag=cfg.optlag),
        "LR": fit_lr,
    }
    unknown = set(cfg.methods) - set(available)
    if unknown:
        raise KeyError(f"unknown methods {sorted(unknown)}; available: {', '.join(available)}")
    return {name: available[name] for name in cfg.methods}

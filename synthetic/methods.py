"""Forced-response estimators fitted on the synthetic data, used exactly as in the real-data pipeline.

The EOFs and data mean of a realization are computed once (data_preparation/compute_eofs.run_eofs, full truncation M)
and passed to every method as precomputed_eofs, as experiments.py does; each class then centers, fits and adds the
mean back in predict() as it does on real data.

make_methods(cfg) binds cfg.lag and cfg.optlag. Each fit returns (forced_est (N, M), modes), where modes is
(eigvals, grid-space eigenvectors eofs.T @ W) of the fitted lag-step propagator, or None for methods without one
(LR, and LIM-opt, whose operator the real-data pipeline never uses).
"""

import pathlib
import sys
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ablations import M
from config import DEFAULT
from data_preparation.compute_eofs import run_eofs
from utils.pullback_dmdc import PullbackDMDc
from utils.lim import LIM
from utils.lim_opt import LIM_opt
from utils.lr import LR


def compute_eofs(data):
    """The precomputed_eofs dict of one realization, as data_preparation/compute_eofs.py stores it."""
    data_mean, eofs, pcs, svals = run_eofs(data, truncation=M)
    return {"data_mean": data_mean, "eofs": eofs, "pcs": pcs, "svals": svals}


def fit_pullback(data, long_forcings, short_forcings, spinup, eofs, lag=1):
    model = PullbackDMDc(truncation=M, lag=lag, transition_time=spinup)
    model.fit(data, short_forcings=short_forcings, long_forcings=long_forcings, precomputed_eofs=eofs)
    model.compute_modes()
    return model


def fit_pullback_dmdc(data, long_forcings, short_forcings, spinup, eofs, lag=1):
    model = fit_pullback(data, long_forcings, short_forcings, spinup, eofs, lag=lag)
    return model.predict(), (model.eigvals, model.modes)


def fit_lim(data, _long_forcings, _short_forcings, _spinup, eofs, lag=1):
    model = LIM(truncation=M, lag=lag)
    model.fit(data, precomputed_eofs=eofs)
    model.compute_modes()
    return model.predict(), (model.eigvals, model.eofs.T @ model.W)


def fit_lim_opt(data, _long_forcings, _short_forcings, _spinup, eofs, lag=1, optlag=DEFAULT.optlag):
    model = LIM_opt(truncation=M, lag=lag, optlag=optlag)
    model.fit(data, precomputed_eofs=eofs)
    return model.predict(), None


def fit_lr(data, _long_forcings, short_forcings, _spinup, eofs):
    model = LR(truncation=M)
    model.fit(data, short_forcings, precomputed_eofs=eofs)
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

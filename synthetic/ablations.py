"""Ablation studies of the synthetic system: each study level is a set of generate_ensemble arguments.

The system itself is synthetic_generator.py's; this module only says how each study bends the reference
(the config) away from it. A level is simulated with

    ens = make_level(cfg, "total_snr", 1)       # the dict generate_ensemble returns
    ens = make_level(cfg)                       # the reference system

Every level restarts its noise from cfg.noise_seed, so all levels see the same ensemble members and differ only
in how the data are generated. The exception is levels with a different spin-up (the tau1 studies): they use up
a different number of draws before the record starts, so their members are different draws.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from config import DEFAULT, generator_kwargs
from synthetic_generator import generate_ensemble

M, T = 20, 1980  # latitudes, record months (Jan 1850 - Dec 2014): fixed in synthetic_generator.py
LAT = np.linspace(-90, 90, M)  # the grid of make_setup, in degrees
NOISE_COMPONENTS = {"slow": 1, "pair": 2, "complement": 3}  # index of each internal component in cfg.variances


def scaled_variances(variances, factor, index=None):
    """variances with the noise components (all of them, or only variances[index]) multiplied by factor."""
    scaled = [index] if index else range(1, len(variances))
    return tuple(v * factor if i in scaled else v for i, v in enumerate(variances))


@lru_cache(maxsize=None)
def forced_variance(cfg, tau1):
    """V_f of the system with slow decay time tau1: the forced response does not depend on the noise."""
    return generate_ensemble(1, **{**generator_kwargs(cfg), "tau1": tau1}, rng=0)["Vf"]


@dataclass(frozen=True)
class Study:
    title: str
    param_name: str
    levels: Callable  # cfg -> the study's levels
    kwargs: Callable  # (cfg, level) -> the generate_ensemble arguments that differ from the reference
    param_value: Callable = lambda level: level


STUDIES = {
    "total_snr": Study(
        "total SNR", "SNR",
        levels=lambda cfg: cfg.total_snrs,
        kwargs=lambda cfg, snr: dict(variances=scaled_variances(cfg.variances, cfg.snr / snr)),
    ),
    **{f"partial_snr_{name}": Study(
        f"partial SNR: {name}", f"{name} variance factor",
        levels=lambda cfg: cfg.partial_snr_factors,
        kwargs=lambda cfg, factor, index=index: dict(variances=scaled_variances(cfg.variances, factor, index)),
    ) for name, index in NOISE_COMPONENTS.items()},
    "slow_timescale_snr": Study(
        r"slow timescale ($\mathrm{SNR}$ fixed)", r"$\tau_1$ (yr)",
        levels=lambda cfg: cfg.slow_timescales_yr,
        kwargs=lambda cfg, tau: dict(tau1=tau),
    ),
    # the variances are relative to each level's own V_f, so holding the absolute (modal) variances at the
    # reference's means rescaling the noise components by V_f(reference) / V_f(level); the SNR then follows V_f
    "slow_timescale_modal_variance": Study(
        r"slow timescale ($V_1$ fixed)", r"$\tau_1$ (yr)",
        levels=lambda cfg: cfg.slow_timescales_yr,
        kwargs=lambda cfg, tau: dict(tau1=tau, variances=scaled_variances(
            cfg.variances, forced_variance(cfg, cfg.tau1) / forced_variance(cfg, tau))),
    ),
    # the measurement-noise ratio V_eps / V_f alone, the internal components fixed: the SNR falls as it grows
    "obs_noise": Study(
        "measurement noise", r"$V_\varepsilon/V_f$",
        levels=lambda cfg: cfg.obs_noise_ratios,
        kwargs=lambda cfg, ratio: dict(variances=(*cfg.variances[:4], ratio * cfg.variances[0])),
    ),
    # the complement modes rotated toward w1 by the overlap alpha, so the noise projects onto the slow pattern
    "overlap": Study(
        "spatial overlap", r"$\alpha$",
        levels=lambda cfg: cfg.alphas,
        kwargs=lambda cfg, alpha: dict(alpha=alpha),
    ),
    "joint_snr_timescale": Study(
        "SNR x slow timescale", "SNR",
        levels=lambda cfg: [(snr, tau) for snr in cfg.total_snrs for tau in cfg.slow_timescales_yr],
        kwargs=lambda cfg, level: dict(tau1=level[1], variances=scaled_variances(cfg.variances, cfg.snr / level[0])),
        param_value=lambda level: level[0],
    ),
}
# the columns of the default results figures, left to right; --studies overrides them
# the partial-SNR and fixed-variance timescale studies are still run, and drawn only on request (--studies)
DEFAULT_STUDIES = ("total_snr", "slow_timescale_snr", "obs_noise", "overlap")
LINEAR_X = {"overlap"}  # starts at 0, so a log x-axis is impossible


def level_kwargs(cfg=DEFAULT, study=None, level=None):
    """All generate_ensemble arguments of one study level (the reference system when study is None)."""
    return {**generator_kwargs(cfg), **(STUDIES[study].kwargs(cfg, level) if study else {})}


def make_level(cfg=DEFAULT, study=None, level=None):
    return generate_ensemble(cfg.n_realizations, **level_kwargs(cfg, study, level), rng=cfg.noise_seed)


def study_levels(cfg=DEFAULT):
    """(study, level) for every level of every study under cfg."""
    return [(study, level) for study, spec in STUDIES.items() for level in spec.levels(cfg)]


def find_level(cfg, study, value):
    """The level of `study` matching value to within 1e-9 relative (a pair (snr, tau1) for the joint study)."""
    if study not in STUDIES:
        raise KeyError(f"unknown study {study!r}; valid studies: {', '.join(STUDIES)}")
    levels = STUDIES[study].levels(cfg)
    for level in levels:
        if np.allclose(level, value, rtol=1e-9, atol=0):
            return level
    raise KeyError(f"no level {value!r} in {study}; levels: {list(levels)}")

"""Synthetic datasets for the ablation studies: total SNR, partial SNR, slow timescale and spatial overlap.

Every tunable value lives in config.Config; the module-level constants below are the DEFAULT config's values,
kept so existing scripts can import them.
"""

from dataclasses import dataclass, replace
from functools import lru_cache, partial
from typing import Tuple

import numpy as np

from config import DEFAULT
from synthetic_system import make_W

M = 20
N = 1200
SPINUP_DECAY_TIMES = 40
SPINUP_CHUNK = 1200
MIN_NOISE_SPINUP = 1200
PARTIAL_SNR_COMPONENTS = dict(slow="s1_sq", pair="sp_sq", complement="sc_sq")
SLOW_TIMESCALE_HOLDS = ("snr", "modal_variance")

HISTORY = DEFAULT.history
N_REALIZATIONS = DEFAULT.n_realizations
PATTERN_SEED = DEFAULT.pattern_seed
EIGENVALUE_SEED = DEFAULT.eigenvalue_seed
COMPLEMENT_EIG_RANGE = DEFAULT.complement_eig_range
TAU1_YR = DEFAULT.tau1_yr
TAU_P_YR = DEFAULT.tau_p_yr
PERIOD_P_YR = DEFAULT.period_p_yr
TOTAL_SNRS = list(DEFAULT.total_snrs)
PARTIAL_SNR_FACTORS = list(DEFAULT.partial_snr_factors)
SLOW_TIMESCALES_YR = list(DEFAULT.slow_timescales_yr)


def eigenvalue(tau_months):
    return np.exp(-1 / tau_months)


def decay_time(eig):
    return -1 / np.log(np.abs(eig))


@dataclass(frozen=True, eq=False)
class System:
    W: np.ndarray
    W_inv: np.ndarray
    b: np.ndarray
    lam1: float
    rho: float
    theta: float
    lam_c: np.ndarray
    forcing_amplitude: float = 1.0
    s1_sq: float = np.nan
    sp_sq: float = np.nan
    sc_sq: float = np.nan
    history: int = HISTORY

    @property
    def Lambda_R(self):
        c, s = self.rho * np.cos(self.theta), self.rho * np.sin(self.theta)
        Lambda = np.diag(np.concatenate([[self.lam1, c, c], self.lam_c]))
        Lambda[1, 2], Lambda[2, 1] = s, -s
        return Lambda

    @property
    def A(self):
        return self.W @ self.Lambda_R @ self.W_inv

    @property
    def eigvals(self):
        pair = self.rho * np.exp(1j * self.theta)
        return np.concatenate([[self.lam1, pair, np.conj(pair)], self.lam_c])

    @property
    def modal_variances(self):
        return np.concatenate([[self.s1_sq, self.sp_sq, self.sp_sq], np.full(len(self.lam_c), self.sc_sq)])

    @property
    def noise_variances(self):
        return self.modal_variances * (1 - np.abs(self.eigvals) ** 2)

    @property
    def noise_spinup(self):
        # independent of history, so changing the methods' forcing history leaves the realizations unchanged
        return max(int(np.ceil(SPINUP_DECAY_TIMES * decay_time(self.eigvals).max())), MIN_NOISE_SPINUP)

    @property
    def spinup(self):
        """Length of the forcing series before the record: the noise spin-up, extended to cover the history."""
        return max(self.noise_spinup, self.history)


def forcing_F(t_yr):
    return (t_yr - 9 * np.tanh(0.1 * t_yr - 4.5) + 0.25 * np.exp(0.05 * t_yr)) / 30


def forcing_series(system):
    F = forcing_F(np.arange(-system.spinup, N) / 12)
    # centered on the record; the same offset is kept during spin-up
    return system.forcing_amplitude * (F - F[system.spinup:].mean())


def run_modal(Lambda, drive, z0=None):
    z = np.zeros(drive.shape[1:]) if z0 is None else z0
    out = np.empty_like(drive)
    for t in range(len(drive)):
        z = z @ Lambda.T + drive[t]
        out[t] = z
    return out


def forced_response(system):
    y = forcing_series(system)
    z = run_modal(system.Lambda_R, np.outer(y, system.W_inv @ system.b))
    return y, z[system.spinup:] @ system.W.T


def internal_variability(system, n_realizations, seed):
    # separate streams so the record noise is shared by systems with different spin-ups
    spinup_rng, record_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    noise_std = np.sqrt(system.noise_variances)
    z = np.zeros((n_realizations, M))
    for start in range(0, system.noise_spinup, SPINUP_CHUNK):
        steps = min(SPINUP_CHUNK, system.noise_spinup - start)
        z = run_modal(system.Lambda_R, noise_std * spinup_rng.standard_normal((steps, n_realizations, M)), z)[-1]
    z = run_modal(system.Lambda_R, noise_std * record_rng.standard_normal((N, n_realizations, M)), z)
    return (z @ system.W.T).transpose(1, 0, 2)


def forced_variance(forced):
    return forced.var(axis=0).sum()


def internal_variance(s1_sq, sp_sq, sc_sq):
    return s1_sq + 2 * sp_sq + (M - 3) * sc_sq


def theoretical_snr(V_f, s1_sq, sp_sq, sc_sq):
    return V_f / internal_variance(s1_sq, sp_sq, sc_sq)


def config_budget(V_f, cfg=DEFAULT, scale=1.0, component=None, factor=1.0):
    """Modal variances from cfg's variance shares (units of V_f); scale multiplies all, factor one component."""
    slow, pair, complement = cfg.variances
    budget = dict(s1_sq=slow * V_f, sp_sq=pair * V_f / 2, sc_sq=complement * V_f / (M - 3))
    if scale != 1.0:
        budget = {k: v * scale for k, v in budget.items()}
    if component is not None:
        budget[component] *= factor
    return budget


def equal_budget(V_f):
    return config_budget(V_f)


def pair_plane_overlap(W):
    Q, _ = np.linalg.qr(W[:, 1:3])
    return np.linalg.norm(Q.T @ W[:, 0]) / np.linalg.norm(W[:, 0])


def tilt_slow(W, overlap):
    Q, _ = np.linalg.qr(W[:, 1:3])
    parallel = Q @ (Q.T @ W[:, 0])
    perpendicular = W[:, 0] - parallel
    W = W.copy()
    W[:, 0] = (np.sqrt(1 - overlap**2) * perpendicular / np.linalg.norm(perpendicular)
               + overlap * parallel / np.linalg.norm(parallel))
    W_inv = np.vstack([np.linalg.pinv(W[:, :3]), W[:, 3:].T])
    return W, W_inv


@dataclass(eq=False)
class SyntheticDataset:
    study: str
    param_name: str
    param_value: float
    system: System
    y: np.ndarray  # (spinup + N,)
    forced: np.ndarray  # (N, M)
    internal: np.ndarray  # (n_realizations, N, M)

    @property
    def data(self):
        return self.forced + self.internal

    @property
    def V_f(self):
        return forced_variance(self.forced)

    @property
    def snr(self):
        s = self.system
        return theoretical_snr(self.V_f, s.s1_sq, s.sp_sq, s.sc_sq)

    def forcings(self, history=None):
        history = self.system.history if history is None else history
        long_forcings = self.y[self.system.spinup - history:, None]
        return long_forcings, long_forcings[history:], history


def make_dataset(system, budget, *, study="", param_name="", param_value=np.nan,
                 n_realizations=N_REALIZATIONS, seed=0):
    y, forced = forced_response(system)
    system = replace(system, **budget(forced_variance(forced)))
    return SyntheticDataset(
        study=study, param_name=param_name, param_value=float(param_value), system=system,
        y=y, forced=forced, internal=internal_variability(system, n_realizations, seed),
    )


def empirical_snr(ds):
    return ds.V_f / ds.internal.var(axis=1).sum(axis=-1)


def modal_coordinates(ds, x):
    return x @ ds.system.W_inv.T


@dataclass(frozen=True, eq=False)
class Reference:
    unit_system: System  # forcing amplitude 1
    system: System  # forcing amplitude calibrated so that V_f = cfg.forced_variance
    budget: dict
    snr: float
    base_overlap: float
    mode_overlaps: Tuple[float, ...]
    phi: np.ndarray


@lru_cache(maxsize=None)
def build_reference(cfg=DEFAULT):
    W, W_inv, b, phi = make_W(M, cfg.pattern_seed)
    unit = System(
        W=W, W_inv=W_inv, b=b,
        lam1=eigenvalue(12 * cfg.tau1_yr), rho=eigenvalue(12 * cfg.tau_p_yr), theta=2 * np.pi / (12 * cfg.period_p_yr),
        lam_c=np.random.default_rng(cfg.eigenvalue_seed).uniform(*cfg.complement_eig_range, size=M - 3),
        history=cfg.history,
    )
    amplitude = np.sqrt(cfg.forced_variance / forced_variance(forced_response(unit)[1]))
    budget = config_budget(cfg.forced_variance, cfg)
    base_overlap = pair_plane_overlap(W)
    overlaps = cfg.mode_overlaps if cfg.mode_overlaps is not None else (0.0, 0.25, base_overlap, 0.75, 0.9, 0.95)
    return Reference(unit_system=unit, system=replace(unit, forcing_amplitude=amplitude), budget=budget,
                     snr=theoretical_snr(cfg.forced_variance, **budget), base_overlap=base_overlap,
                     mode_overlaps=tuple(overlaps), phi=phi)


_REF = build_reference(DEFAULT)
W_REF, W_INV_REF, B_HAT, PHI = _REF.system.W, _REF.system.W_inv, _REF.system.b, _REF.phi
UNIT_FORCING_REFERENCE = _REF.unit_system
FORCING_AMPLITUDE = _REF.system.forcing_amplitude
REFERENCE = _REF.system
REFERENCE_BUDGET = dict(_REF.budget)
REFERENCE_SNR = _REF.snr
BASE_OVERLAP = _REF.base_overlap
MODE_OVERLAPS = list(_REF.mode_overlaps)


def scaled_budget(V_f, factor):
    return config_budget(V_f, scale=factor)


def component_budget(V_f, key, factor):
    return config_budget(V_f, component=key, factor=factor)


def _dataset_kwargs(cfg, kwargs):
    return {"n_realizations": cfg.n_realizations, "seed": cfg.noise_seed, **kwargs}


def total_snr_sweep(snrs=None, cfg=DEFAULT, **kwargs):
    ref = build_reference(cfg)
    return [
        make_dataset(ref.system, partial(config_budget, cfg=cfg, scale=ref.snr / snr),
                     study="total_snr", param_name="SNR", param_value=snr, **_dataset_kwargs(cfg, kwargs))
        for snr in (cfg.total_snrs if snrs is None else snrs)
    ]


def partial_snr_sweep(component, factors=None, cfg=DEFAULT, **kwargs):
    key = PARTIAL_SNR_COMPONENTS[component]
    ref = build_reference(cfg)
    return [
        make_dataset(ref.system, partial(config_budget, cfg=cfg, component=key, factor=factor),
                     study=f"partial_snr_{component}", param_name=f"{component} variance factor",
                     param_value=factor, **_dataset_kwargs(cfg, kwargs))
        for factor in (cfg.partial_snr_factors if factors is None else factors)
    ]


def slow_timescale_sweep(taus_yr=None, hold="snr", cfg=DEFAULT, **kwargs):
    if hold not in SLOW_TIMESCALE_HOLDS:
        raise ValueError(f"hold must be one of {SLOW_TIMESCALE_HOLDS}")
    ref = build_reference(cfg)
    budget = partial(config_budget, cfg=cfg) if hold == "snr" else partial(_fixed_budget, ref.budget)
    return [
        make_dataset(replace(ref.system, lam1=eigenvalue(12 * tau)), budget,
                     study=f"slow_timescale_{hold}", param_name=r"$\tau_1$ (yr)", param_value=tau,
                     **_dataset_kwargs(cfg, kwargs))
        for tau in (cfg.slow_timescales_yr if taus_yr is None else taus_yr)
    ]


def _fixed_budget(budget, V_f):
    return dict(budget)


def spatial_overlap_sweep(overlaps=None, cfg=DEFAULT, **kwargs):
    ref = build_reference(cfg)
    datasets = []
    for overlap in (ref.mode_overlaps if overlaps is None else overlaps):
        W, W_inv = tilt_slow(ref.system.W, overlap)
        datasets.append(make_dataset(replace(ref.system, W=W, W_inv=W_inv), partial(config_budget, cfg=cfg),
                                     study="spatial_overlap", param_name="mode overlap",
                                     param_value=overlap, **_dataset_kwargs(cfg, kwargs)))
    return datasets


def joint_sweep(snrs=None, taus_yr=None, cfg=DEFAULT, **kwargs):
    ref = build_reference(cfg)
    return [
        make_dataset(replace(ref.system, lam1=eigenvalue(12 * tau)),
                     partial(config_budget, cfg=cfg, scale=ref.snr / snr),
                     study="joint_snr_timescale", param_name="SNR", param_value=snr, **_dataset_kwargs(cfg, kwargs))
        for snr in (cfg.total_snrs if snrs is None else snrs)
        for tau in (cfg.slow_timescales_yr if taus_yr is None else taus_yr)
    ]


def study_levels(cfg=DEFAULT):
    """(study, level, one-dataset factory) for every level of every study under cfg."""
    ref = build_reference(cfg)
    levels = [("total_snr", v, partial(total_snr_sweep, snrs=[v], cfg=cfg)) for v in cfg.total_snrs]
    levels += [(f"partial_snr_{c}", v, partial(partial_snr_sweep, c, factors=[v], cfg=cfg))
               for c in PARTIAL_SNR_COMPONENTS for v in cfg.partial_snr_factors]
    levels += [(f"slow_timescale_{h}", v, partial(slow_timescale_sweep, taus_yr=[v], hold=h, cfg=cfg))
               for h in SLOW_TIMESCALE_HOLDS for v in cfg.slow_timescales_yr]
    levels += [("spatial_overlap", v, partial(spatial_overlap_sweep, overlaps=[v], cfg=cfg)) for v in ref.mode_overlaps]
    levels += [("joint_snr_timescale", (s, t), partial(joint_sweep, snrs=[s], taus_yr=[t], cfg=cfg))
               for s in cfg.total_snrs for t in cfg.slow_timescales_yr]
    return levels


def find_level(cfg, study, value):
    """The dataset factory for one study level; value is matched to within 1e-9 relative (a tuple for joint)."""
    levels = study_levels(cfg)
    studies = sorted({s for s, _, _ in levels})
    if study not in studies:
        raise KeyError(f"unknown study {study!r}; valid studies: {', '.join(studies)}")
    candidates = [(v, make) for s, v, make in levels if s == study]
    for v, make in candidates:
        if np.shape(v) == np.shape(value) and np.allclose(v, value, rtol=1e-9, atol=1e-12):
            return make
    raise KeyError(f"no level {value!r} in {study}; valid levels: {[v for v, _ in candidates]}")

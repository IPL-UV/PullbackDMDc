"""Synthetic datasets for the ablation studies: total SNR, partial SNR, slow timescale and spatial overlap.

Every tunable value lives in config.Config; the module-level constants below are the DEFAULT config's values,
kept so existing scripts can import them.
"""

import pathlib
import sys
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from typing import Tuple

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import DEFAULT
from data_preparation.interpolate_full_forcing import interpolate
from synthetic_system import make_W

M = 20
N = 1200
SPINUP_DECAY_TIMES = 40
SPINUP_CHUNK = 1200
MIN_NOISE_SPINUP = 1200
PARTIAL_SNR_COMPONENTS = dict(slow="s1_sq", pair="sp_sq", complement="sc_sq")
SLOW_TIMESCALE_HOLDS = ("snr", "modal_variance")

HISTORY = DEFAULT.history
RECORD_END_YEAR = DEFAULT.record_end_year
AR6_ERF_PATH = REPO_ROOT / "data_preparation" / "AR6_ERF_1750-2019.csv"
# F(t) = c + a exp((t - 2014) / efold) least-squares fitted to the annual AR6 CO2 ERF, 1750-2019 (values at
# mid-year); efold = 100 / k yr. RMSE 0.038 W m^-2 (0.056 over 1915-2014). Constant c in the past.
CO2_FIT = dict(c=0.019123072547470428, a=1.9152644964970247, k=1.6709321713250902)
CO2_FIT_REFERENCE_YEAR = 2014
FORCING_EFOLD_YR = DEFAULT.forcing_efold_yr
# exp + one positive and one negative Gaussian, least-squares fitted to the same AR6 CO2 values: RMSE 0.0092 W m^-2
# (0.0101 over 1915-2014), against 0.038 (0.055) for the plain exp; the Gaussians capture the bump near 1921 and the
# dip near 1966. Constant c in the past (|F - c| < 3e-4 of the rise before 1500).
CO2_GAUSS_FIT = dict(c=-0.005366940477657721, a=1.9642341976276356, k=1.5819078313533055,
                     A_pos=0.04816339012920922, mu_pos=1921.3285060598805, sigma_pos=17.8812620771011,
                     A_neg=0.13721889229450532, mu_neg=1966.0271283481388, sigma_neg=14.114214602524019)
FORCING_SOURCES = ("analytic", "analytic_gauss", "file")
GAUSS_FIELDS = ("gauss_efold_yr", "gauss_bump_amp", "gauss_bump_year", "gauss_bump_width_yr",
                "gauss_dip_amp", "gauss_dip_year", "gauss_dip_width_yr")
FILE_TREND_YEARS = 10  # after a forcing file's data ends, the forcing continues its trend over these last years
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
    record_end_year: int = RECORD_END_YEAR
    forcing_efold_yr: float = FORCING_EFOLD_YR
    forcing_source: str = DEFAULT.forcing_source
    forcing_file: str = DEFAULT.forcing_file
    forcing_column: str = DEFAULT.forcing_column
    gauss_efold_yr: float = DEFAULT.gauss_efold_yr
    gauss_bump_amp: float = DEFAULT.gauss_bump_amp
    gauss_bump_year: float = DEFAULT.gauss_bump_year
    gauss_bump_width_yr: float = DEFAULT.gauss_bump_width_yr
    gauss_dip_amp: float = DEFAULT.gauss_dip_amp
    gauss_dip_year: float = DEFAULT.gauss_dip_year
    gauss_dip_width_yr: float = DEFAULT.gauss_dip_width_yr

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


def resolve_forcing_path(path):
    path = pathlib.Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


@lru_cache(maxsize=None)
def load_forcing_file(path, column):
    """Monthly forcing from a CSV: (fractional years, values).

    An annual file (a `year` column, like AR6_ERF_1750-2019.csv) is interpolated to months as for the real data;
    a monthly file (a `time` column, YYYY-MM-01, like interpolatedAllForcing.csv) is used as is.
    """
    resolved = resolve_forcing_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"forcing file {path!r} not found (resolved to {resolved})")
    df = pd.read_csv(resolved)
    if column not in df.columns:
        available = [c for c in df.columns if c not in ("year", "time") and not c.startswith("Unnamed")]
        raise KeyError(f"forcing column {column!r} not in {resolved}; available: {', '.join(available)}")
    if "year" in df.columns:
        df = interpolate(df[["year", column]])
    elif "time" not in df.columns:
        raise ValueError(f"forcing file {resolved} needs a `year` (annual) or `time` (monthly, YYYY-MM-01) column")
    df = df.dropna(subset=[column])
    time = pd.to_datetime(df["time"])
    t = (time.dt.year + (time.dt.month - 1) / 12).to_numpy()
    order = np.argsort(t, kind="stable")
    return t[order], df[column].to_numpy(dtype=float)[order]


def file_forcing(t_yr, path, column):
    """File forcing at fractional years: held at its first value before the data, the last
    FILE_TREND_YEARS' linear trend after it."""
    t, values = load_forcing_file(path, column)
    slope = (values[-1] - np.interp(t[-1] - FILE_TREND_YEARS, t, values)) / FILE_TREND_YEARS
    t_yr = np.asarray(t_yr, dtype=float)
    return np.where(t_yr > t[-1], values[-1] + slope * (t_yr - t[-1]), np.interp(t_yr, t, values))


def co2_monthly():
    """AR6 CO2 ERF (W m^-2) interpolated to months as for the real data, for comparison with the model."""
    return load_forcing_file(str(AR6_ERF_PATH), "co2")


def co2_forcing_model(t_yr, efold_yr=FORCING_EFOLD_YR):
    """Analytic CO2 forcing (W m^-2) at fractional years: the AR6 fit c + a exp((t - 2014) / efold_yr)."""
    return CO2_FIT["c"] + CO2_FIT["a"] * np.exp((np.asarray(t_yr, dtype=float) - CO2_FIT_REFERENCE_YEAR) / efold_yr)


def gaussian(t, A, mu, sigma):
    return A * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def exp_gauss_model(t, c, a, k, A_pos, mu_pos, sigma_pos, A_neg, mu_neg, sigma_neg):
    """c + a exp(k (t - 2014) / 100) plus a positive and minus a negative Gaussian (A_pos, A_neg >= 0)."""
    t = np.asarray(t, dtype=float)
    return (c + a * np.exp(k * (t - CO2_FIT_REFERENCE_YEAR) / 100)
            + gaussian(t, A_pos, mu_pos, sigma_pos) - gaussian(t, A_neg, mu_neg, sigma_neg))


GAUSS_BUMP = (DEFAULT.gauss_bump_amp, DEFAULT.gauss_bump_year, DEFAULT.gauss_bump_width_yr)
GAUSS_DIP = (DEFAULT.gauss_dip_amp, DEFAULT.gauss_dip_year, DEFAULT.gauss_dip_width_yr)


def co2_forcing_gauss_model(t_yr, efold_yr=DEFAULT.gauss_efold_yr, bump=GAUSS_BUMP, dip=GAUSS_DIP):
    """Analytic CO2 forcing (W m^-2) at fractional years: c + a exp((t - 2014) / efold_yr) + bump - dip, with c, a
    from the AR6 fit and each Gaussian given as (amplitude, year, width); the defaults reproduce the fit."""
    t = np.asarray(t_yr, dtype=float)
    return (CO2_GAUSS_FIT["c"] + CO2_GAUSS_FIT["a"] * np.exp((t - CO2_FIT_REFERENCE_YEAR) / efold_yr)
            + gaussian(t, *bump) - gaussian(t, *dip))


def gauss_params(system):
    """(efold_yr, bump, dip) of a system's analytic_gauss forcing."""
    return (system.gauss_efold_yr, (system.gauss_bump_amp, system.gauss_bump_year, system.gauss_bump_width_yr),
            (system.gauss_dip_amp, system.gauss_dip_year, system.gauss_dip_width_yr))


def record_years(system):
    """Fractional years of record months 0..N-1 (the record ends in December of record_end_year)."""
    return system.record_end_year - N // 12 + 1 + np.arange(N) / 12


def forcing_values(system, t_yr):
    """The system's forcing F (W m^-2) at fractional years, from its forcing_source."""
    if system.forcing_source == "analytic":
        return co2_forcing_model(t_yr, system.forcing_efold_yr)
    if system.forcing_source == "analytic_gauss":
        return co2_forcing_gauss_model(t_yr, *gauss_params(system))
    if system.forcing_source == "file":
        return file_forcing(t_yr, system.forcing_file, system.forcing_column)
    raise ValueError(f"unknown forcing_source {system.forcing_source!r}; valid: {', '.join(FORCING_SOURCES)}")


def forcing_series(system):
    F = forcing_values(system, record_years(system)[0] + np.arange(-system.spinup, N) / 12)
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
    if cfg.forcing_source not in FORCING_SOURCES:
        raise ValueError(f"unknown forcing_source {cfg.forcing_source!r}; valid: {', '.join(FORCING_SOURCES)}")
    if cfg.forcing_source == "analytic_gauss":
        bad = [f"{k}={getattr(cfg, k)!r}" for k in ("gauss_bump_amp", "gauss_dip_amp") if getattr(cfg, k) < 0]
        bad += [f"{k}={getattr(cfg, k)!r}" for k in ("gauss_efold_yr", "gauss_bump_width_yr", "gauss_dip_width_yr")
                if getattr(cfg, k) <= 0]
        if bad:
            raise ValueError(f"analytic_gauss needs amplitudes >= 0 and widths, efold > 0; got {', '.join(bad)}")
    W, W_inv, b, phi = make_W(M, cfg.pattern_seed)
    unit = System(
        W=W, W_inv=W_inv, b=b,
        lam1=eigenvalue(12 * cfg.tau1_yr), rho=eigenvalue(12 * cfg.tau_p_yr), theta=2 * np.pi / (12 * cfg.period_p_yr),
        lam_c=np.random.default_rng(cfg.eigenvalue_seed).uniform(*cfg.complement_eig_range, size=M - 3),
        history=cfg.history, record_end_year=cfg.record_end_year, forcing_efold_yr=cfg.forcing_efold_yr,
        forcing_source=cfg.forcing_source, forcing_file=cfg.forcing_file, forcing_column=cfg.forcing_column,
        **{f: getattr(cfg, f) for f in GAUSS_FIELDS},
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

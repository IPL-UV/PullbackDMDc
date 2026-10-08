"""Synthetic datasets for the ablation studies: total SNR, partial SNR, slow timescale and spatial overlap.

Every tunable value lives in config.Config; the module-level constants below are the DEFAULT config's values.
"""

import pathlib
import sys
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache, partial
from typing import Callable, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import DEFAULT
from data_preparation.interpolate_full_forcing import interpolate
from system_patterns import make_W

M = 20
N = 1980  # record length in months: Jan 1850 - Dec 2014, as in the real-data experiments
SPINUP_DECAY_TIMES = 40
SPINUP_CHUNK = 1200
MIN_SPINUP = 1200  # the floor on the derived spin-up: a century, the shortest window worth giving a method
PARTIAL_SNR_COMPONENTS = dict(slow="s1_sq", pair="sp_sq", complement="sc_sq")
SLOW_TIMESCALE_HOLDS = ("snr", "modal_variance")

RECORD_END_YEAR = DEFAULT.record_end_year
AR6_ERF_PATH = REPO_ROOT / "data_preparation" / "AR6_ERF_1750-2019.csv"
# Least-squares fits to the annual AR6 CO2 ERF, 1750-2019; see README.md for their quality and for how
# the config defaults round them. The forcing is always centered on the record (centered_forcing).
CO2_FIT = dict(c=0.019123072547470428, a=1.9152644964970247, k=1.6709321713250902)
CO2_FIT_REFERENCE_YEAR = 2014
FORCING_EFOLD_YR = DEFAULT.forcing_efold_yr
CO2_GAUSS_FIT = dict(c=-0.005366940477657721, a=1.9642341976276356, k=1.5819078313533055,
                     A_pos=0.04816339012920922, mu_pos=1921.3285060598805, sigma_pos=17.8812620771011,
                     A_neg=0.13721889229450532, mu_neg=1966.0271283481388, sigma_neg=14.114214602524019)
FORCING_SOURCES = ("analytic", "analytic_gauss", "file")
GAUSS_FIELDS = ("gauss_efold_yr", "gauss_bump_amp", "gauss_bump_year", "gauss_bump_width_yr",
                "gauss_dip_amp", "gauss_dip_year", "gauss_dip_width_yr")
FILE_TREND_YEARS = 10  # after a forcing file's data ends, the forcing continues its trend over these last years
N_REALIZATIONS = DEFAULT.n_realizations
PERIOD_P_YR = DEFAULT.period_p_yr
TOTAL_SNRS = list(DEFAULT.total_snrs)
PARTIAL_SNR_FACTORS = list(DEFAULT.partial_snr_factors)
SLOW_TIMESCALES_YR = list(DEFAULT.slow_timescales_yr)


def eigenvalue(tau_months):
    return np.exp(-1 / tau_months)


def decay_time(eig):
    return -1 / np.log(np.abs(eig))


def eigenvalue_yr(tau_yr):
    return eigenvalue(12 * tau_yr)


def decay_time_yr(eig):
    return decay_time(eig) / 12


@dataclass(frozen=True, eq=False)
class System:
    W: np.ndarray
    W_inv: np.ndarray
    b: np.ndarray
    lam1: float
    rho: float
    theta: float
    lam_c: np.ndarray
    s1_sq: float = np.nan
    sp_sq: float = np.nan
    sc_sq: float = np.nan
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
    def spinup(self):
        """Months before the record: where the truth is generated from, and the forcing window the methods get.

        `SPINUP_DECAY_TIMES` e-foldings of the longest-lived mode, floored at MIN_SPINUP. The max over all
        eigenvalues keeps the oscillating pair covered when it outlives the slow mode, at tau_1 = 1 yr.
        Derived rather than fixed so a tau_1 sweep gives every level a window matched to its own memory:
        a flat 1200 months is 100 e-foldings at tau_1 = 1 yr but a single one at 100 yr, which leaves the
        reconstruction short of its own equilibrium offset.
        """
        return max(int(np.ceil(SPINUP_DECAY_TIMES * decay_time(self.eigvals).max())), MIN_SPINUP)

    @cached_property
    def y0(self):
        """The forcing at the first month of the spin-up, which B is normalized against."""
        return centered_forcing(self, record_years(self)[0] - self.spinup / 12)

    @cached_property
    def b_scale(self):
        """||y0 (I - A)^-1 b||_F: the norm of the quasi-equilibrium state the system starts from."""
        return abs(self.y0) * np.linalg.norm(np.linalg.solve(np.eye(M) - self.A, self.b))

    @cached_property
    def B(self):
        """The unit pattern b scaled so ||y0 (I - A)^-1 B||_F = 1; derived, so every `replace` recomputes it."""
        return self.b / self.b_scale


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


def annual_years(system):
    """Mid-year fractional years of the N // 12 annual means of the record."""
    return record_years(system)[::12] + 0.5


def check_forcing_source(source):
    if source not in FORCING_SOURCES:
        raise ValueError(f"unknown forcing_source {source!r}; valid: {', '.join(FORCING_SOURCES)}")


def forcing_values(system, t_yr):
    """The system's forcing F (W m^-2) at fractional years, from its forcing_source."""
    check_forcing_source(system.forcing_source)
    if system.forcing_source == "analytic":
        return co2_forcing_model(t_yr, system.forcing_efold_yr)
    if system.forcing_source == "analytic_gauss":
        return co2_forcing_gauss_model(t_yr, *gauss_params(system))
    return file_forcing(t_yr, system.forcing_file, system.forcing_column)


def centered_forcing(system, t_yr):
    """The system's forcing (W m^-2) at fractional years minus its mean over the observed interval (the record).

    The single definition of the centering: data generation, the forcing given to the methods and every plot
    use it, as the real-world experiments center the forcing on the training record (history included).
    """
    return forcing_values(system, t_yr) - forcing_values(system, record_years(system)).mean()


def spinup_years(system):
    """Fractional years of the spin-up and record months -spinup..N-1."""
    return record_years(system)[0] + np.arange(-system.spinup, N) / 12


def forcing_series(system):
    """Forcing over spin-up + record, centered on the record (the same offset is kept during spin-up)."""
    return centered_forcing(system, spinup_years(system))


def run_modal(Lambda, drive, z0=None):
    z = np.zeros(drive.shape[1:]) if z0 is None else z0
    out = np.empty_like(drive)
    for t in range(len(drive)):
        z = z @ Lambda.T + drive[t]
        out[t] = z
    return out


def drive_modal(system, y, z0=None):
    """Modal coordinates (len(y), M) of the system driven by the scalar forcing y; project with `@ system.W.T`."""
    return run_modal(system.Lambda_R, np.outer(y, system.W_inv @ system.B), z0)


def forced_response(system):
    y = forcing_series(system)
    return y, drive_modal(system, y)[system.spinup:] @ system.W.T


def internal_variability(system, n_realizations, seed):
    # separate streams so the record noise is shared by systems with different spin-ups
    spinup_rng, record_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    noise_std = np.sqrt(system.noise_variances)
    z = np.zeros((n_realizations, M))
    for start in range(0, system.spinup, SPINUP_CHUNK):
        steps = min(SPINUP_CHUNK, system.spinup - start)
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


def forcing_overlap(W, b):
    """cos angle(w_1, b-hat): how far the slow mode's pattern points along the forcing's."""
    return abs(W[:, 0] @ b) / np.linalg.norm(W[:, 0])


def rotate_slow_to_b(W, b, overlap):
    """Rotate w_1 to cos angle(w_1, b) = overlap, within span(w_1, b); no other mode moves.

    R is the Givens rotation of that plane and the identity on its complement, so w_1 travels along the
    unit sphere's geodesic toward b. Unlike tilt_slow's w_1, this one leaves span(S), so Q_perp.T w_1 is
    no longer 0 and [pinv(S); Q_perp.T] is no longer the inverse: W_inv is the exact inverse instead.

    No angle is ever formed. arccos loses precision like 1 / sqrt(1 - x^2), so delta = angle(w_1, b) -
    arccos(overlap) would be noisy exactly where the sweep is most interesting -- w_1 near b, or a level
    near +-1. The subtraction identities need only the cosine and sine of each angle, and both are
    available without cancellation: the first angle's sine is the norm of the perpendicular component,
    and 1 - overlap^2 is factored as (1 - overlap)(1 + overlap).
    """
    if not -1 <= overlap <= 1:
        raise ValueError(f"forcing overlap must be a cosine in [-1, 1]; got {overlap!r}")
    u = W[:, 0] / np.linalg.norm(W[:, 0])
    perpendicular = b - (b @ u) * u  # b is already unit norm
    cos_a, sin_a = u @ b, np.linalg.norm(perpendicular)  # sin_a >= 0, and is a norm, so no cancellation
    if sin_a < 1e-12:
        raise ValueError("w_1 is already parallel to b: the rotation plane is undefined")
    e2 = perpendicular / sin_a
    cos_c, sin_c = overlap, np.sqrt((1 - overlap) * (1 + overlap))
    cos_d = cos_a * cos_c + sin_a * sin_c  # cos(alpha - beta)
    sin_d = sin_a * cos_c - cos_a * sin_c  # sin(alpha - beta)
    R = (np.eye(M)
         + (cos_d - 1) * (np.outer(u, u) + np.outer(e2, e2))
         + sin_d * (np.outer(e2, u) - np.outer(u, e2)))
    W = W.copy()
    W[:, 0] = R @ W[:, 0]
    return W, np.linalg.inv(W)


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

    def forcings(self):
        """(long, short, transition_time): the whole forcing series, its record tail, and the spin-up between."""
        long_forcings = self.y[:, None]
        return long_forcings, long_forcings[self.system.spinup:], self.system.spinup


def make_dataset(system, budget, *, study="", param_name="", param_value=np.nan,
                 n_realizations=N_REALIZATIONS, seed=0):
    y, forced = forced_response(system)
    record = y[system.spinup:]
    assert abs(record.mean()) <= 1e-12 * max(np.ptp(record), 1e-300), "forcing must be centered on the record"
    system = replace(system, **budget(forced_variance(forced)))
    return SyntheticDataset(
        study=study, param_name=param_name, param_value=float(param_value), system=system,
        y=y, forced=forced, internal=internal_variability(system, n_realizations, seed),
    )


def empirical_snr(ds):
    return ds.V_f / ds.internal.var(axis=1).sum(axis=-1)


def modal_coordinates(ds, x):
    return x @ ds.system.W_inv.T


def global_mean(x):
    """Mean over the last (latitude) axis of an (..., M) field; unweighted, to match the patterns."""
    return x.mean(axis=-1)


@dataclass(frozen=True, eq=False)
class Reference:
    system: System  # the system the sweeps build on; b is the raw unit pattern, B is b scaled (see System.B)
    budget: dict
    snr: float
    base_overlap: float
    mode_overlaps: Tuple[float, ...]
    base_b_overlap: float
    b_overlaps: Tuple[float, ...]
    phi: np.ndarray


@lru_cache(maxsize=None)
def build_reference(cfg=DEFAULT):
    check_forcing_source(cfg.forcing_source)
    if cfg.forcing_source == "analytic_gauss":
        bad = [f"{k}={getattr(cfg, k)!r}" for k in ("gauss_bump_amp", "gauss_dip_amp") if getattr(cfg, k) < 0]
        bad += [f"{k}={getattr(cfg, k)!r}" for k in ("gauss_efold_yr", "gauss_bump_width_yr", "gauss_dip_width_yr")
                if getattr(cfg, k) <= 0]
        if bad:
            raise ValueError(f"analytic_gauss needs amplitudes >= 0 and widths, efold > 0; got {', '.join(bad)}")
    W, W_inv, b, phi = make_W(M, cfg.pattern_seed)
    system = System(
        W=W, W_inv=W_inv, b=b,
        lam1=eigenvalue_yr(cfg.tau1_yr), rho=eigenvalue_yr(cfg.tau_p_yr), theta=2 * np.pi / (12 * cfg.period_p_yr),
        lam_c=np.random.default_rng(cfg.eigenvalue_seed).uniform(*cfg.complement_eig_range, size=M - 3),
        record_end_year=cfg.record_end_year, forcing_efold_yr=cfg.forcing_efold_yr,
        forcing_source=cfg.forcing_source, forcing_file=cfg.forcing_file, forcing_column=cfg.forcing_column,
        **{f: getattr(cfg, f) for f in GAUSS_FIELDS},
    )
    # No amplitude calibration: the scale is set by B (System.b_scale). The budget follows the reference's
    # actual V_f, which is no longer a config target, so the reference SNR is cfg.variances' shares.
    reference_V_f = forced_variance(forced_response(system)[1])
    budget = config_budget(reference_V_f, cfg)
    base_overlap = pair_plane_overlap(W)
    overlaps = cfg.mode_overlaps if cfg.mode_overlaps is not None else (0.0, 0.25, base_overlap, 0.75, 0.9, 0.95)
    base_b_overlap = forcing_overlap(W, b)
    b_overlaps = cfg.b_overlaps if cfg.b_overlaps is not None else (0.0, 0.25, 0.5, base_b_overlap, 0.8, 0.9, 1.0)
    return Reference(system=system, budget=budget, snr=theoretical_snr(reference_V_f, **budget),
                     base_overlap=base_overlap, mode_overlaps=tuple(overlaps),
                     base_b_overlap=base_b_overlap, b_overlaps=tuple(b_overlaps), phi=phi)


_REF = build_reference(DEFAULT)
# B_HAT is the RAW unit-norm forcing pattern, not the scaled B (see System.B).
W_REF, W_INV_REF, B_HAT, PHI = _REF.system.W, _REF.system.W_inv, _REF.system.b, _REF.phi
REFERENCE = _REF.system
REFERENCE_BUDGET = dict(_REF.budget)
BASE_OVERLAP = _REF.base_overlap
MODE_OVERLAPS = list(_REF.mode_overlaps)
BASE_B_OVERLAP = _REF.base_b_overlap
B_OVERLAPS = list(_REF.b_overlaps)


# One row per study: how its levels are enumerated, how each level bends the reference system, and what
# budget it gets. study_levels and the five sweep functions below are both views of this table, so the
# study matrix is described exactly once.
@dataclass(frozen=True)
class Study:
    param_name: str
    levels: Callable  # (ref, cfg) -> the study's levels
    system: Callable = lambda ref, level: ref.system
    budget: Callable = lambda ref, cfg, level: partial(config_budget, cfg=cfg)
    param_value: Callable = lambda level: level


def _tilted(ref, overlap):
    W, W_inv = tilt_slow(ref.system.W, overlap)
    return replace(ref.system, W=W, W_inv=W_inv)


def _rotated_to_b(ref, overlap):
    W, W_inv = rotate_slow_to_b(ref.system.W, ref.system.b, overlap)
    return replace(ref.system, W=W, W_inv=W_inv)


STUDIES = {
    "total_snr": Study(
        param_name="SNR",
        levels=lambda ref, cfg: list(cfg.total_snrs),
        budget=lambda ref, cfg, snr: partial(config_budget, cfg=cfg, scale=ref.snr / snr),
    ),
    **{f"partial_snr_{component}": Study(
        param_name=f"{component} variance factor",
        levels=lambda ref, cfg: list(cfg.partial_snr_factors),
        budget=lambda ref, cfg, factor, key=key: partial(config_budget, cfg=cfg, component=key, factor=factor),
    ) for component, key in PARTIAL_SNR_COMPONENTS.items()},
    **{f"slow_timescale_{hold}": Study(
        param_name=r"$\tau_1$ (yr)",
        levels=lambda ref, cfg: list(cfg.slow_timescales_yr),
        system=lambda ref, tau: replace(ref.system, lam1=eigenvalue_yr(tau)),
        # "snr" rescales the budget to each level's own V_f, holding SNR fixed; "modal_variance" pins the
        # budget to the reference, so SNR follows V_f.
        budget=(lambda ref, cfg, tau: partial(config_budget, cfg=cfg)) if hold == "snr"
        else (lambda ref, cfg, tau: lambda V_f: dict(ref.budget)),
    ) for hold in SLOW_TIMESCALE_HOLDS},
    "spatial_overlap": Study(
        param_name="mode overlap",
        levels=lambda ref, cfg: list(ref.mode_overlaps),
        system=_tilted,
    ),
    "forcing_overlap": Study(
        param_name=r"$\cos\angle(w_1,\hat b)$",
        levels=lambda ref, cfg: list(ref.b_overlaps),
        system=_rotated_to_b,
    ),
    "joint_snr_timescale": Study(
        param_name="SNR",
        levels=lambda ref, cfg: [(snr, tau) for snr in cfg.total_snrs for tau in cfg.slow_timescales_yr],
        system=lambda ref, level: replace(ref.system, lam1=eigenvalue_yr(level[1])),
        budget=lambda ref, cfg, level: partial(config_budget, cfg=cfg, scale=ref.snr / level[0]),
        param_value=lambda level: level[0],
    ),
}


def sweep(study, levels=None, cfg=DEFAULT, **kwargs):
    """Datasets for the given levels of `study` (all of them when levels is None)."""
    if study not in STUDIES:
        raise KeyError(f"unknown study {study!r}; valid studies: {', '.join(sorted(STUDIES))}")
    spec = STUDIES[study]
    ref = build_reference(cfg)
    return [
        make_dataset(spec.system(ref, level), spec.budget(ref, cfg, level), study=study,
                     param_name=spec.param_name, param_value=spec.param_value(level),
                     **{"n_realizations": cfg.n_realizations, "seed": cfg.noise_seed, **kwargs})
        for level in (spec.levels(ref, cfg) if levels is None else levels)
    ]


def total_snr_sweep(snrs=None, cfg=DEFAULT, **kwargs):
    return sweep("total_snr", snrs, cfg, **kwargs)


def partial_snr_sweep(component, factors=None, cfg=DEFAULT, **kwargs):
    return sweep(f"partial_snr_{component}", factors, cfg, **kwargs)


def slow_timescale_sweep(taus_yr=None, hold="snr", cfg=DEFAULT, **kwargs):
    return sweep(f"slow_timescale_{hold}", taus_yr, cfg, **kwargs)


def spatial_overlap_sweep(overlaps=None, cfg=DEFAULT, **kwargs):
    return sweep("spatial_overlap", overlaps, cfg, **kwargs)


def forcing_overlap_sweep(overlaps=None, cfg=DEFAULT, **kwargs):
    return sweep("forcing_overlap", overlaps, cfg, **kwargs)


def joint_sweep(snrs=None, taus_yr=None, cfg=DEFAULT, **kwargs):
    levels = None if snrs is None and taus_yr is None else [
        (snr, tau) for snr in (cfg.total_snrs if snrs is None else snrs)
        for tau in (cfg.slow_timescales_yr if taus_yr is None else taus_yr)
    ]
    return sweep("joint_snr_timescale", levels, cfg, **kwargs)


def study_levels(cfg=DEFAULT):
    """(study, level, one-dataset factory) for every level of every study under cfg."""
    ref = build_reference(cfg)
    return [(study, level, partial(sweep, study, [level], cfg))
            for study, spec in STUDIES.items() for level in spec.levels(ref, cfg)]


def find_level(cfg, study, value):
    """The dataset factory for one study level; value is matched to within 1e-9 relative (a tuple for joint)."""
    levels = study_levels(cfg)
    if study not in STUDIES:
        raise KeyError(f"unknown study {study!r}; valid studies: {', '.join(sorted(STUDIES))}")
    candidates = [(level, factory) for s, level, factory in levels if s == study]
    for level, factory in candidates:
        if np.shape(level) == np.shape(value) and np.allclose(level, value, rtol=1e-9, atol=1e-12):
            return factory
    raise KeyError(f"no level {value!r} in {study}; valid levels: {[level for level, _ in candidates]}")

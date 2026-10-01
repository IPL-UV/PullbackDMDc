"""Synthetic datasets for the ablation studies: total SNR, partial SNR, slow timescale and spatial overlap."""

from dataclasses import dataclass, replace
from functools import partial

import numpy as np

from synthetic_system import make_W

M = 20
N = 1200
HISTORY = 1200
N_REALIZATIONS = 100
PATTERN_SEED = 22
EIGENVALUE_SEED = 20
COMPLEMENT_EIG_RANGE = (0.0, 0.1)
TAU1_YR = 20
TAU_P_YR = 2
PERIOD_P_YR = 4
SPINUP_DECAY_TIMES = 40
SPINUP_CHUNK = 1200

TOTAL_SNRS = [1 / 30, 1 / 10, 1 / 3, 1, 3, 10, 30]
PARTIAL_SNR_FACTORS = [1 / 4, 1 / 2, 1, 2, 4]
PARTIAL_SNR_COMPONENTS = dict(slow="s1_sq", pair="sp_sq", complement="sc_sq")
SLOW_TIMESCALES_YR = [1, 2, 5, 10, 20, 50, 100]
SLOW_TIMESCALE_HOLDS = ("snr", "modal_variance")


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
        return max(int(np.ceil(SPINUP_DECAY_TIMES * decay_time(self.eigvals).max())), HISTORY)


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


def equal_budget(V_f):
    return dict(s1_sq=V_f, sp_sq=V_f / 2, sc_sq=V_f / (M - 3))


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

    def forcings(self, history=HISTORY):
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


W_REF, W_INV_REF, B_HAT, PHI = make_W(M, PATTERN_SEED)
UNIT_FORCING_REFERENCE = System(
    W=W_REF, W_inv=W_INV_REF, b=B_HAT,
    lam1=eigenvalue(12 * TAU1_YR), rho=eigenvalue(12 * TAU_P_YR), theta=2 * np.pi / (12 * PERIOD_P_YR),
    lam_c=np.random.default_rng(EIGENVALUE_SEED).uniform(*COMPLEMENT_EIG_RANGE, size=M - 3),
)
FORCING_AMPLITUDE = np.sqrt(M / forced_variance(forced_response(UNIT_FORCING_REFERENCE)[1]))
REFERENCE = replace(UNIT_FORCING_REFERENCE, forcing_amplitude=FORCING_AMPLITUDE)
REFERENCE_BUDGET = equal_budget(M)
REFERENCE_SNR = theoretical_snr(M, **REFERENCE_BUDGET)
BASE_OVERLAP = pair_plane_overlap(W_REF)
MODE_OVERLAPS = [0.0, 0.25, BASE_OVERLAP, 0.75, 0.9, 0.95]


def scaled_budget(V_f, factor):
    return {k: v * factor for k, v in equal_budget(V_f).items()}


def component_budget(V_f, key, factor):
    budget = equal_budget(V_f)
    budget[key] *= factor
    return budget


def total_snr_sweep(snrs=TOTAL_SNRS, **kwargs):
    return [
        make_dataset(REFERENCE, partial(scaled_budget, factor=REFERENCE_SNR / snr),
                     study="total_snr", param_name="SNR", param_value=snr, **kwargs)
        for snr in snrs
    ]


def partial_snr_sweep(component, factors=PARTIAL_SNR_FACTORS, **kwargs):
    key = PARTIAL_SNR_COMPONENTS[component]
    return [
        make_dataset(REFERENCE, partial(component_budget, key=key, factor=factor),
                     study=f"partial_snr_{component}", param_name=f"{component} variance factor",
                     param_value=factor, **kwargs)
        for factor in factors
    ]


def slow_timescale_sweep(taus_yr=SLOW_TIMESCALES_YR, hold="snr", **kwargs):
    if hold not in SLOW_TIMESCALE_HOLDS:
        raise ValueError(f"hold must be one of {SLOW_TIMESCALE_HOLDS}")
    budget = equal_budget if hold == "snr" else (lambda V_f: dict(REFERENCE_BUDGET))
    return [
        make_dataset(replace(REFERENCE, lam1=eigenvalue(12 * tau)), budget,
                     study=f"slow_timescale_{hold}", param_name=r"$\tau_1$ (yr)", param_value=tau, **kwargs)
        for tau in taus_yr
    ]


def spatial_overlap_sweep(overlaps=MODE_OVERLAPS, **kwargs):
    datasets = []
    for overlap in overlaps:
        W, W_inv = tilt_slow(REFERENCE.W, overlap)
        datasets.append(make_dataset(replace(REFERENCE, W=W, W_inv=W_inv), equal_budget,
                                     study="spatial_overlap", param_name="mode overlap",
                                     param_value=overlap, **kwargs))
    return datasets

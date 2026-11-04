"""ACFs and timescales shared by the evaluation scripts."""
import numpy as np


def acf(x, axis=0, length=60):
    """Sample ACF along axis: mean removed, lag-k sum divided by (N - k), normalised by the lag-0 variance."""
    x = x - x.mean(axis=axis, keepdims=True)
    N = x.shape[axis]
    def acf_1d(z):
        gamma0 = np.sum(z * z) / N
        gamma0 = gamma0 if gamma0 > 1e-12 else 1e-12
        return np.array([(np.sum(z[:N-k] * z[k:]) / (N - k)) / gamma0 for k in range(length)])
    return np.apply_along_axis(acf_1d, axis, x)


def acf_bands(acf_values, N, alpha=0.05):
    """White-noise null: 95% band at lag k is +/- 1.96 / sqrt(N - k). alpha is unused (kept for callers)."""
    z_score = 1.96
    acf_values = np.asarray(acf_values, dtype=float).reshape(-1)
    se = np.zeros_like(acf_values, dtype=float)
    max_lag = min(acf_values.size, N)
    for k in range(max_lag):
        se[k] = 1.0 / np.sqrt(N - k)
    band = z_score * se
    return -band, band


def first_crossing_time(acf_values, dt, level):
    """First lag (in units of dt) where the ACF drops to <= level, linearly interpolated. NaN if never."""
    acf_values = np.asarray(acf_values, dtype=float)
    for idx in range(1, acf_values.size):
        if acf_values[idx] <= level:
            prev_v, next_v = acf_values[idx - 1], acf_values[idx]
            if np.isclose(prev_v, next_v):
                return idx * dt
            w = np.clip((level - prev_v) / (next_v - prev_v), 0.0, 1.0)
            return (idx - 1 + w) * dt
    return np.nan


def efolding_time(acf_values, dt):
    return first_crossing_time(acf_values, dt, 1.0 / np.e)


def zero_crossing_time(acf_values, dt):
    return first_crossing_time(acf_values, dt, 0.0)


def eig_decay_time_years(eigvals, lag_months):
    return -lag_months / (12.0 * np.log(np.abs(eigvals)))


def eig_frequency_per_year(eigvals, lag_months):
    """Oscillation frequency in cycles per year of a discrete-time eigenvalue with step lag_months."""
    omegas = np.angle(eigvals) / lag_months
    return (6 * np.abs(omegas)) / np.pi


def zero_crossing_summary(member_acfs, top_n, dt_years):
    """Per member, an array of 4 x first zero crossing for the first top_n mode ACFs (NaN if none)."""
    summaries = []
    for member_acf in member_acfs:
        mode_vals = []
        for mode_idx in range(top_n):
            crossing = zero_crossing_time(member_acf[mode_idx], dt_years)
            mode_vals.append(4.0 * crossing if np.isfinite(crossing) else np.nan)
        summaries.append(np.asarray(mode_vals, dtype=float))
    return summaries

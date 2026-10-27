import numpy as np


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

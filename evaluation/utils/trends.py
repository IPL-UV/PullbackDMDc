import os

import numpy as np


def parse_window(raw):
    """'YYYY-YYYY' -> (start_year, end_year), inclusive."""
    start, end = (int(y) for y in raw.strip().split('-'))
    if start > end:
        raise ValueError(f'Bad trend window: {raw!r}')
    return start, end


def trend_window(default='1985-2014'):
    return parse_window(os.getenv('PBDMDC_TREND_WINDOW', default))


def window_mask(years, window):
    """Boolean mask over time steps with Jan start_year <= t <= Dec end_year."""
    years = np.asarray(years)
    return (years >= window[0]) & (years <= window[1])


def window_label(window):
    return f'{window[0]}-{window[1]}'


def compute_linear_trend(field_2d: np.ndarray) -> np.ndarray:
    """Compute linear trend at each spatial point for data shaped (time, space)."""
    time_index = np.arange(field_2d.shape[0], dtype=np.float64)
    return np.polyfit(time_index, field_2d, 1)[0]

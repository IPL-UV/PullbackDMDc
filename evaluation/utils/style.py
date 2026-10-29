"""Display names, colors and color scaling shared by the evaluation figures."""
import numpy as np

CLIM_VARS_ON_FIG = {'tas_ocean': 'OSAT', 'psl': 'SLP', 'tas': 'SAT', 'tas_masked': 'SAT-masked'}

METHOD_NAMES_ON_FIG = {'RegGMST': 'RegGMST', 'LR': 'Linear reg.', 'LIM': 'LIM', 'LIM-opt': 'LIM-opt',
                       'PullbackDMDc': 'PullbackDMDc', 'PullbackDMDc-2d': 'PullbackDMDc(2d)',
                       'PullbackDMDc-3d': 'PullbackDMDc(3d)'}

MODEL_DISPLAY_NAMES = {'MPI-ESM1-2-LR': 'MPI-ESM'}

ESM_COLORS = {
    '20CRv3': 'black',
    'CESM2': '#7f7f7f',
    'MPI-ESM1-2-LR': '#8c564b',
    'CanESM5': '#2ca02c',
    'MIROC6': '#0b1f5b',
}

# Distinct linestyles keep the datasets separable in grayscale prints.
ESM_LINESTYLES = {
    '20CRv3': '-',
    'CESM2': '--',
    'MPI-ESM1-2-LR': '-.',
    'CanESM5': ':',
    'MIROC6': (0, (3, 1, 1, 1)),
}

# Period bands (years) shaded behind timescale axes.
TIMESCALE_BANDS = {
    'ENSO': {'xlim': (2.0, 7.0), 'color': 'tab:cyan', 'alpha': 0.12},
    'PDO': {'xlim': (10.0, 30.0), 'color': 'tab:purple', 'alpha': 0.10},
}


def display_model_name(model_name):
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def p95_abs(maps):
    """95th percentile of |map| pooled over a list of maps (NaNs ignored), floored at 1e-12."""
    return max(np.nanpercentile(np.abs(np.concatenate([m.ravel() for m in maps])), 95), 1e-12)

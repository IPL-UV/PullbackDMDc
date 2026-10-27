"""The ensemble member shown per ESM in the mode figures, and helpers to find it in multi-member plots."""
import os
import pickle
import re
import warnings

import numpy as np
from matplotlib.lines import Line2D

# Single source for the member shown in plot_modes.py / plot_psd_mtm.py (20CRv3 has one realization).
SELECTED_MEMBERS = {
    '20CRv3': '',
    'MPI-ESM1-2-LR': 'r47i1p1f1',
    'CESM2': '1281.005i1p1f1',
    'CanESM5': 'r16i1p1f1',
    'MIROC6': 'r35i1p1f1',
}

SELECTED_HANDLE = Line2D([], [], color='black', linewidth=1.3, label='Member shown in mode figures')

ACF_LENGTH = 100 * 12  # as in compute_acfs.py


def member_id(f_name):
    """Member ID from a file name, e.g. 'r47i1p1f1' or CESM2's '1281.005i1p1f1'; '' if none."""
    m = re.search(r'r\d+i\d+p\d+f\d+', f_name) or re.search(r'\d+\.\d+i\d+p\d+f\d+', f_name)
    return m.group(0) if m else ''


def method_files(model_dir, method, sort):
    """File names of one method's pickles, in sorted or raw os.listdir order (match the caller's order)."""
    files = [f for f in os.listdir(model_dir) if f.split('_')[-1].split('.')[0] == method]
    return sorted(files) if sort else files


def selected_index(file_names, model):
    """Index of the selected member in file_names; None for datasets without a selected member."""
    target = SELECTED_MEMBERS.get(model, '')
    if not target:
        return None
    hits = [i for i, f in enumerate(file_names) if member_id(f) == target]
    if len(hits) != 1:
        raise ValueError(f'{model}: expected one file for member {target!r}, found {len(hits)}')
    return hits[0]


def _acf(x, length):
    """Same ACF as compute_acfs.py (biased-by-lag normalisation)."""
    x = x - x.mean()
    n = x.size
    gamma0 = max(np.sum(x * x) / n, 1e-12)
    return np.array([(np.sum(x[:n - k] * x[k:]) / (n - k)) / gamma0 for k in range(length)])


def acf_selected_index(model_dir, method, model, acf_rows, top_n, n_check=4):
    """Index of the selected member in an acfs_{var}_{method}.pkl member list.

    compute_acfs.py stores members in raw os.listdir order without names, so replay that order and
    confirm by recomputing the first n_check mode ACFs of the selected member.
    """
    files = method_files(model_dir, method, sort=False)
    idx = selected_index(files, model)
    if idx is None:
        return None
    if len(files) != len(acf_rows):
        raise ValueError(f'{model}: {len(files)} {method} files but {len(acf_rows)} ACF rows')
    with open(os.path.join(model_dir, files[idx]), 'rb') as f:
        dmd = pickle.load(f)
    dmd.compute_modes()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='top_n cuts off a complex-conjugate pair')
        ts = dmd.compute_rotated_modes(top_n=top_n)['internal_time_series']
    stored = np.asarray(acf_rows[idx])
    for k in range(min(n_check, ts.shape[1])):
        if not np.allclose(_acf(ts[:, k], ACF_LENGTH), stored[k], atol=1e-10):
            raise ValueError(f'{model}: ACF row {idx} does not match {files[idx]} (mode {k + 1})')
    print(f'  {model}: selected member {files[idx]} = ACF row {idx} (checked)', flush=True)
    return idx


def mark_selected(ax, x, y, half_height):
    """Short black vertical line at the selected member's value."""
    ax.vlines(x, y - half_height, y + half_height, color='black', linewidth=1.3, zorder=5)

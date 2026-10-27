"""R2-M2 (i): forced-response estimates for the 20CRv3 reanalysis.

One row per GM variable, plus one row per regional pickle (R2-L374) found. The grey curves are the
reanalysis itself (forced + internal variability), not a truth.
Run: PBDMDC_LAGS=3 PBDMDC_YEAR_RANGES=1850-2014 python -m evaluation.plot_obs_forced_timeseries
"""
import os
import pickle as pkl

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.plot_gm_timeseries import (volcanic_events, clim_vars_on_fig, region_labels,
                                           method_names_on_fig, method_order_on_fig)
from utils.params import (method_names, colors, regions, pdf_root, requested_lags, requested_year_ranges,
                          requested_clim_vars, requested_methods, run_suffixes, results_dir)

obs = '20CRv3'
clim_vars = ['tas_ocean', 'tas', 'psl']
units = {'tas_ocean': 'K', 'tas': 'K', 'psl': 'Pa'}
volcanic_color = '#b0c4de'
window_size = 12 * 3


def load_rows(artifact_results_dir, vars_):
    """[(label, entry_by_method)] for every GM / regional pickle found, in display order."""
    rows = []
    for var in vars_:
        candidates = [(f'gm_timeseries_{var}.pkl', f'GM {clim_vars_on_fig[var]}')]
        candidates += [(f'region_timeseries_{var}_{r}.pkl', f'{clim_vars_on_fig[var]} {region_labels[r]}')
                       for r in regions]
        for fname, label in candidates:
            path = os.path.join(artifact_results_dir, fname)
            if not os.path.exists(path):
                print(f'Skipping {label}: {path} not found', flush=True)
                continue
            with open(path, 'rb') as f:
                timeseries = pkl.load(f)
            if obs not in timeseries:
                raise KeyError(f'{obs} missing from {path}; rerun evaluation.compute_gm_timeseries')
            rows.append((var, label, timeseries[obs]))
    return rows


def plot_obs_rows(rows, methods):
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 2.3 * len(rows)), sharex=True, squeeze=False)
    for ax, (var, label, entry) in zip(axes[:, 0], rows):
        first = entry[methods[0]]
        assert first['truth_is_raw'], f'{label}: expected the raw reanalysis as truth'
        time_index = first['time']
        raw = np.ma.getdata(first['truth']).astype(float)
        smooth = pd.Series(raw).rolling(window_size, center=True, min_periods=1).mean().values

        for v_start, v_end, _ in volcanic_events:
            ax.axvspan(pd.Timestamp(f'{v_start}-01-01'), pd.Timestamp(f'{v_end}-01-01'),
                       color=volcanic_color, alpha=0.35, zorder=0, linewidth=0)
        ax.plot(time_index, raw, color='0.75', linewidth=0.4, zorder=1)
        ax.plot(time_index, smooth, color='0.25', linewidth=1.4, zorder=2)
        for method in methods:
            preds = entry[method]['preds']
            assert len(preds) == 1, f'{label} {method}: expected one 20CRv3 realization, got {len(preds)}'
            ax.plot(time_index, preds[0], color=colors[method], linewidth=0.8, zorder=3)

        ax.set_ylabel(f'{label} ({units[var]})', fontsize=10)
        ax.set_xlim(time_index[0], time_index[-1])
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    handles = [mlines.Line2D([], [], color='0.75', linewidth=1.0, label=f'{obs} (forced + internal), monthly'),
               mlines.Line2D([], [], color='0.25', linewidth=1.4, label=f'{obs} (forced + internal), 3-yr mean')]
    handles += [mlines.Line2D([], [], color=colors[m], linewidth=1.5, label=method_names_on_fig.get(m, m))
                for m in methods]
    handles.append(mpatches.Patch(color=volcanic_color, alpha=0.35, label='Volcanic event'))
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01 - 0.06 / len(rows)))
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    methods = [m for m in method_order_on_fig if m in requested_methods(method_names)]

    for lag in requested_lags([3]):
        for start_year, end_year in requested_year_ranges([(1850, 2014)]):
            year_suffix, run_suffix = run_suffixes(lag, (start_year, end_year))
            pdf_results_dir = os.path.join(pdf_root, f'results{year_suffix}{run_suffix}')
            os.makedirs(pdf_results_dir, exist_ok=True)

            rows = load_rows(results_dir(year_suffix, run_suffix), requested_clim_vars(clim_vars))
            fig = plot_obs_rows(rows, methods)
            out_fig = os.path.join(pdf_results_dir, 'obs_forced_timeseries.pdf')
            fig.savefig(out_fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved: {out_fig}', flush=True)

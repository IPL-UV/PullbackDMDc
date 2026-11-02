"""Global means of raw 20CRv3, its forced estimates (best PullbackDMDc variant, LR, LIM) and the ESM ensemble means.

ESM ensemble means are a reference, not a truth. All series are anomalies w.r.t. 1850-1900. As in
compute_gm_timeseries, the series are monthly and the psl truths (20CRv3, ESM means) are 3-year smoothed.
Run: PBDMDC_LAGS=3 PBDMDC_YEAR_RANGES=1850-2014 python -m evaluation.plot_gm_20crv3
"""
import os
import pickle as pkl

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.utils.metrics import coord_to_dtindex, global_mean, global_mean_flat, smooth_truth
from evaluation.utils.style import (CLIM_VARS_ON_FIG, ESM_COLORS, ESM_LINESTYLES, METHOD_NAMES_ON_FIG,
                                    display_model_name)
from utils.data_utils import load_truncated_prediction, load_truth
from utils.params import (artifact_root, colors, pdf_root, predictions_dir, requested_lags,
                          requested_year_ranges, run_suffixes)

obs = '20CRv3'
esms = ['CanESM5', 'CESM2', 'MIROC6', 'MPI-ESM1-2-LR']
clim_vars = ['tas_ocean', 'tas', 'psl']
units = {'tas_ocean': 'K', 'tas': 'K', 'psl': 'Pa'}  # GM means are in the data's units
baseline = (1850, 1900)
late_period = (1985, 2014)  # for the summary csv
FONTSIZE = 11


def best_method(var):
    return 'PullbackDMDc-2d' if var == 'psl' else 'PullbackDMDc-3d'


def methods_for(var):
    return ['LR', 'LIM', best_method(var)]  # best variant last so it is drawn on top


def truth_gm(var, model, year_suffix):
    truth, lats = load_truth(var, model, os.path.join(artifact_root, f'means{year_suffix}'), return_lats=True)
    if var == 'psl':
        truth = smooth_truth(truth, window_years=3)
    return coord_to_dtindex(truth.time), np.ma.getdata(global_mean(truth.to_masked_array(), lats)).astype(float)


def forced_gm(var, method, year_suffix, run_suffix):
    pcs, eofs, data_names = load_truncated_prediction(var, obs, method, predictions_dir(year_suffix, run_suffix),
                                                      return_data_names=True)
    assert len(data_names) == 1, f'{var} {method}: expected one {obs} fit, got {data_names}'
    with open(os.path.join(artifact_root, f'eofs{year_suffix}', var, obs, data_names[0] + '.pkl'), 'rb') as f:
        resc_coef_flat = pkl.load(f)['data_pars']['resc_coef_flat']
    return np.asarray(global_mean_flat(pcs[0] @ eofs[0], resc_coef_flat), dtype=float)


def period_mean(time_index, x, period):
    years = time_index.year
    return x[(years >= period[0]) & (years <= period[1])].mean()


def rebase(time_index, x):
    return x - period_mean(time_index, x, baseline)


def plot_figure(year_suffix, run_suffix):
    fig, axes = plt.subplots(len(clim_vars), 1, figsize=(7, 8), sharex=True)
    summary = []
    for i, (ax, var) in enumerate(zip(axes, clim_vars)):
        time_index, raw = truth_gm(var, obs, year_suffix)
        series = []  # (key, label, values, plot kwargs), back to front
        for model in esms:
            t_esm, gm = truth_gm(var, model, year_suffix)
            # Datasets stamp months on different days; only the months need to agree.
            assert t_esm.to_period('M').equals(time_index.to_period('M')), \
                f'{model} {var}: months differ from {obs} ({t_esm[[0, -1]]} vs {time_index[[0, -1]]})'
            series.append((model, display_model_name(model), gm,
                           dict(color=ESM_COLORS[model], ls=ESM_LINESTYLES[model], lw=1.0, zorder=1)))
        series.append((obs, f'{obs} raw (forced + internal)', raw,
                       dict(color=ESM_COLORS[obs], lw=0.8, alpha=0.6, zorder=2)))
        for method in methods_for(var):
            gm = forced_gm(var, method, year_suffix, run_suffix)
            assert len(gm) == len(time_index), f'{var} {method}: {len(gm)} steps, truth has {len(time_index)}'
            lw, zorder = (2.0, 4) if method == best_method(var) else (1.5, 3)
            series.append((method, METHOD_NAMES_ON_FIG[method], gm, dict(color=colors[method], lw=lw, zorder=zorder)))

        for key, label, x, kw in series:
            x = rebase(time_index, x)
            ax.plot(time_index, x, label=label, **kw)
            summary.append({'var': var, 'series': key, 'is_esm': key in esms,
                            f'change_{late_period[0]}-{late_period[1]}':
                                period_mean(time_index, x, late_period)})

        ax.axhline(0, color='0.5', lw=0.6, zorder=0)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylabel(f'GM {CLIM_VARS_ON_FIG[var]} anomaly ({units[var]})', fontsize=FONTSIZE)
        ax.tick_params(labelsize=FONTSIZE - 1)
        ax.text(0.01, 0.95, f'({chr(97 + i)})', transform=ax.transAxes, fontsize=FONTSIZE,
                fontweight='bold', va='top')
    axes[-1].set_xlim(time_index[0], time_index[-1])

    # Methods (both PullbackDMDc variants appear) and raw on the first row, ESMs on the second.
    method_keys = ['LR', 'LIM'] + sorted({best_method(v) for v in clim_vars}, reverse=True)
    handles = [mlines.Line2D([], [], color=colors[m], lw=2.0 if m.startswith('Pullback') else 1.5,
                             label=METHOD_NAMES_ON_FIG[m]) for m in method_keys]
    handles.append(mlines.Line2D([], [], color=ESM_COLORS[obs], lw=0.8, alpha=0.6, label=f'{obs} raw'))
    esm_handles = [mlines.Line2D([], [], color=ESM_COLORS[m], ls=ESM_LINESTYLES[m], lw=1.0,
                                label=display_model_name(m)) for m in esms]
    esm_handles += [mlines.Line2D([], [], alpha=0, label=' ')] * (len(handles) - len(esm_handles))
    # Legend fills column by column, so interleave the two rows.
    fig.legend(handles=[h for pair in zip(handles, esm_handles) for h in pair], loc='upper center',
               ncol=len(handles), fontsize=FONTSIZE - 1, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    return fig, pd.DataFrame(summary)


if __name__ == '__main__':
    for lag in requested_lags([3]):
        for start_year, end_year in requested_year_ranges([(1850, 2014)]):
            assert start_year <= baseline[0], f'{start_year}-{end_year} does not cover the {baseline} baseline'
            year_suffix, run_suffix = run_suffixes(lag, (start_year, end_year))
            out_dir = os.path.join(pdf_root, f'results{year_suffix}{run_suffix}')
            os.makedirs(out_dir, exist_ok=True)

            fig, summary = plot_figure(year_suffix, run_suffix)
            out_fig = os.path.join(out_dir, f'gm_{obs}_forced.pdf')
            fig.savefig(out_fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved: {out_fig}', flush=True)
            summary.to_csv(os.path.join(out_dir, f'gm_{obs}_forced_summary.csv'), index=False)

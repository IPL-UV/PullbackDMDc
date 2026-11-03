"""Global means of raw 20CRv3, its forced estimates (best PullbackDMDc variant, LR, LIM) and the ESM ensemble means.

psl is averaged south of 65°S (params.regions['s65']) instead of globally; the other variables are global means.
ESM ensemble means are a reference, not a truth. All series are anomalies w.r.t. 1850-1900. As in
compute_gm_timeseries, the series are monthly and the psl truths (20CRv3, ESM means) are 3-year smoothed.
Major eruptions are shaded as in plot_gm_timeseries.
Run: PBDMDC_LAGS=3 PBDMDC_YEAR_RANGES=1850-2014 python -m evaluation.plot_gm_20crv3
"""
import os
import pickle as pkl

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.utils.metrics import (coord_to_dtindex, global_mean, global_mean_flat, lat_flat_from_pars,
                                      regional_mean, regional_mean_flat, smooth_truth)
from evaluation.utils.style import (CLIM_VARS_ON_FIG, ESM_COLORS, ESM_LINESTYLES, METHOD_NAMES_ON_FIG,
                                    VOLCANIC_COLOR, VOLCANIC_EVENTS, display_model_name)
from utils.data_utils import load_truncated_prediction, load_truth
from utils.params import (artifact_root, colors, pdf_root, predictions_dir, regions, requested_lags,
                          requested_year_ranges, run_suffixes)

obs = '20CRv3'
esms = ['CanESM5', 'CESM2', 'MIROC6', 'MPI-ESM1-2-LR']
clim_vars = ['tas_ocean', 'tas', 'psl']
units = {'tas_ocean': 'K', 'tas': 'K', 'psl': 'Pa'}  # GM / regional means are in the data's units
var_regions = {'psl': 's65'}  # vars averaged over a params.regions latitude band; the rest are global means
region_labels = {'s65': 'S of 65°S'}
baseline = (1850, 1900)
late_period = (1985, 2014)  # for the summary csv
FONTSIZE = 14
lim_alpha = 0.45
raw_color = '0.6'


def best_method(var):
    return 'PullbackDMDc-2d' if var == 'psl' else 'PullbackDMDc-3d'


def truth_gm(var, model, year_suffix):
    truth, lats = load_truth(var, model, os.path.join(artifact_root, f'means{year_suffix}'), return_lats=True)
    if var == 'psl':
        truth = smooth_truth(truth, window_years=3)
    x = truth.to_masked_array()
    gm = regional_mean(x, lats, *regions[var_regions[var]]) if var in var_regions else global_mean(x, lats)
    return coord_to_dtindex(truth.time), np.ma.getdata(gm).astype(float)


def forced_gm(var, method, year_suffix, run_suffix):
    pcs, eofs, data_names = load_truncated_prediction(var, obs, method, predictions_dir(year_suffix, run_suffix),
                                                      return_data_names=True)
    assert len(data_names) == 1, f'{var} {method}: expected one {obs} fit, got {data_names}'
    with open(os.path.join(artifact_root, f'eofs{year_suffix}', var, obs, data_names[0] + '.pkl'), 'rb') as f:
        data_pars = pkl.load(f)['data_pars']
    recon = pcs[0] @ eofs[0]
    if var in var_regions:
        gm = regional_mean_flat(recon, data_pars['resc_coef_flat'], lat_flat_from_pars(data_pars),
                                *regions[var_regions[var]])
    else:
        gm = global_mean_flat(recon, data_pars['resc_coef_flat'])
    return np.asarray(gm, dtype=float)


def period_mean(time_index, x, period):
    years = time_index.year
    return x[(years >= period[0]) & (years <= period[1])].mean()


def rebase(time_index, x):
    return x - period_mean(time_index, x, baseline)


def plot_figure(year_suffix, run_suffix):
    fig, axes = plt.subplots(len(clim_vars), 1, figsize=(8.5, 8), sharex=True)
    summary = []
    for ax, var in zip(axes, clim_vars):
        time_index, raw = truth_gm(var, obs, year_suffix)
        best = best_method(var)
        forced = {}
        for method in ['LIM', 'LR', best]:
            forced[method] = forced_gm(var, method, year_suffix, run_suffix)
            assert len(forced[method]) == len(time_index), \
                f'{var} {method}: {len(forced[method])} steps, truth has {len(time_index)}'

        # (key, label, values, plot kwargs), back to front: LIM, raw, ESMs, LR, best PullbackDMDc variant
        series = [('LIM', METHOD_NAMES_ON_FIG['LIM'], forced['LIM'],
                   dict(color=colors['LIM'], lw=1.5, alpha=lim_alpha, zorder=1)),
                  (obs, f'{obs} raw (forced + internal)', raw, dict(color=raw_color, lw=0.8, zorder=2))]
        for model in esms:
            t_esm, gm = truth_gm(var, model, year_suffix)
            # Datasets stamp months on different days; only the months need to agree.
            assert t_esm.to_period('M').equals(time_index.to_period('M')), \
                f'{model} {var}: months differ from {obs} ({t_esm[[0, -1]]} vs {time_index[[0, -1]]})'
            series.append((model, display_model_name(model), gm,
                           dict(color=ESM_COLORS[model], ls=ESM_LINESTYLES[model], lw=1.0, zorder=3)))
        series.append(('LR', METHOD_NAMES_ON_FIG['LR'], forced['LR'], dict(color=colors['LR'], lw=1.5, zorder=4)))
        series.append((best, METHOD_NAMES_ON_FIG[best], forced[best], dict(color=colors[best], lw=2.0, zorder=5)))

        for start, end, _ in VOLCANIC_EVENTS:
            ax.axvspan(pd.Timestamp(f'{start}-01-01'), pd.Timestamp(f'{end}-01-01'), color=VOLCANIC_COLOR,
                       alpha=0.35, zorder=0, linewidth=0)
        for key, label, x, kw in series:
            x = rebase(time_index, x)
            ax.plot(time_index, x, label=label, **kw)
            summary.append({'var': var, 'region': var_regions.get(var, 'global'), 'series': key, 'is_esm': key in esms,
                            f'change_{late_period[0]}-{late_period[1]}':
                                period_mean(time_index, x, late_period)})

        ax.axhline(0, color='0.5', lw=0.6, zorder=0)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        prefix = region_labels[var_regions[var]] if var in var_regions else 'GM'
        ax.set_ylabel(f'{prefix} {CLIM_VARS_ON_FIG[var]} anomaly ({units[var]})', fontsize=FONTSIZE)
        ax.tick_params(labelsize=FONTSIZE - 1)
    axes[-1].set_xlim(time_index[0], time_index[-1])

    # Methods (both PullbackDMDc variants appear) and raw on the first row, ESMs on the second.
    method_keys = ['LR', 'LIM'] + sorted({best_method(v) for v in clim_vars}, reverse=True)
    handles = [mlines.Line2D([], [], color=colors[m], lw=2.0 if m.startswith('Pullback') else 1.5,
                             alpha=lim_alpha if m == 'LIM' else 1.0, label=METHOD_NAMES_ON_FIG[m])
               for m in method_keys]
    handles.append(mlines.Line2D([], [], color=raw_color, lw=0.8, label=f'{obs} raw'))
    esm_handles = [mlines.Line2D([], [], color=ESM_COLORS[m], ls=ESM_LINESTYLES[m], lw=1.0,
                                label=display_model_name(m)) for m in esms]
    esm_handles.append(mpatches.Patch(color=VOLCANIC_COLOR, alpha=0.35, label='Volcanic event'))
    esm_handles += [mlines.Line2D([], [], alpha=0, label=' ')] * (len(handles) - len(esm_handles))
    # Legend fills column by column, so interleave the two rows.
    fig.legend(handles=[h for pair in zip(handles, esm_handles) for h in pair], loc='upper center',
               ncol=len(handles), fontsize=FONTSIZE - 1, frameon=False, columnspacing=1.0, handlelength=1.5,
               bbox_to_anchor=(0.5, 0.0))
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

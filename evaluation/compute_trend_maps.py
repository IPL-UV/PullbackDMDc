"""R2-M2 (ii): linear-trend maps over a window (default 1985-2014) for the truth and the forced estimates.

Trends are computed in PC space: every estimator predicts fr_pcs @ eofs + data_mean, so a member's trend
map is slope(pcs) @ eofs. Metrics are computed in the sqrt(cos lat)-scaled space (area weighted), as in
compute_trend_taylor_viz.py; maps are unscaled to K or hPa per decade for plotting.
Run: PBDMDC_LAGS=3 PBDMDC_YEAR_RANGES=1850-2014 PBDMDC_TREND_WINDOW=1985-2014 python -m evaluation.compute_trend_maps
"""
import json
import os
import pickle as pkl
import time

import numpy as np
import pandas as pd

from evaluation.compute_trend_taylor_viz import compute_linear_trend
from evaluation.utils.taylor_diagram import TaylorDiagramEstimator
from evaluation.utils.trends import trend_window, window_mask, window_label
from utils.data_utils import load_truncated_prediction, load_truth, undo_lat_scaling
from utils.params import (artifact_root, requested_lags, requested_year_ranges, requested_clim_vars,
                          requested_methods, predictions_dir, run_suffixes, results_dir)

datasets = ['CESM2', 'MPI-ESM1-2-LR', 'CanESM5', 'MIROC6', '20CRv3']
obs = '20CRv3'
per_decade = 120.  # slopes are per month
unit_scale = {'tas_ocean': 1., 'tas': 1., 'psl': 0.01}  # psl: Pa -> hPa


def default_methods(var):
    return ['PullbackDMDc-2d' if var == 'psl' else 'PullbackDMDc-3d', 'LR', 'LIM']


def to_map(flat, pars, var):
    lat = np.asarray(pars['lat'])
    return undo_lat_scaling(flat[:, None], lat, int(pars['nlon']), pars['nan_mask'])[:, :, 0] * unit_scale[var]


def check_against_full_period_json(metrics_rows, json_path, var):
    """With the full period as window, pooled metrics must equal compute_trend_taylor_viz's JSON."""
    with open(json_path) as f:
        td = json.load(f)
    worst = 0.0
    for row in metrics_rows:
        if row['member'] != 'pooled':
            continue
        ref = td[row['model']][row['method']]['total']
        worst = max(worst, abs(row['corr'] - ref['corr']), abs(row['rmse'] - ref['rmse']))
    print(f'{var}: max |pooled metric - td_metrics_trend JSON| = {worst:.2e}', flush=True)
    if worst > 1e-6:
        raise RuntimeError(f'{var}: trend-map metrics disagree with {json_path} ({worst:.2e})')


if __name__ == '__main__':
    start_time = time.time()
    window = trend_window()
    wlabel = window_label(window)

    for lag in requested_lags([3]):
        for start_year, end_year in requested_year_ranges([(1850, 2014)]):
            year_suffix, run_suffix = run_suffixes(lag, (start_year, end_year))
            base_pred_dir = predictions_dir(year_suffix, run_suffix)
            base_truth_dir = os.path.join(artifact_root, f'means{year_suffix}')
            eofs_base_path = os.path.join(artifact_root, f'eofs{year_suffix}')
            out_dir = results_dir(year_suffix, run_suffix)
            os.makedirs(out_dir, exist_ok=True)
            full_period = window == (start_year, end_year)

            for var in requested_clim_vars(['tas_ocean', 'psl', 'tas']):
                methods = requested_methods(default_methods(var))
                print(f'\n=== Trend maps {wlabel}: {var}, methods {methods} ===', flush=True)
                arrays, metrics_rows = {}, []
                tde = TaylorDiagramEstimator(centered=False, normalized=True)

                for model in datasets:
                    print(f'Processing {model}', flush=True)
                    truth_ds = load_truth(var, model, base_truth_dir)
                    mask = window_mask(truth_ds['time'].dt.year.values, window)
                    truth = truth_ds.stack(z=('lat', 'lon')).values.astype(np.float64)
                    nan_cols = np.isnan(truth[0, :])
                    truth = truth[:, ~nan_cols]
                    truth_trend = compute_linear_trend(truth[mask]) * per_decade

                    pars = None
                    for method in methods:
                        pcs_list, eofs_list, data_names = load_truncated_prediction(
                            var=var, model=model, method=method, base_pred_dir=base_pred_dir,
                            return_data_names=True)
                        if not data_names:
                            print(f'  {method}: no predictions, skipped', flush=True)
                            continue
                        order = np.argsort(data_names)
                        data_names = [data_names[i] for i in order]

                        if pars is None:
                            with open(os.path.join(eofs_base_path, var, model, data_names[0] + '.pkl'), 'rb') as f:
                                pars = pkl.load(f)['data_pars']
                            assert (~pars['nan_mask']).sum() == truth.shape[1], f'{model}: grid size mismatch'
                            assert np.array_equal(pars['nan_mask'].ravel(), nan_cols), f'{model}: nan mask mismatch'

                        member_trends = []
                        for k, i in enumerate(order):
                            pcs, eofs = pcs_list[i], eofs_list[i]
                            assert pcs.shape[0] == truth.shape[0], f'{model} {method}: time length mismatch'
                            member_trend = compute_linear_trend(pcs[mask]) @ eofs * per_decade
                            if k == 0:  # PC-space shortcut must equal the trend of the reconstructed field
                                direct = compute_linear_trend((pcs @ eofs)[mask]) * per_decade
                                assert np.allclose(member_trend, direct, rtol=1e-8, atol=1e-12 * np.abs(direct).max())
                            member_trends.append(member_trend)
                        member_trends = np.vstack(member_trends)
                        mean_trend = member_trends.mean(axis=0)

                        arrays[f'{model}__{method}__mean'] = to_map(mean_trend, pars, var)
                        arrays[f'{model}__{method}__std'] = to_map(member_trends.std(axis=0), pars, var)

                        if model == obs:
                            continue
                        rows = [dict(member=n, **tde.compute(truth_trend, t)) for n, t in zip(data_names, member_trends)]
                        rows.append(dict(member='pooled', **tde.compute(
                            np.repeat(truth_trend[None, :], len(member_trends), axis=0), member_trends)))
                        rows.append(dict(member='member_mean_map', **tde.compute(truth_trend, mean_trend)))
                        for row in rows:
                            metrics_rows.append({'var': var, 'window': wlabel, 'model': model, 'method': method,
                                                 'member': row['member'], 'corr': float(row['corr']),
                                                 'rmse': float(row['rmse']), 'std_ratio': float(row['std_Y'])})

                    arrays[f'{model}__truth'] = to_map(truth_trend, pars, var)
                    arrays[f'{model}__lat'] = np.asarray(pars['lat'])
                    arrays[f'{model}__lon'] = np.asarray(pars['lon'])

                out_npz = os.path.join(out_dir, f'trend_maps_{wlabel}_{var}.npz')
                np.savez(out_npz, methods=np.array(methods), datasets=np.array(datasets), **arrays)
                out_csv = os.path.join(out_dir, f'trend_maps_{wlabel}_{var}_metrics.csv')
                pd.DataFrame(metrics_rows).to_csv(out_csv, index=False)
                print(f'Saved: {out_npz}\nSaved: {out_csv}', flush=True)

                json_path = os.path.join(out_dir, f'td_metrics_trend_{var}.json')
                if full_period and os.path.exists(json_path):
                    check_against_full_period_json(metrics_rows, json_path, var)

    print(f'\n=== Total time: {time.time() - start_time:.2f} seconds ===', flush=True)

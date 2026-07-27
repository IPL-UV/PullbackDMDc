import gc
import json
import os
import time

import numpy as np

from evaluation.utils.taylor_diagram import TaylorDiagramEstimator
from utils.data_utils import load_prediction, load_truth
from utils.params import artifact_root, esms, lags, method_names, year_ranges, requested_lags, requested_year_ranges, predictions_dir


def compute_linear_trend(field_2d: np.ndarray) -> np.ndarray:
    """Compute linear trend at each spatial point for data shaped (time, space)."""
    time_index = np.arange(field_2d.shape[0], dtype=np.float64)
    return np.polyfit(time_index, field_2d, 1)[0]


if __name__ == '__main__':
    start_time = time.time()

    clim_vars = ['tas_ocean', 'psl', 'tas']

    for lag in requested_lags(lags):
        for start_year, end_year in requested_year_ranges(year_ranges):
            if lag == 3:
                lag_suffix = ''
            else:
                lag_suffix = f'_lag{lag}'

            if (start_year, end_year) == (1950, 2014):
                year_suffix = '_tier1'
            elif (start_year, end_year) == (1850, 2014):
                year_suffix = ''
            else:
                print('year range not recognized')

            base_pred_dir = predictions_dir(year_suffix, lag_suffix)
            base_truth_dir = os.path.join(artifact_root, f'means{year_suffix}')
            results_dir = os.path.join(artifact_root, 'evaluation_results', f'results{year_suffix}{lag_suffix}')

            os.makedirs(results_dir, exist_ok=True)

            for var in clim_vars:
                print(
                    f'\n=== Processing trend TD for {var}, lag={lag}, years={start_year}-{end_year} ===',
                    flush=True,
                )

                # td_metrics[model][method] = {'ensemble': [...], 'total': {...}}
                # same structure as compute_taylor_vis.py, but values are from trend maps
                td_metrics = {}

                for model in esms:
                    print(f'Processing model: {model}', flush=True)
                    td_metrics[model] = {}

                    truth_ds = load_truth(var, model, base_truth_dir)
                    truth = truth_ds.stack(z=('lat', 'lon')).values.astype(np.float64)

                    nan_mask = np.isnan(truth[0, :])
                    truth = truth[:, ~nan_mask]

                    truth_trend = compute_linear_trend(truth)

                    for method in method_names:
                        print(f'  Computing trend Taylor metrics for method: {method}', flush=True)

                        preds = load_prediction(
                            var=var,
                            model=model,
                            method=method,
                            base_pred_dir=base_pred_dir,
                        ).astype(np.float64)

                        n_members = preds.shape[2]
                        trend_member_metrics = []
                        pred_trends = []

                        tde = TaylorDiagramEstimator(centered=False, normalized=True)

                        for member_idx in range(n_members):
                            pred_member = preds[:, :, member_idx]
                            pred_member_trend = compute_linear_trend(pred_member)
                            pred_trends.append(pred_member_trend)

                            td_vals = tde.compute(truth_trend, pred_member_trend)
                            trend_member_metrics.append(td_vals)

                        pred_trends = np.vstack(pred_trends)
                        truth_trends = np.repeat(truth_trend[np.newaxis, :], repeats=n_members, axis=0)
                        total_td_vals = tde.compute(truth_trends, pred_trends)

                        td_metrics[model][method] = {
                            'ensemble': trend_member_metrics,
                            'total': total_td_vals,
                        }

                    gc.collect()

                out_path = os.path.join(results_dir, f'td_metrics_trend_{var}.json')
                with open(out_path, 'w') as f:
                    json.dump(td_metrics, f)
                print(f'Saved: {out_path}', flush=True)

    elapsed = time.time() - start_time
    print(f'\n=== Total time: {elapsed:.2f} seconds ===', flush=True)

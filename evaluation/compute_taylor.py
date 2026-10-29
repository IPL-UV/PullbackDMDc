import numpy as np
import os
import json
import time
import gc
from evaluation.utils.metrics import smooth_truth
from evaluation.utils.taylor_diagram import Ensemble_PCA_TD_Estimator
from utils.data_utils import load_truncated_prediction, load_truth
from utils.params import artifact_root, method_names, esms, year_ranges, lags, requested_lags, requested_year_ranges, predictions_dir
from utils.params import method_names, esms, year_ranges, lags

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
                print(f'\n=== Processing {var}, lag={lag}, years={start_year}-{end_year} ===', flush=True)

                # td_metrics[model][method] = {'ensemble': [...], 'total': {...}}
                # each entry contains only numeric Taylor diagram values (no plot styling)
                td_metrics = {}

                for model in esms:
                    print(f'Processing model: {model}', flush=True)
                    td_metrics[model] = {}

                    truth_ds = load_truth(var, model, base_truth_dir)
                    if var == 'psl':
                        truth_ds = smooth_truth(truth_ds, window_years=3)

                    truth = truth_ds.stack(z=('lat', 'lon')).values.astype(np.float64)
                    nan_mask = np.isnan(truth[0, :])
                    truth = truth[:, ~nan_mask]

                    for method in method_names:
                        print(f'  Computing Taylor Diagram metrics for method: {method}', flush=True)

                        preds_pcs, preds_eofs = load_truncated_prediction(
                            var=var,
                            model=model,
                            method=method,
                            base_pred_dir=base_pred_dir,
                        )

                        TDE = Ensemble_PCA_TD_Estimator(centered=True, normalized=True)
                        td_values_list, total_td_values = TDE.compute(
                            X=truth, P_Y_list=preds_pcs, V_Y_list=preds_eofs
                        )

                        td_metrics[model][method] = {
                            'ensemble': td_values_list,  # list of dicts, one per ensemble member
                            'total': total_td_values,    # single dict for the aggregate point
                        }

                    gc.collect()

                # Save metrics for this var / lag / year-range combination
                out_path = os.path.join(results_dir, f'td_metrics_{var}.json')
                with open(out_path, 'w') as f:
                    json.dump(td_metrics, f)
                print(f'Saved: {out_path}', flush=True)

        elapsed = time.time() - start_time
        print(f'\n=== Total time: {elapsed:.2f} seconds ===', flush=True)
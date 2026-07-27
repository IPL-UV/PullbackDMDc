import numpy as np
import os
import time
import pickle as pkl
from utils.data_utils import load_truncated_prediction, load_truth
from evaluation.utils.metrics import global_mean, global_mean_flat, coord_to_dtindex, smooth_truth
from utils.params import artifact_root, esms, method_names, colors, year_ranges, lags, requested_lags, requested_year_ranges, predictions_dir

if __name__ == '__main__':

    start_time = time.time()

    clim_vars = ['tas_ocean','psl','tas']
    esms = ['CanESM5', 'MIROC6', 'CESM2', 'MPI-ESM1-2-LR']

    for lag in requested_lags(lags):
        for start_year, end_year in requested_year_ranges(year_ranges):

            if lag == 3:
                lag_suffix = ''
            elif lag == 12:
                lag_suffix = '_lag12'
            else:
                print('lag not recognized')

            if (start_year, end_year) == (1950, 2014):
                year_suffix = '_tier1'
            elif (start_year, end_year) == (1850, 2014):
                year_suffix = ''
            else:
                print('year range not recognized')

            base_pred_dir = predictions_dir(year_suffix, lag_suffix)
            base_truth_dir = os.path.join(artifact_root, f'means{year_suffix}')
            eofs_base_path = os.path.join(artifact_root, f'eofs{year_suffix}')
            results_dir = os.path.join(artifact_root, 'evaluation_results', f'results{year_suffix}{lag_suffix}')

            os.makedirs(results_dir, exist_ok=True)

            for var in clim_vars:
                print(f'\n=== Processing variable: {var} ===', flush=True)

                # Nested dict: timeseries[model][method] = {'preds': [...], 'truth': ..., 'time': ...}
                timeseries = {}

                for model in esms:
                    print(f'\nProcessing model: {model}', flush=True)
                    timeseries[model] = {}

                    truth_ds, lats = load_truth(var, model, base_truth_dir, return_lats=True)

                    if var == 'psl':
                        truth_ds = smooth_truth(truth_ds, window_years=3)

                    truth_gm = global_mean(truth_ds, lats)
                    time_index = coord_to_dtindex(truth_ds.time)
                    print(time_index.shape if hasattr(time_index, 'shape') else len(time_index))

                    for method in method_names:
                        print(f'  Processing method: {method}', flush=True)

                        preds_pcs, preds_eofs, data_names = load_truncated_prediction(
                            var=var,
                            model=model,
                            method=method,
                            base_pred_dir=base_pred_dir,
                            return_data_names=True,
                        )

                        pred_gms = []
                        for pcs, eofs, data_name in zip(preds_pcs, preds_eofs, data_names):
                            reconstructed_flat = pcs @ eofs

                            file_path = os.path.join(eofs_base_path, var, model, data_name + '.pkl')
                            print(file_path, flush=True)
                            with open(file_path, 'rb') as f:
                                precomputed_eofs = pkl.load(f)
                                resc_coef_flat = precomputed_eofs['data_pars']['resc_coef_flat']

                            pred_gm = global_mean_flat(reconstructed_flat, resc_coef_flat)
                            pred_gms.append(pred_gm)

                        timeseries[model][method] = {
                            'preds': pred_gms,  # list of 1-D arrays, one per ensemble member
                            'truth': truth_gm,
                            'time': time_index,
                        }

                # Save all timeseries for this var / lag / year-range combination
                out_path = os.path.join(results_dir, f'gm_timeseries_{var}.pkl')
                with open(out_path, 'wb') as f:
                    pkl.dump(timeseries, f)
                print(f'\nSaved timeseries: {out_path}', flush=True)

        elapsed = time.time() - start_time
        print(f'\n=== Total time: {elapsed:.2f} seconds ===', flush=True)
import numpy as np
import os
import time
import pickle as pkl
from utils.data_utils import load_truncated_prediction, load_truth
from evaluation.utils.metrics import (global_mean, global_mean_flat, coord_to_dtindex, smooth_truth,
                                      lat_flat_from_pars, regional_mean, regional_mean_flat)
from utils.params import (artifact_root, esms, method_names, colors, year_ranges, lags, requested_lags,
                          requested_year_ranges, requested_clim_vars, predictions_dir, run_suffixes,
                          results_dir, regions, region_vars)


def _env_list(name, default):
    return [item.strip() for item in os.getenv(name, default).split(',') if item.strip()]


if __name__ == '__main__':

    start_time = time.time()

    clim_vars = requested_clim_vars(['tas_ocean','psl','tas'])
    esms = ['CanESM5', 'MIROC6', 'CESM2', 'MPI-ESM1-2-LR']

    # Regional means (R2-L374): latitude bands from params.regions, for the variables in region_vars.
    region_names = _env_list('PBDMDC_REGIONS', 'n65,s65')
    region_vars = _env_list('PBDMDC_REGION_VARS', ','.join(region_vars))
    # PBDMDC_SKIP_GM=1 writes only the regional pickles (leaves gm_timeseries_{var}.pkl untouched).
    skip_gm = os.getenv('PBDMDC_SKIP_GM', '0') == '1'

    for lag in requested_lags(lags):
        for start_year, end_year in requested_year_ranges(year_ranges):

            year_suffix, run_suffix = run_suffixes(lag, (start_year, end_year))

            base_pred_dir = predictions_dir(year_suffix, run_suffix)
            base_truth_dir = os.path.join(artifact_root, f'means{year_suffix}')
            eofs_base_path = os.path.join(artifact_root, f'eofs{year_suffix}')
            out_dir = results_dir(year_suffix, run_suffix)

            os.makedirs(out_dir, exist_ok=True)

            for var in clim_vars:
                print(f'\n=== Processing variable: {var} ===', flush=True)

                do_regions = var in region_vars
                if skip_gm and not do_regions:
                    print(f'Skipping {var}: PBDMDC_SKIP_GM=1 and no regions requested', flush=True)
                    continue

                # Nested dict: timeseries[model][method] = {'preds': [...], 'truth': ..., 'time': ...}
                timeseries = {}
                # Same structure per region: region_ts[region][model][method]
                region_ts = {r: {} for r in region_names} if do_regions else {}

                for model in esms:
                    print(f'\nProcessing model: {model}', flush=True)
                    timeseries[model] = {}

                    truth_ds, lats = load_truth(var, model, base_truth_dir, return_lats=True)

                    if var == 'psl':
                        truth_ds = smooth_truth(truth_ds, window_years=3)

                    truth_gm = global_mean(truth_ds, lats)
                    time_index = coord_to_dtindex(truth_ds.time)
                    print(time_index.shape if hasattr(time_index, 'shape') else len(time_index))

                    truth_regions = {}
                    for r in region_ts:
                        region_ts[r][model] = {}
                        truth_regions[r] = regional_mean(truth_ds, lats, *regions[r])

                    grid_checked = False
                    for method in method_names:
                        print(f'  Processing method: {method}', flush=True)

                        preds_pcs, preds_eofs, data_names = load_truncated_prediction(
                            var=var,
                            model=model,
                            method=method,
                            base_pred_dir=base_pred_dir,
                            return_data_names=True,
                        )

                        # Sort members by name so member i is the same realization for every method
                        # (os.listdir order differs between methods; needed for paired comparisons).
                        order = np.argsort(data_names)
                        preds_pcs = [preds_pcs[i] for i in order]
                        preds_eofs = [preds_eofs[i] for i in order]
                        data_names = [data_names[i] for i in order]

                        pred_gms = []
                        pred_regions = {r: [] for r in region_ts}
                        for pcs, eofs, data_name in zip(preds_pcs, preds_eofs, data_names):
                            reconstructed_flat = pcs @ eofs

                            file_path = os.path.join(eofs_base_path, var, model, data_name + '.pkl')
                            print(file_path, flush=True)
                            with open(file_path, 'rb') as f:
                                precomputed_eofs = pkl.load(f)
                                resc_coef_flat = precomputed_eofs['data_pars']['resc_coef_flat']

                            pred_gm = global_mean_flat(reconstructed_flat, resc_coef_flat)
                            pred_gms.append(pred_gm)

                            if region_ts:
                                lat_flat = lat_flat_from_pars(precomputed_eofs['data_pars'])
                                if not grid_checked:
                                    # EOF grid must match the truth grid for regional means to be comparable.
                                    assert np.isin(np.unique(lat_flat), lats).all(), f'{data_name}: lat grid differs from truth'
                                    grid_checked = True
                                for r in region_ts:
                                    pred_regions[r].append(
                                        regional_mean_flat(reconstructed_flat, resc_coef_flat, lat_flat, *regions[r]))

                        timeseries[model][method] = {
                            'preds': pred_gms,  # list of 1-D arrays, one per ensemble member
                            'truth': truth_gm,
                            'time': time_index,
                            'data_names': data_names,  # member names, sorted; same order as 'preds'
                        }
                        for r in region_ts:
                            region_ts[r][model][method] = {
                                'preds': pred_regions[r],
                                'truth': truth_regions[r],
                                'time': time_index,
                                'data_names': data_names,
                            }

                # Save all timeseries for this var / lag / year-range combination
                if not skip_gm:
                    out_path = os.path.join(out_dir, f'gm_timeseries_{var}.pkl')
                    with open(out_path, 'wb') as f:
                        pkl.dump(timeseries, f)
                    print(f'\nSaved timeseries: {out_path}', flush=True)

                for r, ts in region_ts.items():
                    out_path = os.path.join(out_dir, f'region_timeseries_{var}_{r}.pkl')
                    with open(out_path, 'wb') as f:
                        pkl.dump(ts, f)
                    print(f'Saved regional timeseries: {out_path}', flush=True)

        elapsed = time.time() - start_time
        print(f'\n=== Total time: {elapsed:.2f} seconds ===', flush=True)

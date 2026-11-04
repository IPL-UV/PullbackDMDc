import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
from utils.data_utils import undo_lat_scaling
from evaluation.utils.timescales import acf
from utils.params import artifact_root, esms, lag_suffix_for, predictions_dir, year_suffix_for

# Parameters for selection
lag = 3
start_year, end_year = 1850, 2014
lag_suffix, year_suffix = lag_suffix_for(lag), year_suffix_for((start_year, end_year))

clim_vars = ['tas_ocean', 'tas', 'psl']


eofs_base_path = os.path.join(artifact_root, f'eofs{year_suffix}')
base_pred_dir = predictions_dir(year_suffix, lag_suffix)
results_dir = os.path.join(artifact_root, 'evaluation_results', f'results{year_suffix}{lag_suffix}')

# --------------------------------------------------
# Compute and save ACFs for all ESMs / members / modes
# --------------------------------------------------

length = 100 * 12

# acf_data[model] = {
#     'acfs':   list of shape (n_members, top_n, length),
#     'titles': list of per-mode title strings (from first member)
# }

def main():
    for clim_var in clim_vars:
        if clim_var == 'psl':
            chosen_method = 'PullbackDMDc-2d'
            top_n = 200
        else:
            chosen_method = 'PullbackDMDc-3d'
            top_n = 20
        acf_data = {}
        for chosen_model in esms:

            model_dir = os.path.join(base_pred_dir, clim_var, chosen_model)

            dmd_files = [
                os.path.join(model_dir, f)
                for f in os.listdir(model_dir)
                if f.split('_')[-1].split('.')[0] == chosen_method
            ]

            member_acfs = []   # one (top_n, length) array per member
            mode_titles = None

            for dmd_file in dmd_files:
                with open(dmd_file, 'rb') as f:
                    dmd = pickle.load(f)

                dmd.compute_modes()

                rotated = dmd.compute_rotated_modes(top_n=top_n)
                mode_ts = rotated['internal_time_series']
                mode_count = mode_ts.shape[1]

                mode_acfs = [np.full(length, np.nan)] * mode_count
                titles = []
                for k in range(mode_count):
                    mode_acfs[k] = acf(mode_ts[:, k], axis=0, length=length)

                member_acfs.append(np.stack(mode_acfs))  # (top_n, length)
                if mode_titles is None:
                    mode_titles = titles  # save titles from first member

            acf_data[chosen_model] = member_acfs

        out_path = os.path.join(results_dir, f'acfs_{clim_var}_{chosen_method}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(acf_data, f)
        print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
from utils.data_utils import undo_lat_scaling
from utils.params import predictions_dir
from matplotlib.patches import Patch

# Helper functions from eigs_vis.py
def compute_yearly_decay_time(lambdas, tau):
    return -tau / (12 * np.log(np.abs(lambdas)))

def compute_yearly_frequency(lambdas, tau):
    #omegas = np.imag(np.log(lambdas / np.abs(lambdas))) / tau
    #return (6 * np.abs(omegas)) / np.pi
    omegas = np.angle(lambdas) / tau
    return (6 * np.abs(omegas)) / np.pi

# Fallback parsers if not already defined in a previous cell
def parse_ensemble_member_from_filename(f_name, esm_name):
    # Match common CMIP-style member IDs first (e.g., r47i1p1f1)
    m = re.search(r'r\d+i\d+p\d+f\d+', f_name)
    if m:
        return m.group(0)

    # CESM2 can encode member as decimal + i/p/f block (e.g., 1281.005i1p1f1)
    m = re.search(r'\d+\.\d+i\d+p\d+f\d+', f_name)
    if m:
        return m.group(0)

    # Legacy fallbacks
    if esm_name == 'CESM2':
        return f_name.split('.')[0].split('_')[-1] + '.' + f_name.split('.')[1].split('_')[0]
    parts = f_name.split('_')
    return parts[5] if len(parts) > 5 else (parts[4] if len(parts) > 4 else 'unknown')

def parse_ensemble_member_from_eof_filename(f_name, esm_name):
    m = re.search(r'r\d+i\d+p\d+f\d+', f_name)
    if m:
        return m.group(0)

    m = re.search(r'\d+\.\d+i\d+p\d+f\d+', f_name)
    if m:
        return m.group(0)

    parts = f_name.split('_')
    if esm_name == 'CESM2':
        return parts[6] if len(parts) > 6 else 'unknown'
    return parts[5] if len(parts) > 5 else (parts[4] if len(parts) > 4 else 'unknown')


def display_model_name(model_name):
    return 'MPI-ESM' if model_name == 'MPI-ESM1-2-LR' else model_name


if __name__ == "__main__":
    # Parameters for selection
    year_suffix = ''#'_tier1'
    lag_suffix = ''#'_lag12'
    clim_var = 'tas_ocean'
    dmdc_method = 'PullbackDMDc-3d'  # Options: 'PullbackDMDc', 'PullbackDMDc-2d', 'PullbackDMDc-3d'
    methods = ['LIM', 'PullbackDMDc', 'PullbackDMDc-2d', 'PullbackDMDc-3d']
    colors = ['tab:blue', 'tab:red', 'tab:purple', 'tab:pink']
    esms = ['20CRv3', 'CESM2', 'MPI-ESM1-2-LR', 'CanESM5', 'MIROC6']  # Include observations + ESMs in Taylor figure order

    vmax = 0.05 # heatmap scale



    base_pred_dir = predictions_dir(year_suffix, lag_suffix)
    clim_dir = os.path.join(base_pred_dir, clim_var)


    results = pd.DataFrame(columns=[
        'Model', 'Ensemble Member', 'Method', 'Eigenvalue Rank',
        'Decay Time', 'Frequency', 'Eigenvalue', 'Eigenvalue Real', 'Eigenvalue Imag'
    ])

    for esm in esms:
        dir_path = os.path.join(clim_dir, esm)
        for method in methods:
            for f_name in os.listdir(dir_path):
                if method != f_name.split('_')[-1].split('.')[0]:
                    continue

                f_path = os.path.join(dir_path, f_name)
                with open(f_path, 'rb') as f:
                    dmd = pickle.load(f)


                dmd.compute_modes()
                eigs=dmd.eigvals
                decay_times = compute_yearly_decay_time(eigs, dmd.lag)
                yearly_frequency = compute_yearly_frequency(eigs, dmd.lag)


                ens_member = parse_ensemble_member_from_filename(f_name, esm)
                for i in range(eigs.shape[0]):
                    eig = eigs[i]
                    row = pd.DataFrame({
                        'Model': [esm],
                        'Ensemble Member': [ens_member],
                        'Method': [method],
                        'Eigenvalue Rank': [i + 1],
                        'Decay Time': [decay_times[i]],
                        'Frequency': [yearly_frequency[i]],
                        'Eigenvalue': [eig],
                        'Eigenvalue Real': [eig.real],
                        'Eigenvalue Imag': [eig.imag],
                    })
                    results = pd.concat([results, row], ignore_index=True)

    results.head()

    # PullbackDMDc-3d only: scatter view of eigenvalue distributions by decay rank

    required_cols = {'Model', 'Ensemble Member', 'Method', 'Decay Time', 'Eigenvalue Real', 'Eigenvalue Imag'}
    missing = required_cols - set(results.columns)
    if missing:
        raise ValueError(f'Missing required columns in results: {missing}')

    method_to_plot = 'PullbackDMDc-3d'  # Change this to the method you want to visualize
    sub = results[
        (results['Method'] == method_to_plot)
        & np.isfinite(results['Decay Time'])
        & np.isfinite(results['Eigenvalue Real'])
        & np.isfinite(results['Eigenvalue Imag'])
    ].copy()

    if sub.empty:
        raise ValueError(f'No rows found for method={method_to_plot}.')

    sub = (
        sub.groupby(['Model', 'Ensemble Member'], group_keys=False)
        .apply(lambda group: group.sort_values('Decay Time', ascending=False).assign(
            **{'Top Decay Rank': lambda frame: np.arange(1, len(frame) + 1)}
        ))
        .reset_index(drop=True)
    )

    if sub.empty:
        raise ValueError('No rows left after decay-rank sorting.')

    # Plot all available ranks; color-group ranks 1-2 vs ranks 3-4 vs all remaining ranks
    rank_order = sorted(sub['Top Decay Rank'].dropna().astype(int).unique())

    def rank_to_color(rank):
        if rank in (1, 2):
            return 'tab:blue'
        if rank in (3, 4):
            return 'tab:orange'
        return 'tab:gray'

    # Build panel order with 20CRv3 first on the left.
    available_models = sorted(sub['Model'].dropna().unique())

    obs_model = '20CRv3'
    if obs_model in available_models:
        obs_models = [obs_model]
    else:
        def normalize_model_name(name):
            return str(name).lower().replace('-', '').replace('_', '')

        obs_models = [m for m in available_models if normalize_model_name(m) == '20crv3']
        if not obs_models:
            obs_models = [m for m in available_models if '20cr' in normalize_model_name(m)]

    remaining_models = [m for m in esms if m in available_models and m not in obs_models]
    remaining_models += [m for m in available_models if m not in remaining_models and m not in obs_models]
    model_order = obs_models + remaining_models

    if len(model_order) == 0:
        raise ValueError('No models available to plot after filtering.')

    coords = sub[['Eigenvalue Real', 'Eigenvalue Imag']].to_numpy(dtype=float)
    max_abs = max(1.2, np.max(np.abs(coords)) * 1.05)

    theta = np.linspace(0, 2 * np.pi, 600)
    fig, axes = plt.subplots(
        1, len(model_order), figsize=(2 * len(model_order), 2),
        sharex=True, sharey=True, squeeze=False
    )

    for j, model in enumerate(model_order):
        ax = axes[0, j]
        model_sub = sub[sub['Model'] == model]

        for r in rank_order:
            rsub = model_sub[model_sub['Top Decay Rank'] == r]
            if rsub.empty:
                continue

            if rank_to_color(r) == 'tab:gray' and model != obs_model:
                alpha = 0.1
                size = 10
            elif model == obs_model:
                alpha = 0.8
                size = 50
            else:
                alpha = 0.4
                size = 10
            ax.scatter(
                rsub['Eigenvalue Real'],
                rsub['Eigenvalue Imag'],
                s=size,
                alpha=alpha,
                color=rank_to_color(r)
            )

        ax.plot(np.cos(theta), np.sin(theta), color='black', ls='--', lw=1.4)
        ax.axhline(0, color='0.7', lw=0.8)
        ax.axvline(0, color='0.7', lw=0.8)
        ax.set_aspect('equal')
        ax.set_xlim(0, max_abs)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(display_model_name(model), fontsize=10, pad=4)
        ax.set_xlabel('Real')
        if j == 0:
            ax.set_ylabel('Imag')

    legend_handles = [
        Patch(facecolor='tab:blue', edgecolor='tab:blue', alpha=0.4, label='Rank 1-2'),
        Patch(facecolor='tab:orange', edgecolor='tab:orange', alpha=0.4, label='Rank 3-4'),
        Patch(facecolor='tab:gray', edgecolor='tab:gray', alpha=0.4, label='Rank 5+'),
    ]
    fig.legend(
        handles=legend_handles,
        title='Decay Rank Group',
        loc='lower center',
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.25)
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    plt.savefig(f'evaluation_results/results{year_suffix}{lag_suffix}/eigenvalue_scatter_by_decay_rank_{clim_var}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
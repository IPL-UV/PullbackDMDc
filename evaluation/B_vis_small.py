import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr

from utils.data_utils import undo_lat_scaling
from utils.params import predictions_dir


def mean_pattern(patterns):
    if len(patterns) == 0:
        raise ValueError('No patterns found to average.')
    return np.mean(patterns, axis=0)


def display_model_name(model_name):
    return 'MPI-ESM' if model_name == 'MPI-ESM1-2-LR' else model_name


if __name__ == '__main__':

    clim_vars_on_fig = {'tas_ocean': 'OSAT', 'psl': 'SLP', 'tas': 'SAT'}
    
    method_names_on_fig = {'RegGMST': 'RegGMST', 'LR': 'Linear reg.','LIM': 'LIM','LIM-opt': 'LIM-opt',
                        'PullbackDMDc': 'PullbackDMDc', 'PullbackDMDc-2d': 'PullbackDMDc(2d)', 'PullbackDMDc-3d': 'PullbackDMDc(3d)'}

    base_path = '/data/databases/dmdc-variants/mmlea_v2/'
    lag = 3
    start_year, end_year = 1850, 2014

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

    clim_var = 'tas_ocean'  # 'psl', 'tas', 'tas_ocean'

    models = ['20CRv3', 'MPI-ESM1-2-LR']
    method_keys = ['LR', 'PullbackDMDc']

    base_pred_dir = predictions_dir(year_suffix, lag_suffix)
    clim_dir = os.path.join(base_pred_dir, clim_var)
    out_dir = os.path.join('.', f'evaluation_results/results{year_suffix}{lag_suffix}')
    os.makedirs(out_dir, exist_ok=True)

    sample_ds = xr.load_dataset(
        os.path.join(
            base_path,
            'ensembles',
            clim_var,
            'CanESM5',
            f'{clim_var}_Amon_CanESM5_historical_r25i1p1f1_g025_185001-201412.nc',
        )
    )
    nan_mask = np.isnan(sample_ds[clim_var].isel(time=0).values)
    lat = sample_ds.lat.values
    lon = sample_ds.lon.values
    nlon = len(lon)

    patterns = {(model, method): [] for model in models for method in method_keys}

    for model in models:
        model_dir = os.path.join(clim_dir, model)
        for f_name in os.listdir(model_dir):
            method = f_name.split('_')[-1].split('.')[0]
            if method not in method_keys:
                continue

            f_path = os.path.join(model_dir, f_name)
            with open(f_path, 'rb') as f:
                dmd = pickle.load(f)

            B = dmd.B.copy()
            eofs = dmd.eofs.copy()
            pattern = eofs.T @ B
            pattern = undo_lat_scaling(pattern, lat, nlon, nan_mask)
            patterns[(model, method)].append(pattern)

    fields = {}
    for model in models:
        lr = mean_pattern(patterns[(model, 'LR')]).reshape(len(lat), len(lon))
        dmdcp = mean_pattern(patterns[(model, 'PullbackDMDc')]).reshape(len(lat), len(lon))
        diff = dmdcp - lr
        fields[(model, method_names_on_fig['LR'])] = lr
        fields[(model, method_names_on_fig['PullbackDMDc'])] = dmdcp
        fields[(model, 'Difference')] = diff

    row_labels = [
        method_names_on_fig['LR'],
        method_names_on_fig['PullbackDMDc'],
        "Difference",
    ]

    row_limits = {}
    for row_label in row_labels:
        row_vals = np.concatenate([np.ravel(fields[(m, row_label)]) for m in models])
        row_vals = row_vals[np.isfinite(row_vals)]
        vmax = np.max(np.abs(row_vals)) if row_vals.size > 0 else 1e-12
        if vmax == 0:
            vmax = 1e-12
        vmax *= 0.5
        row_limits[row_label] = (-vmax, vmax)
    n_rows = len(row_labels)
    n_cols = len(models)
    title_fontsize = 10
    axis_label_fontsize = 10

    projection = ccrs.PlateCarree(central_longitude=180)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 1.5 * n_rows),
        subplot_kw={'projection': projection},
        constrained_layout=True,
    )

    lon2d, lat2d = np.meshgrid(lon, lat)
    row_mappables = {}

    for i, row_label in enumerate(row_labels):
        vmin, vmax = row_limits[row_label]
        levels = np.linspace(vmin, vmax, 11)
        for j, model in enumerate(models):
            ax = axes[i, j]
            field = fields[(model, row_label)]

            m = ax.contourf(
                lon2d,
                lat2d,
                field,
                transform=ccrs.PlateCarree(),
                cmap='RdBu_r',
                levels=levels,
                extend='both',
            )
            if j == 0:
                row_mappables[row_label] = m

            ax.coastlines(linewidth=0.7)
            mappable = m
            if clim_var == 'tas_ocean':
                ax.set_extent([-180, 180, -40, 60], crs=ccrs.PlateCarree())
            else:
                ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

            if i == 0:
                title_model = 'Obs (20CRv3)' if model == '20CRv3' else display_model_name(model)
                ax.set_title(title_model, fontsize=title_fontsize)
            if j == 0:
                ax.text(
                    -0.08,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va='center',
                    ha='right',
                    fontsize=axis_label_fontsize,
                )

    for i, row_label in enumerate(row_labels):
        cbar = fig.colorbar(
            row_mappables[row_label],
            ax=axes[i, :].tolist(),
            orientation='vertical',
            fraction=0.015,
            pad=0.02,
            format='%.2f',
            aspect=30,
        )

    fig.suptitle(
        f"{clim_vars_on_fig[clim_var]} B patterns: {method_names_on_fig['LR']} vs {method_names_on_fig['PullbackDMDc']}",
        fontsize=title_fontsize,
    )
    out_file = os.path.join(out_dir, f'B_vis_small_{clim_var}.pdf')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_file}', flush=True)
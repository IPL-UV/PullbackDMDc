import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr

from utils.data_utils import undo_lat_scaling, load_forcings
from evaluation.utils.style import CLIM_VARS_ON_FIG, METHOD_NAMES_ON_FIG, display_model_name
from utils.params import predictions_dir


def mean_pattern(patterns):
    if len(patterns) == 0:
        raise ValueError('No patterns found to average.')
    return np.mean(patterns, axis=0)


if __name__ == '__main__':

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

    clim_var = os.getenv('PBDMDC_CLIM_VAR', 'tas_ocean')  # 'psl', 'tas', 'tas_ocean'

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

    # LR and PullbackDMDc are fit to total ERF standardised by its 1850-2014 std (load_forcings /
    # load_forcings_pullback), so dividing B by that std gives field units per W m^-2.
    sigma_forcing = np.std(load_forcings(mean_center=False, start_year=start_year, end_year=end_year))
    print(f'std of total ERF {start_year}-{end_year}: {sigma_forcing:.4f} W m-2', flush=True)
    field_unit = {'tas_ocean': 'K', 'tas': 'K', 'psl': 'Pa'}[clim_var]
    per_forcing = f'{field_unit} per W m$^{{-2}}$'
    cbar_labels = {METHOD_NAMES_ON_FIG['LR']: per_forcing,
                   METHOD_NAMES_ON_FIG['PullbackDMDc']: per_forcing + '\nper 3-month step'}

    # No difference row: LR's B is a regression coefficient and PullbackDMDc's a per-step input
    # coefficient, so their difference has no consistent meaning.
    fields = {}
    for model in models:
        lr = mean_pattern(patterns[(model, 'LR')]).reshape(len(lat), len(lon)) / sigma_forcing
        dmdcp = mean_pattern(patterns[(model, 'PullbackDMDc')]).reshape(len(lat), len(lon)) / sigma_forcing
        fields[(model, METHOD_NAMES_ON_FIG['LR'])] = lr
        fields[(model, METHOD_NAMES_ON_FIG['PullbackDMDc'])] = dmdcp

    row_labels = [
        METHOD_NAMES_ON_FIG['LR'],
        METHOD_NAMES_ON_FIG['PullbackDMDc'],
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

    # Map latitude extent: for OSAT, the mask rows that contain data (40S-65N), rounded outward
    # to 5 degrees, as in plot_modes.py; falls back to the old fixed extent.
    if clim_var == 'tas_ocean':
        data_rows = (~nan_mask).any(axis=1)
        if data_rows.any():
            map_lat_extent = [max(-90., 5 * np.floor(lat[data_rows].min() / 5)),
                              min(90., 5 * np.ceil(lat[data_rows].max() / 5))]
        else:
            map_lat_extent = [-40, 60]
    else:
        map_lat_extent = [-90, 90]

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
            ax.set_extent([-180, 180] + map_lat_extent, crs=ccrs.PlateCarree())

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
        cbar.set_label(cbar_labels[row_label], fontsize=8)

    fig.suptitle(
        f"{CLIM_VARS_ON_FIG[clim_var]} B patterns: {METHOD_NAMES_ON_FIG['LR']} vs {METHOD_NAMES_ON_FIG['PullbackDMDc']}",
        fontsize=title_fontsize,
    )
    out_file = os.path.join(out_dir, f'B_vis_small_{clim_var}.pdf')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_file}', flush=True)
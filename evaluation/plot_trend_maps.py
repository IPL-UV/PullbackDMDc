"""R2-M2 (ii): plot the trend maps written by compute_trend_maps.py.

Rows = datasets (4 ESMs + 20CRv3), columns = truth then each method's member-mean trend. One shared
symmetric colorbar per figure. ESM estimate panels show the pooled pattern correlation r and normalised
RMSE over all members (the 'total' of the Taylor diagrams).
Run: PBDMDC_LAGS=3 PBDMDC_YEAR_RANGES=1850-2014 PBDMDC_TREND_WINDOW=1985-2014 python -m evaluation.plot_trend_maps
"""
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.utils.style import CLIM_VARS_ON_FIG, METHOD_NAMES_ON_FIG, display_model_name, p95_abs
from evaluation.utils.trends import trend_window, window_label
from utils.params import (pdf_root, requested_lags, requested_year_ranges, requested_clim_vars,
                          run_suffixes, results_dir)

obs = '20CRv3'
units = {'tas_ocean': 'K', 'tas': 'K', 'psl': 'hPa'}
map_cmap = 'RdBu_r'


def lat_extent(maps, lat):
    """Latitude rows that contain data in any map, rounded outward to 5 degrees."""
    data_rows = np.any([np.isfinite(m).any(axis=1) for m in maps], axis=0)
    return [max(-90., 5 * np.floor(lat[data_rows].min() / 5)), min(90., 5 * np.ceil(lat[data_rows].max() / 5))]


def plot_trend_figure(npz, metrics, var, wlabel):
    methods = list(npz['methods'])
    datasets = list(npz['datasets'])
    panels = {(d, c): npz[f'{d}__truth' if c == 'truth' else f'{d}__{c}__mean']
              for d in datasets for c in ['truth'] + methods}
    scale = p95_abs(list(panels.values()))
    levels = np.linspace(-scale, scale, 11)

    lat0 = npz[f'{datasets[0]}__lat']
    extent = lat_extent(list(panels.values()), lat0)
    aspect = 360. / (extent[1] - extent[0])
    ncols = 1 + len(methods)
    col_w = 3.6
    fig, axes = plt.subplots(len(datasets), ncols, figsize=(col_w * ncols, len(datasets) * col_w / aspect + 1.3),
                             subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
                             gridspec_kw={'hspace': 0.06, 'wspace': 0.03}, squeeze=False)
    mappable = None
    for i, d in enumerate(datasets):
        lat, lon = npz[f'{d}__lat'], npz[f'{d}__lon']
        Lon, Lat = np.meshgrid(lon, lat)
        for j, c in enumerate(['truth'] + methods):
            ax = axes[i, j]
            m = ax.contourf(Lon, Lat, panels[(d, c)], levels=levels, cmap=map_cmap, extend='both',
                            transform=ccrs.PlateCarree())
            if mappable is None:
                mappable = m
            ax.set_extent([-180, 180] + extent, crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.5)
            if i == 0:
                ax.set_title('Truth (ESM ensemble mean)' if c == 'truth' else METHOD_NAMES_ON_FIG.get(c, c),
                             fontsize=11)
            if j == 0:
                ax.text(-0.04, 0.5, display_model_name(d), transform=ax.transAxes, rotation=90, ha='right',
                        va='center', fontsize=10)
                if d == obs:  # the reanalysis is not a forced response
                    ax.text(0.02, 0.04, 'raw (forced + internal)', transform=ax.transAxes, fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))
            if c != 'truth' and d != obs:
                (pooled,) = metrics[(metrics.model == d) & (metrics.method == c)
                                    & (metrics.member == 'pooled')].itertuples()
                ax.text(0.02, 0.04, f'r={pooled.corr:.2f}  nRMSE={pooled.rmse:.2f}',
                        transform=ax.transAxes, fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

    fmt = '%.2f' if scale < 1 else '%.1f'
    cbar = fig.colorbar(mappable, ax=axes, orientation='horizontal', fraction=0.04, pad=0.03, aspect=50, format=fmt)
    cbar.set_label(f'{CLIM_VARS_ON_FIG[var]} trend {wlabel} ({units[var]}/decade)', fontsize=11)
    return fig


if __name__ == '__main__':
    wlabel = window_label(trend_window())
    for lag in requested_lags([3]):
        for start_year, end_year in requested_year_ranges([(1850, 2014)]):
            year_suffix, run_suffix = run_suffixes(lag, (start_year, end_year))
            in_dir = results_dir(year_suffix, run_suffix)
            pdf_results_dir = os.path.join(pdf_root, f'results{year_suffix}{run_suffix}')
            os.makedirs(pdf_results_dir, exist_ok=True)

            for var in requested_clim_vars(['tas_ocean', 'psl', 'tas']):
                npz = dict(np.load(os.path.join(in_dir, f'trend_maps_{wlabel}_{var}.npz')))
                metrics = pd.read_csv(os.path.join(in_dir, f'trend_maps_{wlabel}_{var}_metrics.csv'))
                fig = plot_trend_figure(npz, metrics, var, wlabel)
                out_fig = os.path.join(pdf_results_dir, f'trend_maps_{wlabel}_{var}.pdf')
                fig.savefig(out_fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f'Saved: {out_fig}', flush=True)

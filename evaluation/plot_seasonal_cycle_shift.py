"""Seasonal-cycle change in 20CRv3: smoothed seasonal deviations in a box and DJF - JJA trend maps.

Also maps the DJF - JJA trend of each ESM's ensemble mean on the same scale.
Run: PBDMDC_LAGS=3 PBDMDC_YEAR_RANGES=1850-2014 python -m evaluation.plot_seasonal_cycle_shift
"""
import os

import cartopy.crs as ccrs
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.util import add_cyclic_point
from scipy.ndimage import gaussian_filter1d

from evaluation.utils.metrics import coord_to_dtindex, regional_mean
from evaluation.utils.style import display_model_name, set_font_sizes
from utils.data_utils import load_truth
from utils.params import artifact_root, base_path, pdf_root, requested_lags, requested_year_ranges, run_suffixes

# Prints at ~7/8/9/10 pt when included at \textwidth (figure saves ~15.9 in wide).
FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE = 18, 20, 23, 25
set_font_sizes(FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE)

obs = '20CRv3'
esms = ['CESM2', 'CanESM5', 'MIROC6', 'MPI-ESM1-2-LR']
panels = [('SAT N of 65°N', 'tas', 'n65'),
          ('SLP N Pacific', 'psl', 'npac')]
# (lat_min, lat_max, lon_min, lon_max); npac covers the negative DJF-JJA SLP trend of the North Pacific.
boxes = {'n65': (65, 90, 0, 360), 'npac': (25, 45, 135, 195)}
var_names = {'tas': 'SAT', 'psl': 'SLP'}
units = {'tas': 'K', 'psl': 'Pa'}  # regional means are in the data's units
smooth_sigma_years = 8  # years, as internal_smooth_sigma in plot_modes.py
per_century = 100.0
map_cmap = 'RdBu_r'
n_boot = 1000
block_years = 11
level = 99

season_hues = {'DJF': '#0072B2', 'MAM': '#009E73', 'JJA': '#D55E00', 'SON': '#CC79A7'}
season_months = {'DJF': [11, 0, 1], 'MAM': [2, 3, 4], 'JJA': [5, 6, 7], 'SON': [8, 9, 10]}


def by_month(x, time_index):
    x = np.ma.getdata(x).astype(float)
    time_index = pd.DatetimeIndex(time_index)
    assert time_index[0].month == 1 and len(x) % 12 == 0, 'series must start in January and cover full years'
    return np.asarray(time_index.year[::12]), x.reshape(-1, 12, *x.shape[1:])


def smooth(x):
    return gaussian_filter1d(x, sigma=smooth_sigma_years)


def contrast(X):
    return X[:, season_months['DJF']].mean(axis=1) - X[:, season_months['JJA']].mean(axis=1)


def ols_trend(years, Y):
    t = years - years.mean()
    Yc = Y - Y.mean(axis=0)
    slope = np.tensordot(t, Yc, axes=(0, 0)) / (t @ t)
    resid = Yc - np.multiply.outer(t, slope)
    se = np.sqrt((resid ** 2).sum(axis=0) / (len(t) - 2) / (t @ t))
    return slope * per_century, se * per_century


def block_bootstrap_significance(years, C, trend, block_years=block_years):
    """True where |trend| exceeds the level-th percentile of trends of block-bootstrapped detrended series.

    DJF trend - JJA trend is the trend of C = DJF - JJA, so one trend per surrogate suffices.
    """
    rng = np.random.default_rng(0)
    n = len(years)
    assert n % block_years == 0, f'{n} years do not split into {block_years}-year blocks'
    t = years - years.mean()
    Cc = (C - C.mean(axis=0)).reshape(n, -1)
    resid = Cc - np.outer(t, trend.reshape(-1) / per_century)
    starts = rng.integers(0, n - block_years + 1, size=(n_boot, n // block_years))
    idx = (starts[:, :, None] + np.arange(block_years)).reshape(n_boot, -1)
    null = np.stack([t @ resid[rows] for rows in idx]) / (t @ t) * per_century
    return np.abs(trend) > np.percentile(np.abs(null), level, axis=0).reshape(trend.shape)


def box_stats(lat, lon, trend, significant, box):
    lat0, lat1, lon0, lon1 = box
    rows = (lat >= lat0) & (lat <= lat1)
    cols = (lon % 360 >= lon0) & (lon % 360 <= lon1)
    tr, sig = trend[rows][:, cols], significant[rows][:, cols]
    w = np.cos(np.deg2rad(lat[rows]))[:, None] * np.ones_like(tr)
    mean = np.nansum(w * tr) / np.sum(w[np.isfinite(tr)])
    return mean, np.mean(sig & (np.sign(tr) == np.sign(mean)))


def box_mean(truth, lat, lon, box):
    lat0, lat1, lon0, lon1 = box
    sel = (lon % 360 >= lon0) & (lon % 360 <= lon1)
    return np.ma.getdata(regional_mean(truth.to_masked_array()[..., sel], lat, lat0, lat1)).astype(float)


def unscale(field, lat):
    coef = np.sqrt(np.cos(np.deg2rad(lat)))
    coef = np.where(coef > 1e-3, coef, np.nan)
    return field / coef[:, None]


def obs_contrast_trend_map(var, year_suffix, model):
    truth, lat = load_truth(var, model, os.path.join(artifact_root, f'means{year_suffix}'), return_lats=True)
    lon = truth['lon'].values
    years, X = by_month(unscale(truth.values, lat), coord_to_dtindex(truth.time))
    C = contrast(X)
    trend = ols_trend(years, C)[0]
    return lat, lon, trend, block_bootstrap_significance(years, C, trend)


def member_files(var, model):
    input_dir = os.path.join(base_path, 'ensembles', var, model)
    files = [f for f in sorted(os.listdir(input_dir)) if 'historical' in f and (model != 'CESM2' or '_cmip6' in f)]
    return [os.path.join(input_dir, f) for f in files]


def seasonal_deviation(X):
    dev = X - X.mean(axis=1, keepdims=True)
    return {season: dev[:, months].mean(axis=1) for season, months in season_months.items()}


def draw_deviation(ax, years, dev):
    smoothed = {season: smooth(x) for season, x in dev.items()}
    for season in dev:
        ax.plot(years, smoothed[season], color=season_hues[season], linewidth=2.0, zorder=3)
    ax.axhline(0, color='0.5', linewidth=0.6)
    ax.set_xlim(years[0], years[-1])
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return smoothed


def pattern_corr(a, b, lat):
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], a.shape)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, w = a[ok], b[ok], w[ok]
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))


def draw_map(ax, lat, lon, field, levels, region, stipple=None, outline=True):
    field_c, lon_c = add_cyclic_point(field, coord=lon)
    m = ax.contourf(lon_c, lat, field_c, levels=levels, cmap=map_cmap, extend='both', transform=ccrs.PlateCarree())
    if stipple is not None:
        stipple_c, _ = add_cyclic_point(stipple.astype(float), coord=lon)
        ax.contourf(lon_c, lat, stipple_c, levels=[0.5, 1.5], colors='none', hatches=['..'],
                    transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    if outline:
        lat0, lat1, lon0, lon1 = boxes[region]
        if lon1 - lon0 >= 360:
            lons, lats = np.linspace(-180, 180, 361), np.full(361, lat0 if lat1 == 90 else lat1)
        else:
            lons, lats = [lon0, lon1, lon1, lon0, lon0], [lat0, lat0, lat1, lat1, lat0]
        ax.plot(lons, lats, color='black', linestyle='--', linewidth=1.0, transform=ccrs.PlateCarree())
    ax.set_global()
    return m


def nice_levels(field):
    scale = max(np.nanpercentile(np.abs(field), 95), 1e-12)
    step = scale / 5
    mag = 10 ** np.floor(np.log10(step))
    step = mag * min(s for s in (1, 2, 5, 10) if s * mag >= step)
    return step * np.arange(-5, 6)


def plot_obs(year_suffix):
    """One row per panel: 20CRv3 smoothed seasonal deviations in the box and the DJF - JJA trend map."""
    plt.rcParams['hatch.linewidth'] = 0.3
    fig = plt.figure(figsize=(14.5, 9), layout='constrained')
    gs = fig.add_gridspec(len(panels), 2, width_ratios=[1, 1.3])
    proj = ccrs.PlateCarree(central_longitude=180)
    summary, levels = [], {}
    for i, (label, var, region) in enumerate(panels):
        truth, lat = load_truth(var, obs, os.path.join(artifact_root, f'means{year_suffix}'), return_lats=True)
        raw = box_mean(truth, lat, truth['lon'].values, boxes[region])
        years, X = by_month(raw, coord_to_dtindex(truth.time))
        ax = fig.add_subplot(gs[i, 0])
        draw_deviation(ax, years, seasonal_deviation(X))
        ax.set_ylabel(f'{label}:\ndeviation ({units[var]})', fontsize=FS_LABEL)
        if i == 0:
            ax.set_title(f'{obs} seasonal deviation', fontsize=FS_TITLE)

        lat_o, lon_o, map_obs, stip_obs = obs_contrast_trend_map(var, year_suffix, obs)
        levels[var] = nice_levels(map_obs)
        ax_map = fig.add_subplot(gs[i, 1], projection=proj)
        m = draw_map(ax_map, lat_o, lon_o, map_obs, levels[var], region, stip_obs)
        if i == 0:
            ax_map.set_title(f'{obs} DJF − JJA trend', fontsize=FS_TITLE)
        cbar = fig.colorbar(m, ax=ax_map, orientation='horizontal', fraction=0.06, aspect=40, format='%g')
        cbar.set_label(f'{var_names[var]} DJF − JJA trend, {years[0]}–{years[-1]} ({units[var]}/century)',
                       fontsize=FS_LABEL)

        row = {'index': label}
        w = np.cos(np.deg2rad(lat_o))[:, None] * np.ones_like(map_obs)
        row['stippled_area'] = np.sum(w * stip_obs) / np.sum(w)
        row['box_mean_trend'], row['box_frac_significant_same_sign'] = box_stats(lat_o, lon_o, map_obs, stip_obs,
                                                                                 boxes[region])
        print(f'{label} box {boxes[region]}: mean trend {row["box_mean_trend"]:.3g} {units[var]}/century, '
              f'{row["box_frac_significant_same_sign"]:.0%} of cells significant with the same sign', flush=True)
        summary.append(row)

    handles = [mlines.Line2D([], [], color=c, linewidth=2.0, label=s) for s, c in season_hues.items()]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=FS_TICK, frameon=False, bbox_to_anchor=(0.5, 0.0),
               title=f'Season (smoothed, σ = {smooth_sigma_years} yr); stippling: |trend| exceeds {level}% of '
                     f'{n_boot} block-bootstrap ({block_years}-yr) null trends', title_fontsize=FS_TICK)
    return fig, summary, levels


def plot_esm_ensmean(year_suffix, levels):
    """DJF - JJA trend maps of 20CRv3 and of each ESM's ensemble mean."""
    plt.rcParams['hatch.linewidth'] = 0.3
    fig = plt.figure(figsize=(16, 13), layout='constrained')
    gs = fig.add_gridspec(4, 3)
    proj = ccrs.PlateCarree(central_longitude=180)
    for i, (label, var, region) in enumerate(panels):
        box = boxes[region]
        lat, lon, map_obs, stip_obs = obs_contrast_trend_map(var, year_suffix, obs)
        no_stip = np.zeros(map_obs.shape, dtype=bool)
        ax = fig.add_subplot(gs[2 * i, 0], projection=proj)
        m = draw_map(ax, lat, lon, map_obs, levels[var], region, stip_obs, outline=False)
        ax.set_title(f'{obs}\nbox {box_stats(lat, lon, map_obs, stip_obs, box)[0]:.3g} {units[var]}/century',
                     fontsize=FS_LABEL)
        for j, model in enumerate(esms, start=1):
            lat_e, lon_e, trend, _ = obs_contrast_trend_map(var, year_suffix, model)
            assert np.allclose(lat_e, lat) and np.allclose(lon_e, lon), f'{var} {model}: grid differs from {obs}'
            r = pattern_corr(map_obs, trend, lat)
            ax = fig.add_subplot(gs[2 * i + j // 3, j % 3], projection=proj)
            draw_map(ax, lat, lon, trend, levels[var], region, outline=False)
            ax.set_title(f'{display_model_name(model)} (n = {len(member_files(var, model))})\n'
                         f'r = {r:.2f}, box {box_stats(lat, lon, trend, no_stip, box)[0]:.3g} {units[var]}/century',
                         fontsize=FS_LABEL)
            print(f'{var} {model}: ensemble-mean r {r:.2f}', flush=True)
        ax = fig.add_subplot(gs[2 * i + 1, 2])
        ax.axis('off')
        cbar = fig.colorbar(m, cax=ax.inset_axes([0.05, 0.55, 0.9, 0.08]), orientation='horizontal', format='%g')
        cbar.set_label(f'{var_names[var]} DJF − JJA trend ({units[var]}/century)', fontsize=FS_LABEL)
        ax.text(0.5, 0.3, f'ESMs: ensemble mean\nstippling ({obs} only): significant at {level}%',
                transform=ax.transAxes, ha='center', va='top', fontsize=FS_TICK)
    return fig


if __name__ == '__main__':
    for lag in requested_lags([3]):
        for start_year, end_year in requested_year_ranges([(1850, 2014)]):
            year_suffix, run_suffix = run_suffixes(lag, (start_year, end_year))
            out_dir = os.path.join(pdf_root, f'results{year_suffix}{run_suffix}')
            os.makedirs(out_dir, exist_ok=True)

            fig, summary, levels = plot_obs(year_suffix)
            out_fig = os.path.join(out_dir, f'seasonal_cycle_drift_{obs}.pdf')
            fig.savefig(out_fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved: {out_fig}', flush=True)
            pd.DataFrame(summary).to_csv(os.path.join(out_dir, f'seasonal_cycle_drift_{obs}_summary.csv'),
                                         index=False)

            fig = plot_esm_ensmean(year_suffix, levels)
            out_fig = os.path.join(out_dir, 'seasonal_cycle_drift_esm_ensmean.pdf')
            fig.savefig(out_fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved: {out_fig}', flush=True)

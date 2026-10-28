"""Seasonal-cycle change in 20CRv3 and in the internal part of PullbackDMDc modes 3-4 (annual-period pair).

The forced estimate has no seasonal cycle, so seasonal-cycle changes end up in the internal component.
Run: PBDMDC_LAGS=3 PBDMDC_YEAR_RANGES=1850-2014 python -m evaluation.plot_seasonal_cycle_shift
"""
import os
import pickle as pkl

import cartopy.crs as ccrs
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.util import add_cyclic_point
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from scipy.ndimage import gaussian_filter1d

from evaluation.plot_gm_timeseries import units
from evaluation.utils.metrics import coord_to_dtindex, lat_flat_from_pars, regional_mean, regional_mean_flat
from utils.data_utils import load_truth, undo_lat_scaling
from utils.params import (artifact_root, pdf_root, predictions_dir, regions, requested_lags,
                          requested_year_ranges, run_suffixes)

obs = '20CRv3'
panels = [('SAT N of 65°N', 'tas', 'n65', 'PullbackDMDc-3d'),
          ('SLP S of 65°S', 'psl', 's65', 'PullbackDMDc-2d')]
var_names = {'tas': 'SAT', 'psl': 'SLP'}
mode_ks = [2, 3]  # modes 3 and 4
smooth_sigma_years = 8  # as internal_smooth_sigma in plot_modes.py
per_century = 100.0
map_cmap = 'RdBu_r'

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


def load_fit(var, method, year_suffix, run_suffix):
    model_dir = os.path.join(predictions_dir(year_suffix, run_suffix), var, obs)
    files = [f for f in os.listdir(model_dir) if f.split('_')[-1].split('.')[0] == method]
    assert len(files) == 1, f'{model_dir}: expected one {method} file, got {files}'
    with open(os.path.join(model_dir, files[0]), 'rb') as f:
        dmd = pkl.load(f)
    eof_file = os.path.join(artifact_root, f'eofs{year_suffix}', var, obs, files[0].replace('_' + method, ''))
    with open(eof_file, 'rb') as f:
        pars = pkl.load(f)['data_pars']
    return dmd, pars


def components(var, region, dmd, pars, year_suffix):
    """Regional mean of the raw reanalysis and of the forced estimate, as in compute_gm_timeseries."""
    truth, lat = load_truth(var, obs, os.path.join(artifact_root, f'means{year_suffix}'), return_lats=True)
    raw = np.ma.getdata(regional_mean(truth, lat, *regions[region])).astype(float)
    # Same reconstruction as load_truncated_prediction.
    svals = np.sqrt((dmd.eofs ** 2).sum(axis=-1))
    recon = (dmd.fr_pcs.real * svals) @ (dmd.eofs / svals[:, None])
    forced = regional_mean_flat(recon, pars['resc_coef_flat'], lat_flat_from_pars(pars), *regions[region])
    return coord_to_dtindex(truth.time), {'raw': raw, 'forced': forced, 'internal': raw - forced}


def unscale(field, lat):
    coef = np.sqrt(np.cos(np.deg2rad(lat)))
    coef = np.where(coef > 1e-3, coef, np.nan)
    return field / coef[:, None]


def obs_contrast_trend_map(var, year_suffix):
    truth, lat = load_truth(var, obs, os.path.join(artifact_root, f'means{year_suffix}'), return_lats=True)
    lon = truth['lon'].values
    years, X = by_month(unscale(truth.values, lat), coord_to_dtindex(truth.time))
    return lat, lon, ols_trend(years, contrast(X))[0]


def modes34(var, region, method, dmd, pars, time_index):
    dmd.compute_modes()
    rotated = dmd.compute_rotated_modes(top_n=max(mode_ks) + 1)
    labels = [rotated['labels'][k] for k in mode_ks]
    assert all(lab == (f'{mode_ks[0] + 1},{mode_ks[1] + 1}', 'complex') for lab in labels), labels
    lam = rotated['eigvals'][mode_ks[0]]
    print(f'{var} {method} modes 3-4: {labels}, freq {6 * abs(np.angle(lam) / dmd.lag) / np.pi:.3f}/y', flush=True)

    patterns = rotated['spatial_patterns'][:, mode_ks]
    ts = rotated['internal_time_series'][:, mode_ks]
    # The sum over the pair is real and does not depend on its PCA rotation.
    recon = ts @ patterns.T
    series = regional_mean_flat(recon, pars['resc_coef_flat'], lat_flat_from_pars(pars), *regions[region])

    years, TS = by_month(ts, time_index)
    b = ols_trend(years, contrast(TS))[0]
    map_flat = patterns @ b
    idx = np.linspace(0, recon.shape[1] - 1, 7).astype(int)
    slow = ols_trend(years, contrast(by_month(recon[:, idx], time_index)[1]))[0]
    assert np.allclose(slow, map_flat[idx], rtol=1e-6, atol=1e-10 * np.abs(map_flat).max()), (slow, map_flat[idx])

    lat, lon = pars['lat'].values, pars['lon'].values
    trend_map = undo_lat_scaling(map_flat[:, None], lat, lon.size, nan_mask=pars['nan_mask'])[:, :, 0]
    trend_map[np.cos(np.deg2rad(lat)) < 1e-6] = np.nan
    return series, lat, lon, trend_map


def seasonal_deviation(X):
    dev = X - X.mean(axis=1, keepdims=True)
    return {season: dev[:, months].mean(axis=1) for season, months in season_months.items()}


def draw_deviation(ax, years, dev):
    smoothed = {season: smooth(x) for season, x in dev.items()}
    for season, x in smoothed.items():
        ax.plot(years, x, color=season_hues[season], linewidth=2.0)
    ax.axhline(0, color='0.5', linewidth=0.6)
    ax.set_xlim(years[0], years[-1])
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return smoothed


def annotate_r(ax, r):
    parts = [TextArea('r', textprops=dict(fontsize=9))]
    parts += [TextArea(f'{s} {v:.2f}', textprops=dict(color=season_hues[s], fontsize=9, fontweight='bold'))
              for s, v in r.items()]
    box = AnchoredOffsetbox(loc='upper left', child=HPacker(children=parts, pad=0, sep=6), pad=0.3, borderpad=0.4,
                            frameon=True)
    box.patch.set(facecolor='white', alpha=0.85, edgecolor='none')
    ax.add_artist(box)


def pattern_corr(a, b, lat):
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], a.shape)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, w = a[ok], b[ok], w[ok]
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))


def draw_map(ax, lat, lon, field, levels, region):
    field_c, lon_c = add_cyclic_point(field, coord=lon)
    m = ax.contourf(lon_c, lat, field_c, levels=levels, cmap=map_cmap, extend='both', transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    edge = regions[region][0] if region == 'n65' else regions[region][1]
    ax.plot(np.linspace(-180, 180, 361), np.full(361, edge), color='black', linestyle='--', linewidth=1.0,
            transform=ccrs.PlateCarree())
    ax.set_global()
    return m


def plot_drift(year_suffix, run_suffix):
    fig = plt.figure(figsize=(12, 13.5), layout='constrained')
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1.2, 1, 1.2])
    summary = []
    for i, (label, var, region, method) in enumerate(panels):
        dmd, pars = load_fit(var, method, year_suffix, run_suffix)
        time_index, comps = components(var, region, dmd, pars, year_suffix)
        series, lat_m, lon_m, map_modes = modes34(var, region, method, dmd, pars, time_index)
        assert series.shape == comps['raw'].shape
        comps['modes34'] = series
        by = {key: by_month(x, time_index) for key, x in comps.items()}

        ax_obs = fig.add_subplot(gs[2 * i, 0])
        ax_mod = fig.add_subplot(gs[2 * i, 1], sharex=ax_obs, sharey=ax_obs)
        dev_obs, dev_mod = seasonal_deviation(by['raw'][1]), seasonal_deviation(by['modes34'][1])
        sm_obs = draw_deviation(ax_obs, by['raw'][0], dev_obs)
        sm_mod = draw_deviation(ax_mod, by['modes34'][0], dev_mod)
        r_smooth = {s: np.corrcoef(sm_obs[s], sm_mod[s])[0, 1] for s in season_months}
        r_yearly = {s: np.corrcoef(dev_obs[s], dev_mod[s])[0, 1] for s in season_months}
        annotate_r(ax_mod, r_smooth)
        ax_obs.set_ylabel(f'{label}\ndeviation from annual mean ({units[var]})\nGaussian smoothing, σ = {smooth_sigma_years} yr',
                          fontsize=10)

        lat_o, lon_o, map_obs = obs_contrast_trend_map(var, year_suffix)
        assert np.allclose(lat_o, lat_m) and np.allclose(lon_o, lon_m), f'{var}: means and EOF grids differ'
        scale = max(np.nanpercentile(np.abs(np.concatenate([map_obs.ravel(), map_modes.ravel()])), 95), 1e-12)
        step = scale / 5
        mag = 10 ** np.floor(np.log10(step))
        step = mag * min(s for s in (1, 2, 5, 10) if s * mag >= step)
        levels = step * np.arange(-5, 6)
        proj = ccrs.PlateCarree(central_longitude=180)
        map_axes = [fig.add_subplot(gs[2 * i + 1, j], projection=proj) for j in range(2)]
        m = draw_map(map_axes[0], lat_o, lon_o, map_obs, levels, region)
        draw_map(map_axes[1], lat_m, lon_m, map_modes, levels, region)
        r_pattern = pattern_corr(map_obs, map_modes, lat_o)
        map_axes[1].text(0.02, 0.04, f'pattern r = {r_pattern:.2f}', transform=map_axes[1].transAxes, fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5))
        map_axes[0].text(-0.04, 0.5, f'{var_names[var]}: DJF − JJA trend', transform=map_axes[0].transAxes,
                         rotation=90, ha='right', va='center', fontsize=10)
        cbar = fig.colorbar(m, ax=map_axes, orientation='horizontal', fraction=0.06, aspect=50,
                            format='%g')
        cbar.set_label(f'{var_names[var]} DJF − JJA trend, {by["raw"][0][0]}–{by["raw"][0][-1]} '
                       f'({units[var]}/century)', fontsize=9)
        if i == 0:
            ax_obs.set_title(obs, fontsize=12)
            ax_mod.set_title('PullbackDMDc internal, modes 3–4', fontsize=12)

        trends = {key: ols_trend(*by[key])[0] for key in ['forced', 'internal', 'modes34']}
        a = trends['internal'] - trends['internal'].mean()
        b = trends['modes34'] - trends['modes34'].mean()
        row = {'index': label, 'method': method, 'trend_profile_corr': np.corrcoef(a, b)[0, 1],
               'trend_profile_regression_slope': (a @ b) / (a @ a), 'forced_trend_spread': np.ptp(trends['forced']),
               'pattern_r': r_pattern}
        row.update({f'r_{s}_smoothed': r_smooth[s] for s in season_months})
        row.update({f'r_{s}_yearly': r_yearly[s] for s in season_months})
        summary.append(row)
        print(f'{label}: r smoothed {np.round(list(r_smooth.values()), 2)}, r yearly '
              f'{np.round(list(r_yearly.values()), 2)} (DJF MAM JJA SON); pattern r {r_pattern:.2f}; '
              f'trend-profile corr {row["trend_profile_corr"]:.2f}', flush=True)

    handles = [mlines.Line2D([], [], color=c, linewidth=2.0, label=s) for s, c in season_hues.items()]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0),
               title='Season', title_fontsize=9)
    return fig, pd.DataFrame(summary)


if __name__ == '__main__':
    for lag in requested_lags([3]):
        for start_year, end_year in requested_year_ranges([(1850, 2014)]):
            year_suffix, run_suffix = run_suffixes(lag, (start_year, end_year))
            out_dir = os.path.join(pdf_root, f'results{year_suffix}{run_suffix}')
            os.makedirs(out_dir, exist_ok=True)

            fig, summary = plot_drift(year_suffix, run_suffix)
            out_fig = os.path.join(out_dir, f'seasonal_cycle_drift_{obs}.pdf')
            fig.savefig(out_fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            summary.to_csv(os.path.join(out_dir, f'seasonal_cycle_drift_{obs}_summary.csv'), index=False)
            print(f'Saved: {out_fig}', flush=True)

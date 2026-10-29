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

from evaluation.utils.metrics import coord_to_dtindex, global_mean_flat, lat_flat_from_pars, regional_mean
from utils.data_utils import load_truth, undo_lat_scaling
from utils.params import (artifact_root, pdf_root, predictions_dir, requested_lags, requested_year_ranges,
                          run_suffixes)

obs = '20CRv3'
panels = [('SAT N of 65°N', 'tas', 'n65', 'PullbackDMDc-3d'),
          ('SLP N Pacific', 'psl', 'npac', 'PullbackDMDc-2d')]
# (lat_min, lat_max, lon_min, lon_max); npac covers the negative DJF-JJA SLP trend of the North Pacific.
boxes = {'n65': (65, 90, 0, 360), 'npac': (25, 45, 135, 195)}
var_names = {'tas': 'SAT', 'psl': 'SLP'}
units = {'tas': 'K', 'psl': 'Pa'}  # regional means are in the data's units
mode_ks = [2, 3]  # modes 3 and 4
smooth_sigma_years = 8  # as internal_smooth_sigma in plot_modes.py
per_century = 100.0
map_cmap = 'RdBu_r'
n_perm = 1000

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


def permutation_significance(years, C, trend):
    """True where |trend| exceeds 95% of the trends with permuted years.

    DJF trend - JJA trend is the trend of C = DJF - JJA, so one trend per permutation suffices.
    """
    rng = np.random.default_rng(0)
    n = len(years)
    t = years - years.mean()
    perms = np.stack([rng.permutation(n) for _ in range(n_perm)])
    Cc = (C - C.mean(axis=0)).reshape(n, -1)
    null = (t[perms] @ Cc) / (t @ t) * per_century
    return np.abs(trend) > np.percentile(np.abs(null), 95, axis=0).reshape(trend.shape)


def box_stats(lat, lon, trend, significant, box):
    lat0, lat1, lon0, lon1 = box
    rows = (lat >= lat0) & (lat <= lat1)
    cols = (lon % 360 >= lon0) & (lon % 360 <= lon1)
    tr, sig = trend[rows][:, cols], significant[rows][:, cols]
    w = np.cos(np.deg2rad(lat[rows]))[:, None] * np.ones_like(tr)
    mean = np.nansum(w * tr) / np.sum(w[np.isfinite(tr)])
    return mean, np.mean(sig & (np.sign(tr) == np.sign(mean)))


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
    raw = box_mean(truth, lat, truth['lon'].values, boxes[region])
    # Same reconstruction as load_truncated_prediction.
    svals = np.sqrt((dmd.eofs ** 2).sum(axis=-1))
    recon = (dmd.fr_pcs.real * svals) @ (dmd.eofs / svals[:, None])
    forced = box_mean_flat(recon, pars, boxes[region])
    return coord_to_dtindex(truth.time), {'raw': raw, 'forced': forced, 'internal': raw - forced}


def box_mean(truth, lat, lon, box):
    lat0, lat1, lon0, lon1 = box
    sel = (lon % 360 >= lon0) & (lon % 360 <= lon1)
    return np.ma.getdata(regional_mean(truth.to_masked_array()[..., sel], lat, lat0, lat1)).astype(float)


def box_mean_flat(data_flat, pars, box):
    lat0, lat1, lon0, lon1 = box
    lat_flat = lat_flat_from_pars(pars)
    lon_flat = np.repeat(pars['lon'].values[None, :], pars['lat'].size, axis=0)[~pars['nan_mask']] % 360
    sel = (lat_flat >= lat0) & (lat_flat <= lat1) & (lon_flat >= lon0) & (lon_flat <= lon1)
    assert sel.any(), f'No grid points in {box}'
    return global_mean_flat(data_flat[..., sel], pars['resc_coef_flat'][sel])


def unscale(field, lat):
    coef = np.sqrt(np.cos(np.deg2rad(lat)))
    coef = np.where(coef > 1e-3, coef, np.nan)
    return field / coef[:, None]


def obs_contrast_trend_map(var, year_suffix):
    truth, lat = load_truth(var, obs, os.path.join(artifact_root, f'means{year_suffix}'), return_lats=True)
    lon = truth['lon'].values
    years, X = by_month(unscale(truth.values, lat), coord_to_dtindex(truth.time))
    C = contrast(X)
    trend = ols_trend(years, C)[0]
    return lat, lon, trend, permutation_significance(years, C, trend)


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
    series = box_mean_flat(recon, pars, boxes[region])

    years, TS = by_month(ts, time_index)
    map_flat = patterns @ ols_trend(years, contrast(TS))[0]
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
    for season in dev:
        ax.plot(years, dev[season], color=season_hues[season], linewidth=0.6, alpha=0.3, zorder=1)
        ax.plot(years, smoothed[season], color=season_hues[season], linewidth=2.0, zorder=3)
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


def draw_map(ax, lat, lon, field, levels, region, stipple=None):
    field_c, lon_c = add_cyclic_point(field, coord=lon)
    m = ax.contourf(lon_c, lat, field_c, levels=levels, cmap=map_cmap, extend='both', transform=ccrs.PlateCarree())
    if stipple is not None:
        stipple_c, _ = add_cyclic_point(stipple.astype(float), coord=lon)
        ax.contourf(lon_c, lat, stipple_c, levels=[0.5, 1.5], colors='none', hatches=['..'],
                    transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    lat0, lat1, lon0, lon1 = boxes[region]
    if lon1 - lon0 >= 360:
        lons, lats = np.linspace(-180, 180, 361), np.full(361, lat0 if lat1 == 90 else lat1)
    else:
        lons, lats = [lon0, lon1, lon1, lon0, lon0], [lat0, lat0, lat1, lat1, lat0]
    ax.plot(lons, lats, color='black', linestyle='--', linewidth=1.0, transform=ccrs.PlateCarree())
    ax.set_global()
    return m


def plot_drift(year_suffix, run_suffix):
    plt.rcParams['hatch.linewidth'] = 0.3
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
        yearly = np.abs(np.ravel(list(dev_obs.values())))
        smoothed = np.abs(np.concatenate([np.ravel(list(d.values())) for d in (sm_obs, sm_mod)]))
        lim = max(np.percentile(yearly, 98), 1.1 * smoothed.max())
        ax_obs.set_ylim(-lim, lim)
        r_smooth = {s: np.corrcoef(sm_obs[s], sm_mod[s])[0, 1] for s in season_months}
        r_yearly = {s: np.corrcoef(dev_obs[s], dev_mod[s])[0, 1] for s in season_months}
        annotate_r(ax_mod, r_smooth)
        ax_obs.set_ylabel(f'{label}\ndeviation from annual mean ({units[var]})\nthick: Gaussian σ = {smooth_sigma_years} yr, thin: yearly',
                          fontsize=10)

        lat_o, lon_o, map_obs, stip_obs = obs_contrast_trend_map(var, year_suffix)
        assert np.allclose(lat_o, lat_m) and np.allclose(lon_o, lon_m), f'{var}: means and EOF grids differ'
        scale = max(np.nanpercentile(np.abs(np.concatenate([map_obs.ravel(), map_modes.ravel()])), 95), 1e-12)
        step = scale / 5
        mag = 10 ** np.floor(np.log10(step))
        step = mag * min(s for s in (1, 2, 5, 10) if s * mag >= step)
        levels = step * np.arange(-5, 6)
        proj = ccrs.PlateCarree(central_longitude=180)
        map_axes = [fig.add_subplot(gs[2 * i + 1, j], projection=proj) for j in range(2)]
        m = draw_map(map_axes[0], lat_o, lon_o, map_obs, levels, region, stip_obs)
        draw_map(map_axes[1], lat_m, lon_m, map_modes, levels, region)
        r_pattern = pattern_corr(map_obs, map_modes, lat_o)
        map_axes[1].text(0.02, 0.04, f'pattern r = {r_pattern:.2f}', transform=map_axes[1].transAxes, fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5))
        map_axes[0].text(-0.04, 0.5, f'{var_names[var]}: DJF − JJA trend', transform=map_axes[0].transAxes,
                         rotation=90, ha='right', va='center', fontsize=10)
        cbar = fig.colorbar(m, ax=map_axes, orientation='horizontal', fraction=0.06, aspect=50,
                            format='%g')
        cbar.set_label(f'{var_names[var]} DJF − JJA trend, {by["raw"][0][0]}–{by["raw"][0][-1]} '
                       f'({units[var]}/century); stippling: |{obs} trend| exceeds 95% of {n_perm} year-permuted trends',
                       fontsize=9)
        if i == 0:
            ax_obs.set_title(obs, fontsize=12)
            ax_mod.set_title('PullbackDMDc internal, modes 3–4', fontsize=12)

        trends = {key: ols_trend(*by[key])[0] for key in ['forced', 'internal', 'modes34']}
        a = trends['internal'] - trends['internal'].mean()
        b = trends['modes34'] - trends['modes34'].mean()
        row = {'index': label, 'method': method, 'trend_profile_corr': np.corrcoef(a, b)[0, 1],
               'trend_profile_regression_slope': (a @ b) / (a @ a), 'forced_trend_spread': np.ptp(trends['forced']),
               'pattern_r': r_pattern}
        w = np.cos(np.deg2rad(lat_o))[:, None] * np.ones_like(map_obs)
        row['stippled_area'] = np.sum(w * stip_obs) / np.sum(w)
        row['box_mean_trend'], row['box_frac_significant_same_sign'] = box_stats(lat_o, lon_o, map_obs, stip_obs,
                                                                                 boxes[region])
        print(f'{label} box {boxes[region]}: mean trend {row["box_mean_trend"]:.3g} {units[var]}/century, '
              f'{row["box_frac_significant_same_sign"]:.0%} of cells significant with the same sign', flush=True)
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

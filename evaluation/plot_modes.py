import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from utils.data_utils import undo_lat_scaling
from evaluation.utils.members import SELECTED_MEMBERS, member_id
from evaluation.utils.style import TIMESCALE_BANDS, p95_abs, set_font_sizes
from evaluation.utils.timescales import acf, acf_bands, eig_decay_time_years, eig_frequency_per_year, zero_crossing_time
from utils.params import artifact_root, lag_suffix_for, pdf_root, predictions_dir, year_suffix_for
from scipy.ndimage import gaussian_filter1d
import matplotlib.gridspec as gridspec

# Prints at ~7/8/9/10 pt when included at \textwidth (figure saves ~13.6-14.9 in wide).
FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE = 17, 18, 21, 23
set_font_sizes(FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE)


def inner_yticks(ax, nbins=4, margin=0.08):
    """Y ticks strictly inside the limits, so the end labels of vertically stacked panels do not clash."""
    lo, hi = ax.get_ylim()
    pad = margin * (hi - lo)
    ax.set_yticks([t for t in MaxNLocator(nbins=nbins).tick_values(lo, hi) if lo + pad <= t <= hi - pad])

# Parameters for selection
lag=3
start_year, end_year = 1850, 2014
lag_suffix, year_suffix = lag_suffix_for(lag), year_suffix_for((start_year, end_year))

clim_vars = ['tas', 'tas_ocean', 'psl']

chosen_model = '20CRv3'
chosen_member = ''
# (dataset, member) shown per figure; members are defined once in evaluation/utils/members.py.
esm_list = list(SELECTED_MEMBERS.items())

# Patterns are multiplied by the std of the mode time series, so they are in data units per 1 sigma.
cbar_labels = {'tas': 'K per 1σ of mode time series',
               'tas_ocean': 'K per 1σ of mode time series',
               'psl': 'Pa per 1σ of mode time series'}


results_dir = os.path.join(pdf_root, f'results{year_suffix}{lag_suffix}')


eofs_base_path = os.path.join(artifact_root, f'eofs{year_suffix}')
base_pred_dir = predictions_dir(year_suffix, lag_suffix)

top_n = 4
map_cmap = 'RdBu_r'
internal_smooth_sigma = 12 * 8 # eight years
#acf_length = [100 * 12] * 2 + [20*12] * (top_n-2)

def load_mode_rows(clim_var, chosen_model, chosen_member, chosen_method):
    """Load the DMD fit for one dataset and compute the top-mode maps, time series and ACFs."""
    # --------------------------------------------------
    # Load DMD object for the chosen model/member
    # --------------------------------------------------
    model_dir = os.path.join(base_pred_dir, clim_var, chosen_model)
    eof_dir = os.path.join(eofs_base_path, clim_var, chosen_model)

    matched_dmd_file = None
    eof_file = None

    for f_name in os.listdir(model_dir):
        method = f_name.split('_')[-1].split('.')[0]
        if method != chosen_method:
            continue
        if member_id(f_name) == chosen_member:
            matched_dmd_file = os.path.join(model_dir, f_name)
            eof_file = os.path.join(eof_dir, f_name.replace('_' + chosen_method, ''))
            break

    if matched_dmd_file is None or eof_file is None:
        print(
            f'Skipping {clim_var} / {chosen_model}: '
            f'no file for method={chosen_method} member={chosen_member!r}'
        )
        return None

    with open(matched_dmd_file, 'rb') as f:
        dmd = pickle.load(f)

    with open(eof_file, 'rb') as f:
        data_parameters = pickle.load(f)['data_pars']
        nan_mask = data_parameters['nan_mask']
        lat = data_parameters['lat'].values
        lon = data_parameters['lon'].values

    dmd.compute_modes()
    rotated = dmd.compute_rotated_modes(top_n=top_n)

    eigvals = rotated['eigvals']
    modes = rotated['spatial_patterns']
    ts_iv = rotated['internal_time_series']
    ts_fr = rotated['forced_time_series']
    labels = rotated['labels']

    # R2-L432: variance of the modes 1-2 forced reconstruction relative to the full forced response
    # (sqrt(cos lat)-weighted, time mean removed). Modes are not orthogonal, so the ratio is not bounded by 1.
    forced = np.real(dmd.fr_pcs) @ dmd.eofs
    forced_12 = ts_fr[:, :2] @ modes[:, :2].T
    forced_var_ratio_12 = (np.sum((forced_12 - forced_12.mean(axis=0)) ** 2)
                           / np.sum((forced - forced.mean(axis=0)) ** 2))
    # Variance of each orange (forced) curve as plotted: forced share of that mode's own time series.
    forced_fraction = np.var(ts_fr, axis=0) / np.var(ts_fr + ts_iv, axis=0)
    nrows = min(top_n, eigvals.size)

    decay_times = eig_decay_time_years(eigvals, dmd.lag)
    frequencies = eig_frequency_per_year(eigvals, dmd.lag)

    # First pass: compute all data
    modes_data = []
    ts_data = []
    acf_data = []
    row_labels = []

    check_sum=0.
    pair_counts = {}
    for k in range(nrows):
        mode_flat = modes[:, k].copy()
        ts = ts_iv[:, k].copy()
        ts_f = ts_fr[:, k].copy()
        time = 1850 + np.arange(ts.shape[0]) / 12.

        check_sum = check_sum + mode_flat * (ts[:, None] + ts_f[:, None])

        # --- spatial mode ---
        mode_flat = undo_lat_scaling(mode_flat[:, None], lat, lon.shape[0], nan_mask=nan_mask)[:, :, 0]
        mode_real = mode_flat.reshape(lat.size, lon.size)

        # normalize time series and patterns
        coeff = (ts_f + ts).std()
        ts_f /= coeff
        ts /= coeff
        mode_real *= coeff

        # flip sign based on trend of forced response
        flip = np.polyfit(time, ts_f, 1)[0] < 0
        if flip:
            ts_f = -ts_f
            ts = -ts
            mode_real = -mode_real

        # --- ACF ---
        acf_length = 40 * 12
        ts_acf = acf(ts, axis=0, length=acf_length)
        acf_time = np.arange(acf_length) / 12.
        acf_lower, acf_upper = acf_bands(ts_acf, len(ts), alpha=0.05)

        # Store data for plotting
        modes_data.append(mode_real)
        ts_data.append((time, ts, ts_f))
        acf_data.append((acf_time, ts_acf, acf_lower, acf_upper))

        label_idx, label_type = labels[k]
        if label_type == 'complex':
            mode_label = f'Mode pair k={label_idx}'
            pair_counts[label_idx] = pair_counts.get(label_idx, 0) + 1
            part_label = f'Part {pair_counts[label_idx]}'
        else:
            mode_label = f'Mode k={label_idx}'
            part_label = ''

        row_labels.append(
            f'{mode_label}\n'
            f'Decay={decay_times[k]:.2f}y\n'
            f'Freq={frequencies[k]:.3f}/y\n'
            f'{part_label}'
        )

    print(np.amax(np.abs(dmd._pcs@ dmd.eofs-check_sum)))

    return {'lat': lat, 'lon': lon, 'nan_mask': nan_mask, 'nrows': nrows,
            'modes_data': modes_data, 'ts_data': ts_data, 'acf_data': acf_data,
            'row_labels': row_labels, 'forced_var_ratio_12': forced_var_ratio_12,
            'forced_fraction': forced_fraction[:nrows]}


def plot_mode_figure(clim_var, chosen_model, d, scale):
    """Plot one dataset: rows = modes, cols = [map, time series, ACF]. scale sets the map levels of every row."""
    lat, lon, nan_mask, nrows = d['lat'], d['lon'], d['nan_mask'], d['nrows']
    modes_data, ts_data, acf_data, row_labels = d['modes_data'], d['ts_data'], d['acf_data'], d['row_labels']

    # --------------------------------------------------
    # Plot: rows = modes, cols = [map, time series, ACF]
    # --------------------------------------------------
    ncols = 3

    # Favor map area aggressively in this layout.
    col1_width = 2.4 if clim_var == 'tas_ocean' else 1.8
    col_widths = [col1_width, 1.1, 1.0]  # relative widths: map, time series, ACF

    # No colorbars between map rows: shorter rows so the maps (width-limited by their
    # aspect ratio) sit close together; the time series and ACF panels get shorter.
    row_height = 1.9 if clim_var == 'tas_ocean' else 2.7
    map_row_hspace = 0.05 if clim_var == 'tas_ocean' else 0.04
    fig = plt.figure(figsize=(sum(col_widths) * 3.4, row_height * nrows))
    gs = gridspec.GridSpec(nrows, ncols, width_ratios=col_widths, figure=fig, hspace=map_row_hspace)

    # Map latitude extent: for OSAT, the mask rows that contain data (40S-65N), rounded outward
    # to 5 degrees, so the whole analysis domain is shown; falls back to the old fixed extent.
    if clim_var == 'tas_ocean':
        data_rows = (~nan_mask).any(axis=1)
        if data_rows.any():
            map_lat_extent = [max(-90., 5 * np.floor(lat[data_rows].min() / 5)),
                              min(90., 5 * np.ceil(lat[data_rows].max() / 5))]
        else:
            map_lat_extent = [-40, 60]
    else:
        map_lat_extent = [-90, 90]

    # Second pass: create plots with shared limits
    Lon, Lat = np.meshgrid(lon, lat)
    map_axes = []
    ts_axes = []
    acf_axes = []
    ts_anchor_ax = None
    acf_anchor_ax = None
    map_mappable = None
    shared_legend_handles = [
        Line2D([0], [0], color='tab:blue', lw=2, alpha=0.9, label='Internal'),
        Line2D([0], [0], color='tab:orange', lw=2, alpha=0.9, label='Forced'),
    ]
    acf_legend_handles = [
        Line2D([0], [0], color='black', lw=2, label='Internal ACF'),
        Patch(facecolor='lightgray', edgecolor='none', alpha=1.0, label='95% CI'),
        Line2D([0], [0], linestyle='None', marker='o', markerfacecolor='none', markeredgecolor='red', markersize=5, label='1st zero crossing'),
    ]

    for k in range(len(modes_data)):#range(top_n):
        mode_real = modes_data[k]
        time, ts, ts_f = ts_data[k]
        acf_time, ts_acf, acf_lower, acf_upper = acf_data[k]

        # --- map ---
        ax_map = fig.add_subplot(gs[k, 0], projection=ccrs.PlateCarree(central_longitude=180))
        m = ax_map.contourf(Lon, Lat, mode_real,
                            transform=ccrs.PlateCarree(),
                            cmap=map_cmap, levels=np.linspace(-scale, scale, 11), extend='both')
        ax_map.set_extent([-180, 180] + map_lat_extent, crs=ccrs.PlateCarree())
        ax_map.coastlines(linewidth=0.7)
        map_axes.append(ax_map)
        if map_mappable is None:
            map_mappable = m
        # Keep map panel size intact by drawing colorbar in an inset axes.
        # One scale per figure, so only the bottom map gets one; it applies to every row.
        if k == len(modes_data) - 1:
            cax_map = ax_map.inset_axes([0.12, -0.12, 0.76, 0.05])
            # Pa values are large, so whole numbers keep the colorbar ticks from overlapping.
            cbar_format = '%.0f' if clim_var == 'psl' else '%.2f'
            cbar_map = fig.colorbar(m, cax=cax_map, orientation='horizontal', format=cbar_format,
                                    ticks=[-scale, 0, scale])  # three ticks fit at print size
            cbar_map.ax.tick_params(labelsize=FS_TICK)
            cbar_map.set_label(cbar_labels[clim_var], fontsize=FS_LABEL)

        # --- time series ---
        if ts_anchor_ax is None:
            ax_ts = fig.add_subplot(gs[k, 1])
            ts_anchor_ax = ax_ts
        else:
            ax_ts = fig.add_subplot(gs[k, 1], sharex=ts_anchor_ax)
        ts_smoothed = gaussian_filter1d(ts, sigma=internal_smooth_sigma)
        ax_ts.plot(time, ts,   color='tab:blue', label='Internal z_i(t)', alpha=0.8, lw=0.1)
        ax_ts.plot(time, ts_smoothed, color='navy', linestyle='--', alpha=0.8, lw=3)
        ax_ts.plot(time, ts_f, color='tab:orange',  label='Forced z_f(t)',   alpha=0.9, lw=2)
        ax_ts.set_xlim(time[0], time[-1])

        row_ts_values = np.concatenate([ts, ts_f])
        ts_min_k = row_ts_values.min()
        ts_max_k = row_ts_values.max()
        ts_span_k = ts_max_k - ts_min_k
        ts_margin_k = 0.02 * ts_span_k if ts_span_k > 0 else 0.05
        ax_ts.set_ylim(ts_min_k - ts_margin_k, ts_max_k + ts_margin_k)

        ax_ts.grid(True)
        inner_yticks(ax_ts)
        ax_ts.tick_params(axis='y', labelsize=FS_TICK)
        ts_axes.append(ax_ts)
        if k < top_n - 1:
            ax_ts.tick_params(labelbottom=False)
        else:
            ax_ts.set_xlabel('Year', fontsize=FS_LABEL)
            ax_ts.tick_params(axis='x', labelsize=FS_TICK)

        # --- ACF ---
        if acf_anchor_ax is None:
            ax_acf = fig.add_subplot(gs[k, 2])
            acf_anchor_ax = ax_acf
        else:
            ax_acf = fig.add_subplot(gs[k, 2], sharex=acf_anchor_ax)

        # Log-scale lag axis cannot include zero, so start from first positive lag.
        acf_time_plot = acf_time[0:]
        ts_acf_plot = ts_acf[0:]
        acf_lower_plot = acf_lower[0:]
        acf_upper_plot = acf_upper[0:]


        for band_name, band_config in TIMESCALE_BANDS.items():
            ax_acf.axvspan(
                band_config['xlim'][0],
                band_config['xlim'][1],
                color=band_config['color'],
                alpha=band_config['alpha'],
                zorder=0,
            )

        ax_acf.plot(acf_time_plot, np.zeros_like(ts_acf_plot), '--', lw=1, color='black')
        ax_acf.fill_between(
            acf_time_plot,
            acf_lower_plot,
            acf_upper_plot,
            color='lightgray',
            alpha=.4,
            zorder=0,
        )
        ax_acf.plot(acf_time_plot, ts_acf_plot, '-', color='black', lw=.1)

        zero_crossing_t = zero_crossing_time(ts_acf_plot, acf_time_plot[1] - acf_time_plot[0])
        if np.isfinite(zero_crossing_t):
            ax_acf.scatter(
                zero_crossing_t,
                0.0,
                facecolors='none',
                edgecolors='red',
                s=16,
                linewidths=1.0,
                zorder=4,
            )

        ax_acf.set_xlim(acf_time_plot[0], acf_time_plot[-1])
        acf_row_values = np.concatenate([ts_acf_plot[4:], acf_lower_plot, acf_upper_plot])
        acf_min_k = acf_row_values.min()
        acf_max_k = acf_row_values.max()
        acf_span_k = acf_max_k - acf_min_k
        acf_margin_k = 0.05 * acf_span_k if acf_span_k > 0 else 0.05
        ax_acf.set_ylim(acf_min_k - acf_margin_k, acf_max_k + acf_margin_k)
        if k == 0:
            ax_acf.text(
                0.5 * (TIMESCALE_BANDS['ENSO']['xlim'][0] + TIMESCALE_BANDS['ENSO']['xlim'][1]),
                0.92,
                'ENSO',
                color=TIMESCALE_BANDS['ENSO']['color'],
                fontsize=FS_TICK,
                ha='center',
                va='top',
                transform=ax_acf.get_xaxis_transform(),
            )
            ax_acf.text(
                0.5 * (TIMESCALE_BANDS['PDO']['xlim'][0] + TIMESCALE_BANDS['PDO']['xlim'][1]),
                0.92,
                'PDO',
                color=TIMESCALE_BANDS['PDO']['color'],
                fontsize=FS_TICK,
                ha='center',
                va='top',
                transform=ax_acf.get_xaxis_transform(),
            )
        ax_acf.grid(True, which='both')
        inner_yticks(ax_acf)
        ax_acf.tick_params(axis='y', labelsize=FS_TICK)
        acf_axes.append(ax_acf)
        if k < top_n - 1:
            ax_acf.tick_params(labelbottom=False)
        else:
            ax_acf.set_xlabel('Lag (years)', fontsize=FS_LABEL)
            ax_acf.tick_params(axis='x', labelsize=FS_TICK)

        row_label_lines = row_labels[k].split('\n')
        mode_text = TextArea(
            row_label_lines[0],
            textprops={'fontsize': FS_LABEL, 'ha': 'right', 'va': 'center'}
        )
        details_text = TextArea(
            '\n'.join(row_label_lines[1:]),
            textprops={'fontsize': FS_TICK, 'ha': 'right', 'va': 'center'}
        )
        row_label_box = VPacker(
            children=[mode_text, details_text],
            align='right',
            pad=0,
            sep=2,
        )
        row_label_artist = AnnotationBbox(
            row_label_box,
            (-0.015, 0.6),
            xycoords=ax_map.transAxes,
            frameon=False,
            box_alignment=(1.0, 0.5),
            annotation_clip=False,
            pad=0,
        )
        ax_map.add_artist(row_label_artist)

    plt.tight_layout(rect=[0.03, 0.10, 1.0, 0.95])

    # Make rows contiguous in columns 2 and 3 (no vertical white gaps).
    for column_axes in (ts_axes, acf_axes):
        col_top = max(ax.get_position().y1 for ax in column_axes)
        col_bottom = min(ax.get_position().y0 for ax in column_axes)
        col_height = (col_top - col_bottom) / nrows
        for row_idx, ax in enumerate(column_axes):
            pos = ax.get_position()
            new_y0 = col_top - (row_idx + 1) * col_height
            ax.set_position([pos.x0, new_y0, pos.width, col_height])

    # Place titles and shared elements after layout so they align with column bounds.
    col1_x0 = min(ax.get_position().x0 for ax in map_axes)
    col1_x1 = max(ax.get_position().x1 for ax in map_axes)
    col2_x0 = min(ax.get_position().x0 for ax in ts_axes)
    col2_x1 = max(ax.get_position().x1 for ax in ts_axes)
    col3_x0 = min(ax.get_position().x0 for ax in acf_axes)
    col3_x1 = max(ax.get_position().x1 for ax in acf_axes)
    top_y = max(
        max(ax.get_position().y1 for ax in map_axes),
        max(ax.get_position().y1 for ax in ts_axes),
        max(ax.get_position().y1 for ax in acf_axes),
    )

    pt = 1. / 72. / fig.get_figheight()  # one point as a figure fraction
    title_y = top_y + 6 * pt
    fig.text(0.5 * (col1_x0 + col1_x1), title_y, 'Spatial pattern', ha='center', va='bottom', fontsize=FS_TITLE, transform=fig.transFigure)
    fig.text(0.5 * (col2_x0 + col2_x1), title_y, 'Time series', ha='center', va='bottom', fontsize=FS_TITLE, transform=fig.transFigure)
    fig.text(0.5 * (col3_x0 + col3_x1), title_y, r'Internal ACF', ha='center', va='bottom', fontsize=FS_TITLE, transform=fig.transFigure)

    maps_bottom = min(ax.get_position().y0 for ax in map_axes)
    ts_bottom = min(ax.get_position().y0 for ax in ts_axes)
    acf_bottom = min(ax.get_position().y0 for ax in acf_axes)

    # One legend for the time series and ACF columns, below their x tick labels and axis labels.
    below_xlabel = 1.6 * (FS_TICK + FS_LABEL) * pt
    legend = fig.legend(
        handles=shared_legend_handles + acf_legend_handles,
        loc='upper center',
        ncol=3,
        fontsize=FS_TICK,
        frameon=True,
        fancybox=False,
        bbox_to_anchor=(0.5 * (col2_x0 + col3_x1), min(ts_bottom, acf_bottom) - below_xlabel),
        bbox_transform=fig.transFigure,
    )
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)

    plt.savefig(f'{results_dir}/dmd_modes_{chosen_model}_{clim_var}.pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def main():
    os.makedirs(results_dir, exist_ok=True)
    for clim_var in clim_vars:

        if clim_var == 'psl':
            chosen_method = 'PullbackDMDc-2d'
        else:
            chosen_method = 'PullbackDMDc-3d'

        datasets = {}
        for (chosen_model, chosen_member) in esm_list:
            d = load_mode_rows(clim_var, chosen_model, chosen_member, chosen_method)
            if d is not None:
                datasets[chosen_model] = d
        if not datasets:
            continue

        # R2-L432: forced-variance ratio of modes 1-2 for the member shown in each figure.
        # Plus the forced fraction of each plotted mode's time series (variance of the orange curves).
        ratio_rows = [{'dataset': m, 'member': dict(esm_list)[m], 'forced_var_ratio_12': d['forced_var_ratio_12'],
                       **{f'forced_fraction_mode{k + 1}': f for k, f in enumerate(d['forced_fraction'])}}
                      for m, d in datasets.items()]
        for row, d in zip(ratio_rows, datasets.values()):
            print(f'{clim_var} {row["dataset"]}: var(modes 1-2 forced) / var(forced) = {row["forced_var_ratio_12"]:.3f}; '
                  f'forced fraction per mode = {np.round(d["forced_fraction"], 3)}', flush=True)
        out_csv = os.path.join(results_dir, f'forced_variance_ratio_{clim_var}.csv')
        pd.DataFrame(ratio_rows).to_csv(out_csv, index=False)
        print(f'Saved: {out_csv}', flush=True)

        for chosen_model, d in datasets.items():
            plot_mode_figure(clim_var, chosen_model, d, p95_abs(d['modes_data']))


if __name__ == '__main__':
    main()

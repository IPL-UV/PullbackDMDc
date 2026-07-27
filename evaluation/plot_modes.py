import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from utils.data_utils import undo_lat_scaling
from utils.params import artifact_root, pdf_root, predictions_dir
from scipy.ndimage import gaussian_filter1d
import matplotlib.gridspec as gridspec

# Parameters for selection
lag=3
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

clim_vars = ['tas', 'tas_ocean', 'psl']

chosen_model = '20CRv3'
chosen_member = ''
esm_list = [
    ('20CRv3', ''),
    ('MPI-ESM1-2-LR', 'r47i1p1f1'),
    ('CESM2', '1281.005i1p1f1'),
    ('CanESM5','r16i1p1f1'),
    ('MIROC6', 'r35i1p1f1')
]

clim_vars_on_fig = {'tas_ocean': 'OSAT', 'psl': 'SLP', 'tas': 'SAT'}


results_dir = os.path.join(pdf_root, f'results{year_suffix}{lag_suffix}')


eofs_base_path = os.path.join(artifact_root, f'eofs{year_suffix}')
base_pred_dir = predictions_dir(year_suffix, lag_suffix)

top_n = 4
map_cmap = 'RdBu_r'
shared_fontsize = 12
internal_smooth_sigma = 12 * 8 # eight years
timescale_bands = {
    'ENSO': {'xlim': (2.0, 7.0),   'color': 'tab:cyan',   'alpha': 0.12},
    'PDO':  {'xlim': (10.0, 30.0), 'color': 'tab:purple', 'alpha': 0.10},
}
#acf_length = [100 * 12] * 2 + [20*12] * (top_n-2)

def compute_yearly_decay_time(lambdas, tau):
    return -tau / (12 * np.log(np.abs(lambdas)))

def compute_yearly_frequency(lambdas, tau):
    omegas = np.angle(lambdas) / tau
    return (6 * np.abs(omegas)) / np.pi

def parse_ensemble_member_from_filename(f_name, esm_name):
    if esm_name == '20CRv3': return ''
    return f_name.split('_')[-4]

def acf(x, axis=0, length=60):
    x = x - x.mean(axis=axis, keepdims=True)
    N = x.shape[axis]
    def acf_1d(z):
        gamma0 = np.sum(z * z) / N
        gamma0 = gamma0 if gamma0 > 1e-12 else 1e-12
        return np.array([(np.sum(z[:N-k] * z[k:]) / (N - k)) / gamma0 for k in range(length)])
    return np.apply_along_axis(acf_1d, axis, x)

def acf_bands(acf_values, N, alpha=0.05):
    # White-noise null: CI at lag k is +/- z_score / sqrt(N-k).
    z_score = 1.96 # 95% rejection of white noise hypothesis
    # z_score = 2.576 # 99% rejection of white noise hypothesis
    acf_values = np.asarray(acf_values, dtype=float).reshape(-1)
    se = np.zeros_like(acf_values, dtype=float)
    max_lag = min(acf_values.size, N)
    for k in range(max_lag):
        se[k] = 1.0 / np.sqrt(N - k)
    band = z_score * se
    return -band, band


def first_zero_crossing_time(acf_time, acf_values):
    for idx in range(1, len(acf_values)):
        if acf_values[idx] <= 0.0:
            prev_idx = idx - 1
            prev_value = acf_values[prev_idx]
            next_value = acf_values[idx]
            if np.isclose(prev_value, next_value):
                return float(acf_time[idx])
            weight = (0.0 - prev_value) / (next_value - prev_value)
            weight = np.clip(weight, 0.0, 1.0)
            return float(acf_time[prev_idx] + weight * (acf_time[idx] - acf_time[prev_idx]))
    return np.nan


def display_model_name(model_name):
    return 'MPI-ESM' if model_name == 'MPI-ESM1-2-LR' else model_name

for clim_var in clim_vars:

    if clim_var == 'psl':
        chosen_method = 'PullbackDMDc-2d'
    else:
        chosen_method = 'PullbackDMDc-3d'
    for (chosen_model, chosen_member) in esm_list:
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
            if parse_ensemble_member_from_filename(f_name, chosen_model) == chosen_member:
                matched_dmd_file = os.path.join(model_dir, f_name)
                eof_file = os.path.join(eof_dir, f_name.replace('_' + chosen_method, ''))
                break

        if matched_dmd_file is None or eof_file is None:
            print(
                f'Skipping {clim_var} / {chosen_model}: '
                f'no file for method={chosen_method} member={chosen_member!r}'
            )
            continue

        with open(matched_dmd_file, 'rb') as f:
            dmd = pickle.load(f)

        with open(eof_file, 'rb') as f:
            data_parameters = pickle.load(f)['data_pars']
            nan_mask = data_parameters['nan_mask']
            lat = data_parameters['lat'].values
            lon = data_parameters['lon'].values

        dmd.compute_modes()
        rotated = dmd.compute_rotated_modes(top_n=top_n)

        # --------------------------------------------------
        # Plot: rows = modes, cols = [map, time series, ACF]
        # --------------------------------------------------
        ncols = 3
        eigvals = rotated['eigvals']
        modes = rotated['spatial_patterns']
        ts_iv = rotated['internal_time_series']
        ts_fr = rotated['forced_time_series']
        labels = rotated['labels']
        nrows = min(top_n, eigvals.size)

        decay_times = compute_yearly_decay_time(eigvals, dmd.lag)
        frequencies = compute_yearly_frequency(eigvals, dmd.lag)


        # Favor map area aggressively in this layout.
        col1_width = 2.4 if clim_var == 'tas_ocean' else 1.8
        col_widths = [col1_width, 1.1, 1.0]  # relative widths: map, time series, ACF

        row_height = 2.5 if clim_var == 'tas_ocean' else 3.5
        fig = plt.figure(figsize=(sum(col_widths) * 3.4, row_height * nrows))
        map_row_hspace = 0.18 if clim_var == 'tas_ocean' else 0.12
        gs = gridspec.GridSpec(nrows, ncols, width_ratios=col_widths, figure=fig, hspace=map_row_hspace)

        # First pass: compute all data and determine global limits
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
        # Compute global limits for non-map columns
        # Column 1: spatial modes
        all_mode_values = np.concatenate([m.flatten() for m in modes_data])
        v_global = max(np.nanpercentile(np.abs(all_mode_values), 90), 1e-12)

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
            mode_scale_k = max(np.nanpercentile(np.abs(mode_real), 95), 1e-12)

            # --- map ---
            ax_map = fig.add_subplot(gs[k, 0], projection=ccrs.PlateCarree(central_longitude=180))
            m = ax_map.contourf(Lon, Lat, mode_real,
                                transform=ccrs.PlateCarree(),
                                cmap=map_cmap, levels=np.linspace(-mode_scale_k, mode_scale_k, 11), extend='both')
            if clim_var == 'tas_ocean':
                ax_map.set_extent([-180, 180, -40, 60], crs=ccrs.PlateCarree())
            else:
                ax_map.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
            ax_map.coastlines(linewidth=0.7)
            map_axes.append(ax_map)
            if map_mappable is None:
                map_mappable = m
            # Keep map panel size intact by drawing colorbar in an inset axes.
            cax_map = ax_map.inset_axes([0.12, -0.12, 0.76, 0.05])
            cbar_map = fig.colorbar(m, cax=cax_map, orientation='horizontal', format='%.2f')
            cbar_map.ax.tick_params(labelsize=shared_fontsize)

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
            ax_ts.tick_params(axis='y', labelsize=shared_fontsize)
            ts_axes.append(ax_ts)
            if k < top_n - 1:
                ax_ts.tick_params(labelbottom=False)
            else:
                ax_ts.set_xlabel('Year', fontsize=shared_fontsize)
                ax_ts.tick_params(axis='x', labelsize=shared_fontsize)

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


            for band_name, band_config in timescale_bands.items():
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

            zero_crossing_t = first_zero_crossing_time(acf_time_plot, ts_acf_plot)
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
                    0.5 * (timescale_bands['ENSO']['xlim'][0] + timescale_bands['ENSO']['xlim'][1]),
                    0.92,
                    'ENSO',
                    color=timescale_bands['ENSO']['color'],
                    fontsize=shared_fontsize - 1,
                    ha='center',
                    va='top',
                    transform=ax_acf.get_xaxis_transform(),
                )
                ax_acf.text(
                    0.5 * (timescale_bands['PDO']['xlim'][0] + timescale_bands['PDO']['xlim'][1]),
                    0.92,
                    'PDO',
                    color=timescale_bands['PDO']['color'],
                    fontsize=shared_fontsize - 1,
                    ha='center',
                    va='top',
                    transform=ax_acf.get_xaxis_transform(),
                )
            ax_acf.grid(True, which='both')
            ax_acf.tick_params(axis='y', labelsize=shared_fontsize)
            acf_axes.append(ax_acf)
            if k < top_n - 1:
                ax_acf.tick_params(labelbottom=False)
            else:
                ax_acf.set_xlabel('Lag (years)', fontsize=shared_fontsize)
                ax_acf.tick_params(axis='x', labelsize=shared_fontsize)

            row_label_lines = row_labels[k].split('\n')
            mode_text = TextArea(
                row_label_lines[0],
                textprops={'fontsize': 14, 'ha': 'right', 'va': 'center'}
            )
            details_text = TextArea(
                '\n'.join(row_label_lines[1:]),
                textprops={'fontsize': 12, 'ha': 'right', 'va': 'center'}
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

        title_y = min(0.985, top_y + 0.015)
        suptitle_y = min(0.995, title_y + 0.06)
        fig.suptitle(
            f'{display_model_name(chosen_model)}, {clim_vars_on_fig[clim_var]}: Top {top_n} dynamic modes',
            fontsize=17,
            y=suptitle_y,
        )
        fig.text(0.5 * (col1_x0 + col1_x1), title_y, 'Spatial pattern', ha='center', va='bottom', fontsize=16, transform=fig.transFigure)
        fig.text(0.5 * (col2_x0 + col2_x1), title_y, 'Time series', ha='center', va='bottom', fontsize=16, transform=fig.transFigure)
        fig.text(0.5 * (col3_x0 + col3_x1), title_y, r'Internal ACF', ha='center', va='bottom', fontsize=16, transform=fig.transFigure)

        maps_bottom = min(ax.get_position().y0 for ax in map_axes)
        ts_bottom = min(ax.get_position().y0 for ax in ts_axes)
        acf_bottom = min(ax.get_position().y0 for ax in acf_axes)

        ts_legend = fig.legend(
            handles=shared_legend_handles,
            loc='upper center',
            ncol=1,
            fontsize=shared_fontsize,
            frameon=True,
            fancybox=False,
            bbox_to_anchor=(0.5 * (col2_x0 + col2_x1), max(0.02, ts_bottom - 0.065)),
            bbox_transform=fig.transFigure,
        )
        ts_legend.get_frame().set_edgecolor('gray')
        ts_legend.get_frame().set_linewidth(0.8)
        ts_legend.get_frame().set_alpha(1.0)

        acf_legend = fig.legend(
            handles=acf_legend_handles,
            loc='upper center',
            ncol=2,
            fontsize=shared_fontsize,
            frameon=True,
            fancybox=False,
            bbox_to_anchor=(0.5 * (col3_x0 + col3_x1), max(0.01, acf_bottom - 0.06)),
            bbox_transform=fig.transFigure,
        )
        acf_legend.get_frame().set_edgecolor('gray')
        acf_legend.get_frame().set_linewidth(0.8)
        acf_legend.get_frame().set_alpha(1.0)

        plt.savefig(f'{results_dir}/dmd_modes_{chosen_model}_{clim_var}.pdf', dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
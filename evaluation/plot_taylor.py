import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import json
from evaluation.utils.metrics import add_one_custom_plot, compute_taylor_std_range
from evaluation.utils.style import METHOD_NAMES_ON_FIG, display_model_name, set_font_sizes
from evaluation.utils.trends import parse_window, window_label
from utils.params import artifact_root, colors, method_names, esms, year_ranges, lags, method_markers, pdf_root, requested_lags, requested_year_ranges
from utils.params import requested_clim_vars

# Prints at ~7/8/9/10 pt when included at 0.49 \textwidth (figure saves ~6.8 in wide).
FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE = 16, 18, 19, 21
set_font_sizes(FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE)
esms=esms[1:] # Exclude obs for this figure
esms = ['CESM2', 'MPI-ESM1-2-LR', 'CanESM5', 'MIROC6']

###########################################
# Plot only necessary to save some space
clim_vars = requested_clim_vars(['tas_ocean', 'psl', 'tas'])
lags=[3]#, 3]
year_ranges=[(1850, 2014)]#, (1950, 2014)]
# Choose which Taylor metrics to plot: 'raw' or 'trend'.
taylor_plot_mode = os.getenv('PBDMDC_TAYLOR_MODE', 'trend')
###########################################


def get_mode_config(mode):
    if mode == 'raw':
        return {
            'metrics_prefix': 'td_metrics',
            'title_mode': 'raw',
            'output_prefix': 'classic_taylor_diagram',
        }
    if mode == 'trend':
        # Optional trend window (R2-M2), as written by compute_trend_taylor with PBDMDC_TREND_WINDOW.
        raw_window = os.getenv('PBDMDC_TREND_WINDOW', '').strip()
        if raw_window:
            window = parse_window(raw_window)
            return {
                'metrics_prefix': f'td_metrics_trend_{window_label(window)}',
                'title_mode': f'trend {window[0]}\u2013{window[1]}',
                'output_prefix': f'classic_taylor_diagram_trend_{window_label(window)}',
            }
        return {
            'metrics_prefix': 'td_metrics_trend',
            'title_mode': 'trend',
            'output_prefix': 'classic_taylor_diagram_trend',
        }
    raise ValueError("taylor_plot_mode must be either 'raw' or 'trend'")


def _sanitize_nonnegative_extent(extent, min_span=1e-3):
    """Clamp extent to nonnegative x/y limits and ensure finite span."""
    (xlim, ylim) = extent
    x_min = max(0.0, float(xlim[0]))
    x_max = max(x_min + min_span, float(xlim[1]))
    y_min = max(0.0, float(ylim[0]))
    y_max = max(y_min + min_span, float(ylim[1]))
    return (x_min, x_max), (y_min, y_max)


def _tight_std_limit(td_list, padding_frac=0.015):
    """Return a tight std upper bound that still contains all plotted points."""
    max_std = max(
        max(float(sample['std_Y']) for sample in td_list),
        float(td_list[0]['std_X']),
    )
    return max_std * (1.0 + padding_frac)


def _robust_std_limit(td_list, keep_percentile=90.0, padding_frac=0.015):
    """Return a cropped std upper bound that removes high-end outliers."""
    std_values = [float(sample['std_Y']) for sample in td_list]
    std_values.append(float(td_list[0]['std_X']))
    capped_std = float(np.percentile(std_values, keep_percentile))
    capped_std = max(capped_std, float(td_list[0]['std_X']))
    return capped_std * (1.0 + padding_frac)


def _td_to_xy(std_value, corr_value):
    corr = float(max(-1.0, min(1.0, corr_value)))
    theta = np.arccos(corr)
    return float(std_value * np.cos(theta)), float(std_value * np.sin(theta))


def _large_point_coords(td_list):
    """Coordinates for larger plotted markers (used to dodge contour labels)."""
    coords = []
    # Reference point added by add_one_custom_plot.
    coords.append((float(td_list[0]['std_X']), 0.0))
    for sample in td_list:
        if float(sample.get('size', 0)) >= 8:
            coords.append(_td_to_xy(float(sample['std_Y']), float(sample['corr'])))
    return np.asarray(coords, dtype=float)


def _pick_contour_label_positions(contour_set, levels, avoid_points, xlim, ylim):
    """Pick one manual label position per level, far from large markers."""
    has_avoid_points = avoid_points.size > 0

    level_to_index = {
        float(level): idx
        for idx, level in enumerate(contour_set.levels)
    }
    x_span = max(1e-9, xlim[1] - xlim[0])
    y_span = max(1e-9, ylim[1] - ylim[0])
    outer_radius = max(xlim[1], ylim[1])  # radius of the outer arc that carries the correlation labels

    positions = []
    for level in levels:
        idx = level_to_index.get(float(level))
        if idx is None:
            continue

        best_pos = None
        best_score = -np.inf
        fallback_pos = None
        fallback_len = -1
        for seg in contour_set.allsegs[idx]:
            if seg is None or len(seg) == 0:
                continue

            stride = max(1, len(seg) // 80)
            for x, y in seg[::stride]:
                # Keep labels inside safe interior and below title region.
                if not (xlim[0] + 0.12 * x_span <= x <= xlim[1] - 0.22 * x_span):
                    continue
                if not (ylim[0] + 0.10 * y_span <= y <= ylim[1] - 0.30 * y_span):
                    continue
                # Avoid the upper-right arc area where correlation tick labels are drawn.
                if x >= xlim[0] + 0.70 * x_span and y >= ylim[0] + 0.56 * y_span:
                    continue
                if np.hypot(x, y) > 0.85 * outer_radius:  # near the outer arc and its labels
                    continue

                if has_avoid_points:
                    dx = (avoid_points[:, 0] - x) / x_span
                    dy = (avoid_points[:, 1] - y) / y_span
                    min_dist = float(np.min(np.sqrt(dx * dx + dy * dy)))
                else:
                    # No avoidance targets: rank all safe positions equally.
                    min_dist = 1.0
                if min_dist > best_score:
                    best_score = min_dist
                    best_pos = (float(x), float(y))

        if best_pos is not None:
            positions.append(best_pos)

    return positions  # empty: no safe spot, leave the contours unlabelled rather than auto-placing


def _std_tick_label(v, _):
    """y tick label of the top diagram: blank at 0, where the x axis already shows it; else 0.2, 1.0, 1.25."""
    if abs(v) < 1e-9:
        return ''
    label = f'{v:.2f}'
    return label[:-1] if label.endswith('0') else label


def _tightened_contour_levels(contour_set):
    """Drop edge-most contour levels so labels do not get clipped; label every other level so they fit."""
    levels = list(contour_set.levels)
    if len(levels) > 4:
        levels = levels[1:-1]
    return levels[::2]


if __name__ == '__main__':

    default_std_model = 'MPI-ESM1-2-LR'
    mode_config = get_mode_config(taylor_plot_mode)
    top_corr_label_fontsize = FS_TICK
    crop_corr_label_fontsize = FS_TICK
    crop_corr_labels = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9])  # sparser arc labels for the zoomed panels
    top_corr_label_pad = 0.03
    crop_corr_label_pad = 0.005
    top_title_y = 1.06
    crop_title_y = 1.12
    top_title_fontsize = FS_TITLE
    top_correlation_text_fontsize = FS_LABEL
    top_axis_label_fontsize = top_correlation_text_fontsize
    top_panel_keep_percentile = 95.0
    model_palette = ['#7f7f7f', '#8c564b', '#2ca02c', '#0b1f5b']
    model_colors = {model: model_palette[i] for i, model in enumerate(esms)}

    def build_classic_td(model_metrics):
        classic_td = []

        for method in method_names:
            color = colors[method]
            marker = method_markers[method]

            for i, td_values in enumerate(model_metrics[method]['ensemble']):
                classic_td.append({
                    **td_values,
                    'name': None,
                    'color': color,
                    'alpha': 0.2,
                    'size': 6,
                    'marker': '.',
                })

            classic_td.append({
                **model_metrics[method]['total'],
                'name': METHOD_NAMES_ON_FIG[method],
                'color': color,
                'alpha': 1,
                'size': 8,
                'marker': marker,
            })

        return classic_td

    def build_model_colored_td(model_metrics, model_color):
        td_points = []

        for method in method_names:
            marker = method_markers[method]

            for td_values in model_metrics[method]['ensemble']:
                td_points.append({
                    **td_values,
                    'name': None,
                    'color': model_color,
                    'alpha': 0.18,
                    'size': 5,
                    'marker': '.',
                })

            td_points.append({
                **model_metrics[method]['total'],
                'name': None,
                'color': model_color,
                'alpha': 0.95,
                'size': 8,
                'marker': marker,
            })

        return td_points
 
    for lag in requested_lags(lags):
        for start_year, end_year in requested_year_ranges(year_ranges):
 
            if lag == 3:
                lag_suffix = ''
            else:
                lag_suffix = f'_lag{lag}'
 
            if (start_year, end_year) == (1950, 2014):
                year_suffix = '_tier1'
            elif (start_year, end_year) == (1850, 2014):
                year_suffix = ''
            else:
                print('year range not recognized')
 
            artifact_results_dir = os.path.join(artifact_root, 'evaluation_results', f'results{year_suffix}{lag_suffix}')
            pdf_results_dir = os.path.join(pdf_root, f'results{year_suffix}{lag_suffix}')
 
            os.makedirs(artifact_results_dir, exist_ok=True)
            os.makedirs(pdf_results_dir, exist_ok=True)

            for var in clim_vars:
                print(
                    f"\n=== Plotting {taylor_plot_mode} Taylor: {var}, lag={lag}, years={start_year}-{end_year} ===",
                    flush=True,
                )
 
                # Load saved metrics
                in_path = os.path.join(artifact_results_dir, f"{mode_config['metrics_prefix']}_{var}.json")
                with open(in_path, 'r') as f:
                    td_metrics = json.load(f)

                classic_td_by_model = {
                    model: build_classic_td(td_metrics[model])
                    for model in esms
                }
                top_td = []
                for model in esms:
                    top_td.extend(build_model_colored_td(td_metrics[model], model_colors[model]))
                std_range_model = default_std_model if default_std_model in classic_td_by_model else esms[-1]
                shared_std_range = compute_taylor_std_range(
                    classic_td_by_model[std_range_model],
                    percentage=95,
                )
                top_std_range = compute_taylor_std_range(top_td, percentage=95)
                full_axis_extent = (
                    (top_std_range[0], top_std_range[1]),
                    (top_std_range[0], top_std_range[1]),
                )

                legend_kwargs = {
                    'loc': 'center',
                    'ncol': 1,
                    'frameon': False,
                }
                zoom_highlight_color = '#8c6d46'
 
                fig = plt.figure(figsize=(7, 15))
                grid_spec = fig.add_gridspec(2, 1, height_ratios=[4.1, 4.8], hspace=0.5)
                top_ax = fig.add_subplot(grid_spec[0, 0])
                bottom_grid = grid_spec[1, 0].subgridspec(2, 2, hspace=0.55, wspace=0.28)
                bottom_layout = {
                    'CESM2': fig.add_subplot(bottom_grid[0, 0]),
                    'MPI-ESM1-2-LR': fig.add_subplot(bottom_grid[0, 1]),
                    'CanESM5': fig.add_subplot(bottom_grid[1, 0]),
                    'MIROC6': fig.add_subplot(bottom_grid[1, 1]),
                }

                top_ax = add_one_custom_plot(
                    top_td,
                    fig,
                    legend=False,
                    percentage=95,
                    std_range=top_std_range,
                    legend_kwargs=legend_kwargs,
                    axis_extent=full_axis_extent,
                    ax=top_ax,
                    corr_label_pad=top_corr_label_pad,
                    corr_label_fontsize=top_corr_label_fontsize,
                )
                # top_ax.set_title('All ESMs', y=top_title_y, fontsize=top_title_fontsize)
                top_ax.set_xlabel('Standard Deviation', fontsize=top_axis_label_fontsize)
                top_ax.set_ylabel('Standard Deviation', fontsize=top_axis_label_fontsize)
                top_ax.yaxis.set_major_formatter(FuncFormatter(_std_tick_label))
                top_ax.text(
                    0.87,
                    0.87,
                    'Correlation',
                    transform=top_ax.transAxes,
                    rotation=-45,
                    ha='center',
                    va='center',
                    fontsize=top_correlation_text_fontsize,
                )

                top_contour_set = getattr(top_ax, '_taylor_contours', None)
                if top_contour_set is not None:
                    top_levels = _tightened_contour_levels(top_contour_set)
                    top_manual_positions = _pick_contour_label_positions(
                        top_contour_set,
                        top_levels,
                        _large_point_coords(top_td),
                        top_ax.get_xlim(),
                        top_ax.get_ylim(),
                    )
                    if top_manual_positions:
                        top_ax.clabel(
                            top_contour_set,
                            levels=top_levels,
                            manual=top_manual_positions,
                            inline=True,
                            inline_spacing=8,
                            fontsize=top_corr_label_fontsize,
                            fmt='%.2f',
                            use_clabeltext=True,
                            rightside_up=True,
                        )



                panel_extents = {}
                bottom_contours = {}
                bottom_large_points = {}
 
                for index, model in enumerate(esms):
                    ax = bottom_layout[model]
                    print(f'Plotting model: {model}', flush=True)
                    classic_td = classic_td_by_model[model]

                    ax = add_one_custom_plot(
                        classic_td,
                        fig,
                        legend=False,
                        percentage=95,
                        std_range=shared_std_range,
                        legend_kwargs=legend_kwargs,
                        ax=ax,
                        corr_label_pad=crop_corr_label_pad,
                        corr_label_fontsize=crop_corr_label_fontsize,
                        corr_labels=crop_corr_labels,
                    )
                    contour_set = getattr(ax, '_taylor_contours', None)
                    if contour_set is not None:
                        bottom_contours[model] = contour_set
                    bottom_large_points[model] = _large_point_coords(classic_td)
                    ax.set_title(display_model_name(model), y=crop_title_y)
                    panel_extents[model] = (ax.get_xlim(), ax.get_ylim())
                    ax.set_xlabel("")
                    ax.set_ylabel("")

                # Enforce common square crop while excluding invalid regions:
                # x < 0 corresponds to negative correlation and y < 0 to negative std geometry.
                panel_extents = {
                    model: _sanitize_nonnegative_extent(extent)
                    for model, extent in panel_extents.items()
                }

                shared_side = max(
                    max(xlim[1] - xlim[0], ylim[1] - ylim[0])
                    for xlim, ylim in panel_extents.values()
                )

                for model, ax in bottom_layout.items():
                    xlim, ylim = panel_extents[model]
                    x_center = 0.5 * (xlim[0] + xlim[1])
                    y_center = 0.5 * (ylim[0] + ylim[1])

                    x_min = max(0.0, x_center - 0.5 * shared_side)
                    y_min = max(0.0, y_center - 0.5 * shared_side)
                    x_max = x_min + shared_side
                    y_max = y_min + shared_side

                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    # Keep tick labels away from the top and right edges, where the arc labels are drawn.
                    ax.set_xticks([t for t in ax.get_xticks() if x_min <= t <= x_max - 0.12 * (x_max - x_min)])
                    ax.set_yticks([t for t in ax.get_yticks() if y_min <= t <= y_max - 0.15 * (y_max - y_min)])
                    # Ensure correlation tick labels are recomputed for cropped views.
                    if hasattr(ax, '_taylor_plot'):
                        ax._taylor_plot._draw_corr_ticklabels()

                    contour_set = bottom_contours.get(model)
                    if contour_set is not None:
                        levels = list(contour_set.levels)[::2]  # every other level, so labels fit
                        manual_positions = _pick_contour_label_positions(
                            contour_set,
                            levels,
                            bottom_large_points.get(model, np.empty((0, 2))),
                            ax.get_xlim(),
                            ax.get_ylim(),
                        )
                        if manual_positions:
                            ax.clabel(
                                contour_set,
                                levels=levels,
                                manual=manual_positions,
                                inline=True,
                                inline_spacing=8,
                                fontsize=crop_corr_label_fontsize,
                                fmt='%.2f',
                                use_clabeltext=True,
                                rightside_up=True,
                            )

                fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.98))

                method_handles = [
                    plt.Line2D(
                        [], [],
                        color=colors[method],
                        marker=method_markers[method],
                        linestyle='',
                        markersize=6,
                        label=METHOD_NAMES_ON_FIG[method],
                    )
                    for method in method_names
                ]
                model_handles = [
                    plt.Line2D(
                        [], [],
                        color=model_colors[model],
                        marker='o',
                        linestyle='',
                        markersize=6,
                        label=display_model_name(model),
                    )
                    for model in esms
                ]

                pt = 1. / 72. / fig.get_figheight()  # one point as a figure fraction
                frame = dict(frameon=True, fancybox=False, edgecolor='#888888', framealpha=1.0, fontsize=FS_TICK,
                             title_fontsize=FS_TICK, handletextpad=0.35, columnspacing=0.8)

                # Climate-model legend below the top all-ESMs panel's tick labels and axis label.
                model_legend = fig.legend(
                    model_handles,
                    [h.get_label() for h in model_handles],
                    loc='upper center',
                    bbox_to_anchor=(0.5, top_ax.get_position().y0 - 1.8 * (FS_TICK + FS_LABEL) * pt),
                    ncol=4,
                    title='Climate Model',
                    **frame,
                )
                model_legend.get_frame().set_linewidth(0.8)

                # Method legend below the 2x2 plots; rows: baselines, then PullbackDMDc variants.
                order = ['RegGMST', 'LIM-opt', 'PullbackDMDc', 'PullbackDMDc-3d', 'LIM', 'LR', 'PullbackDMDc-2d']
                handle_of = {h.get_label(): h for h in method_handles}
                method_legend_handles = [handle_of[METHOD_NAMES_ON_FIG[m]] for m in order]
                bottom_y0 = min(ax.get_position().y0 for ax in bottom_layout.values())
                method_legend = fig.legend(
                    method_legend_handles,
                    [h.get_label() for h in method_legend_handles],
                    loc='upper center',
                    bbox_to_anchor=(0.5, bottom_y0 - 1.8 * FS_TICK * pt),
                    ncol=2,
                    title='Method (Pooled Comparison)',
                    **frame,
                )
                method_legend.get_frame().set_linewidth(0.8)

                out_fig = os.path.join(pdf_results_dir, f"{mode_config['output_prefix']}_{var}.pdf")
                plt.savefig(out_fig, bbox_inches='tight')
                plt.close()
                print(f'Saved: {out_fig}', flush=True)
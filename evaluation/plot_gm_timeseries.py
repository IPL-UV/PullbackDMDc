import numpy as np
import os
import time
import pickle as pkl
from utils.data_utils import load_truncated_prediction, load_truth
from evaluation.utils.metrics import global_mean, global_mean_flat, coord_to_dtindex, smooth_truth
from evaluation.utils.bootstrap import member_bootstrap_rmsd, paired_difference
from utils.params import esms, method_names, colors, year_ranges, lags
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
import pandas as pd
from utils.params import method_markers
from utils.params import artifact_root, pdf_root, requested_lags, requested_year_ranges
esms = esms[1:] # Exclude obs for this figure

###########################################
# Plot only necessary to save some space
clim_vars = ['tas_ocean', 'psl', 'tas']
lags=[3]
year_ranges=[(1850, 2014), (1950, 2014)]
###########################################

clim_vars_on_fig = {'tas_ocean': 'OSAT', 'psl': 'SLP', 'tas': 'SAT'}
method_names_on_fig = {'RegGMST': 'RegGMST', 'LR': 'Linear reg.','LIM': 'LIM','LIM-opt': 'LIM-opt',
                       'PullbackDMDc': 'PullbackDMDc', 'PullbackDMDc-2d': 'PullbackDMDc(2d)', 'PullbackDMDc-3d': 'PullbackDMDc(3d)'}
method_order_on_fig = ['RegGMST', 'LIM-opt', 'LIM',  'PullbackDMDc', 'LR', 'PullbackDMDc-2d', 'PullbackDMDc-3d']


def display_model_name(model_name):
    return 'MPI-ESM' if model_name == 'MPI-ESM1-2-LR' else model_name


volcanic_events = [
    (1861, 1866, 'Makian'),
    (1883, 1888, 'Krakatoa'),
    (1902, 1907, 'Santa María'),
    (1912, 1917, 'Novarupta'),
    (1963, 1968, 'Agung'),
    (1982, 1987, 'El Chichón'),
    (1991, 1996, 'Pinatubo'),
]

# Pre-build volcanic mask intervals as Timestamps
volcanic_intervals = [
    (pd.Timestamp(f'{v_start}-01-01'), pd.Timestamp(f'{v_end}-01-01'))
    for v_start, v_end, _ in volcanic_events
]


def is_in_volcanic(t):
    return any(t_s <= t < t_e for t_s, t_e in volcanic_intervals)


ci_columns = ['var', 'model', 'period', 'method', 'n_members', 'rmsd_pct', 'ci_lo', 'ci_hi',
              'ref_method', 'diff_vs_ref', 'diff_ci_lo', 'diff_ci_hi', 'ref_significantly_better',
              'member_p2_5', 'member_p97_5']


def draw_rmsd_bars(ax_bar, timeseries_model, methods, truth, time_index, var, seed):
    """
    Grouped bars of pooled RMSD by period (% of truth std) with member-bootstrap 95% CIs.
    The same member resampling is used for every method and period (paired bootstrap), so a
    '*' marks methods whose RMSD is significantly higher than the reference PullbackDMDc.
    Returns one CSV row (dict) per (period, method); the caller adds 'model'.
    """
    tick_fontsize = 10
    ref_method = 'PullbackDMDc-2d' if var == 'psl' else 'PullbackDMDc-3d'
    n_boot = int(os.getenv('PBDMDC_N_BOOT', '2000'))
    sig_markers = os.getenv('PBDMDC_SIG_MARKERS', '1') != '0'

    assert not np.ma.is_masked(truth)
    truth = np.ma.getdata(truth)
    norm_const = truth.std()

    preds_all = {method: np.array(timeseries_model[method]['preds']) for method in methods}
    n_members = {preds.shape[0] for preds in preds_all.values()}
    assert len(n_members) == 1, f'Member counts differ across methods: {n_members}'
    n_members = n_members.pop()

    # The paired comparison needs member i to be the same realization for every method.
    member_names = [timeseries_model[method].get('data_names') for method in methods]
    paired = all(names is not None for names in member_names)
    if paired:
        assert all(list(names) == list(member_names[0]) for names in member_names), \
            'Member order differs across methods'
    else:
        print('  WARNING: no data_names in timeseries pickle, member alignment unknown; '
              'skipping significance test (rerun evaluation.compute_gm_timeseries)', flush=True)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_members, size=(n_boot, n_members))

    t_arr = np.array(time_index)

    mask_volc = np.array([is_in_volcanic(t) for t in t_arr])

    period_early = (t_arr >= pd.Timestamp('1850-01-01')) & (t_arr < pd.Timestamp('1980-01-01'))
    mask_early   = period_early & ~mask_volc

    period_late  = (t_arr >= pd.Timestamp('1980-01-01')) & (t_arr < pd.Timestamp('2015-01-01'))
    mask_late    = period_late & ~mask_volc

    group_labels  = ['Volcanic\nintervals', f'{time_index[0].year}–1980\n(non-volc.)', f'1980–{time_index[-1].year}\n(non-volc.)']
    group_masks   = [mask_volc, mask_early, mask_late]
    n_groups      = len(group_labels)
    n_methods     = len(methods)

    # Point estimates and bootstrap replicates for every (method, period) before drawing,
    # so the reference replicates are available for the paired comparison.
    stats = {}
    for method in methods:
        preds = preds_all[method]
        for gi, mask in enumerate(group_masks):
            if not mask.any():
                continue
            point, boot = member_bootstrap_rmsd(preds, truth, mask, norm_const, idx)
            assert np.isclose(point, np.sqrt(((preds[:, mask] - truth[None, mask]) ** 2).mean()) / norm_const * 100.)
            member_rmsd = np.sqrt(((preds[:, mask] - truth[None, mask]) ** 2).mean(axis=1)) / norm_const * 100.
            stats[method, gi] = (point, boot, member_rmsd)

    ax_bar.grid(axis='y', alpha=0.3)
    ax_bar.set_axisbelow(True)

    bar_width     = 0.8 / n_methods
    group_centers = np.arange(n_groups)

    rows = []
    ci_top = 0.0
    for mi, method in enumerate(methods):
        x_pos = group_centers + (mi - (n_methods - 1) / 2) * bar_width
        bar_vals, err_lo, err_hi, stars = [], [], [], []
        for gi in range(n_groups):
            if (method, gi) not in stats:
                bar_vals.append(0.0)
                err_lo.append(0.0)
                err_hi.append(0.0)
                continue
            point, boot, member_rmsd = stats[method, gi]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            bar_vals.append(point)
            err_lo.append(point - lo)
            err_hi.append(hi - point)
            ci_top = max(ci_top, hi)

            diff_vs_ref, diff_lo, diff_hi, ref_better = np.nan, np.nan, np.nan, None
            if method == ref_method:
                diff_vs_ref = 0.0
            elif (ref_method, gi) in stats:
                diff_vs_ref = point - stats[ref_method, gi][0]
                if paired:
                    diff_lo, diff_hi, ref_better = paired_difference(boot, stats[ref_method, gi][1])
                    if ref_better:
                        stars.append((x_pos[gi], hi))

            member_lo, member_hi = np.percentile(member_rmsd, [2.5, 97.5])
            rows.append({'var': var, 'period': group_labels[gi].replace('\n', ' '), 'method': method,
                         'n_members': n_members, 'rmsd_pct': point, 'ci_lo': lo, 'ci_hi': hi,
                         'ref_method': ref_method, 'diff_vs_ref': diff_vs_ref,
                         'diff_ci_lo': diff_lo, 'diff_ci_hi': diff_hi,
                         'ref_significantly_better': ref_better,
                         'member_p2_5': member_lo, 'member_p97_5': member_hi})

        ax_bar.bar(
            x_pos, bar_vals,
            yerr=[err_lo, err_hi],
            width=bar_width * 0.9,
            color=colors[method],
            edgecolor='black',
            linewidth=0.4,
            zorder=1,
            error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor='black'),
        )
        if sig_markers:
            for x_star, y_star in stars:
                ax_bar.annotate('*', xy=(x_star, y_star), xytext=(0, 0.5), textcoords='offset points',
                                ha='center', va='bottom', fontsize=8)

    if ci_top > 0:
        ax_bar.set_ylim(0, ci_top * 1.12)
    ax_bar.set_xticks(group_centers)
    ax_bar.set_xticklabels(group_labels, fontsize=tick_fontsize)
    ax_bar.tick_params(axis='x', labelsize=tick_fontsize)
    ax_bar.tick_params(axis='y', labelsize=tick_fontsize)

    return rows



def plot_timeseries_v1(timeseries, esms, method_names, colors, clim_vars_on_fig, method_names_on_fig, var):
    n_models = len(esms)
    n_cols = 4
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 6 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    method_order_on_fig = method_names
    alphas = {'RegGMST': 0.3, 'LR': 1,'LIM': 0.7,'LIM-opt': 1,
                       'PullbackDMDc': 1, 'PullbackDMDc-2d': 1, 'PullbackDMDc-3d': 1}
    methods_plot_order = [m for m in method_order_on_fig if m in method_names]
    errorbar_methods = {'LIM'}#{'LIM', 'LIM-opt'}
    markers = {'LIM': None, 'LIM-opt': None}

    for j, model in enumerate(esms):
        print(f'\nPlotting model: {model}', flush=True)

        row, col = divmod(j, n_cols)
        ax = axes[row, col]

        truth_gm = timeseries[model][method_names[0]]['truth']
        time_index = timeseries[model][method_names[0]]['time']


        for method in methods_plot_order:
            print(f'  Plotting method: {method}', flush=True)
            color = colors[method]
            preds = np.array(timeseries[model][method]['preds'])
            print(truth_gm.shape, preds.shape)

            # Compute continuous RMSD across all predictions and time steps, normalized by truth std
            preds_rmsd_continuous = np.sqrt(((preds - truth_gm[None, :])**2).mean(axis=0))/ truth_gm.std(axis=0)


            pred_mean = np.percentile(preds, 50, axis=0)#np.mean(preds, axis=0)
            pred_min  = np.percentile(preds, 0, axis=0)#np.min(preds, axis=0)
            pred_max  = np.percentile(preds, 100, axis=0)#np.max(preds, axis=0)

            if method in errorbar_methods:
                yerr = np.array([pred_mean - pred_min, pred_max - pred_mean])
                marker = markers[method]
                ax.errorbar(
                    time_index, pred_mean,
                    yerr=yerr,
                    color=color,
                    marker=marker,
                    markersize=4,
                    markerfacecolor=color,
                    markeredgecolor='black',
                    markeredgewidth=0.5,
                    linewidth=1.0,
                    elinewidth=0.5,
                    capsize=2,
                    zorder=1,
                    alpha=alphas[method],
                )
            else:
                ax.fill_between(time_index, pred_min, pred_max, color=color, alpha=alphas[method])

        ax.plot(time_index, truth_gm, '--',color='black', linewidth=2.0)
        #for model1 in esms:
        #    ax.plot(time_index, timeseries[model1][method_names[0]]['truth'],'--', color='gray', linewidth=1.0)

        ax.set_title(f'{display_model_name(model)}: Global mean {clim_vars_on_fig[var]}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_index[0], time_index[-1])
        ax.tick_params(axis='x', labelrotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    # Hide unused axes
    for j in range(n_models, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].set_visible(False)

    # Build shared legend in original method_names order
    legend_handles = []
    for method in method_names:
        color = colors[method]
        if method in errorbar_methods:
            marker = markers[method]
            legend_handles.append(
                mlines.Line2D([], [], color=color, marker='|',
                              markersize=6, markeredgecolor='black', markeredgewidth=0.5,
                              linewidth=1.0, label=method_names_on_fig[method], alpha=alphas[method])
            )
        else:
            legend_handles.append(
                mlines.Line2D([], [], color=color, linewidth=8, alpha=alphas[method], label=method_names_on_fig[method])
            )
    legend_handles.append(
        mlines.Line2D([], [], color='black', linewidth=3.0, label='Ground truth')
    )

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncols=4,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.1),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    return fig, axes

# Main volcanic activity:
# Rank 1 - Krakatau, Indonesia - Aug 1883 - Period: 1883-1886 - Peak Forcing: ~-1.9
# Rank 2 - Pinatubo, Philippines - Jun 1991 - Period: 1991-1994 - Peak Forcing: ~-1.7
# Rank 3 - Katmai / Novarupta, Alaska - Jun 1912 - Period: 1912-1915 - Peak Forcing: ~-0.7
# Rank 4 - Santa María, Guatemala - Oct 1902 - Period: 1902-1904 - Peak Forcing: ~-0.6
# Rank 5 - Agung, Indonesia - Mar 1963 - Period: 1963-1966 - Peak Forcing: ~-0.9
# Rank 6 - El Chichón, Mexico - Mar-Apr 1982 - Period: 1982-1984 - Peak Forcing: ~-0.3
# Rank 7 - Tarawera / Ruapehu cluster or Mt. St. Helens area - ~1866-1870 - Period: 1866-1870 - Peak Forcing: ~-0.6

def plot_timeseries_v2(timeseries, esms, method_names, colors, clim_vars_on_fig,
                    method_names_on_fig, var,
                    interval_years=(1883, 1888, 1991, 1996, 1912, 1917, 1902, 1907, 1963, 1968, 1982, 1987, 1861, 1866, 1940),
                    panel_height_ratios=(2, 1, 1)):
    n_models = len(esms)
    n_cols = 4
    n_rows = math.ceil(n_models / n_cols)
    interval_years = sorted(interval_years)
    window_size = 12 * 3

    # Top 7 volcanic events: (start_year, end_year, label)
    volcanic_events = [
        (1861, 1866, 'Makian'),
        (1883, 1888, 'Krakatoa'),
        (1902, 1907, 'Santa María'),
        (1912, 1917, 'Novarupta'),
        (1963, 1968, 'Agung'),
        (1982, 1987, 'El Chichón'),
        (1991, 1996, 'Pinatubo'),
    ]
    volcanic_color = '#b0c4de'  # light steel blue

    fig, axes = plt.subplots(
        n_rows * 3, n_cols,
        figsize=(5 * n_cols, sum(panel_height_ratios) / 5 * 10 * n_rows),
        sharex=True,
        squeeze=False,
        gridspec_kw={'height_ratios': list(panel_height_ratios) * n_rows},
    )

    method_order_on_fig = method_names
    alphas = {'RegGMST': 0.3, 'LR': 1,'LIM': 0.7,'LIM-opt': 1,
                       'PullbackDMDc': 1, 'PullbackDMDc-2d': 1, 'PullbackDMDc-3d': 1}
    methods_plot_order = [m for m in method_order_on_fig if m in method_names]
    errorbar_methods = {'LIM'}
    markers = {'LIM': None, 'LIM-opt': None}

    interval_bounds = [pd.Timestamp(f'{y}-01-01') for y in interval_years]

    def draw_volcanic_stripes(ax):
        for v_start, v_end, v_label in volcanic_events:
            ax.axvspan(
                pd.Timestamp(f'{v_start}-01-01'),
                pd.Timestamp(f'{v_end}-01-01'),
                color=volcanic_color, alpha=0.35, zorder=0, linewidth=0,
            )

    for j, model in enumerate(esms):
        print(f'\nPlotting model: {model}', flush=True)

        row_base = (j // n_cols) * 3
        col = j % n_cols

        ax_ts    = axes[row_base,     col]
        ax_rmsd  = axes[row_base + 1, col]
        ax_stair = axes[row_base + 2, col]

        truth_gm   = timeseries[model][method_names[0]]['truth']
        time_index = timeseries[model][method_names[0]]['time']
        norm_const = truth_gm.std()

        # ── Top panel: time series ──────────────────────────────────────────
        draw_volcanic_stripes(ax_ts)

        for method in methods_plot_order:
            color = colors[method]
            preds = np.array(timeseries[model][method]['preds'])

            pred_mean = np.percentile(preds, 50, axis=0)
            pred_min  = np.percentile(preds,  0, axis=0)
            pred_max  = np.percentile(preds, 100, axis=0)
            alpha     = alphas.get(method, 1.0)

            if method in errorbar_methods:
                yerr = np.array([pred_mean - pred_min, pred_max - pred_mean])
                ax_ts.errorbar(
                    time_index, pred_mean, yerr=yerr,
                    fmt='none',
                    color=color, marker=markers[method],
                    markersize=4, markerfacecolor=color,
                    markeredgecolor='black', markeredgewidth=0.5,
                    linewidth=1.0, elinewidth=0.5, capsize=2,
                    zorder=1, alpha=alpha,
                )
            else:
                ax_ts.fill_between(time_index, pred_min, pred_max,
                                   color=color, alpha=alpha, zorder=1)

        ax_ts.plot(time_index, truth_gm, '--', color='black', linewidth=2.0, zorder=2)
        ax_ts.set_title(f'{display_model_name(model)}: GM {clim_vars_on_fig[var]}, time series',
                        fontsize=14)
        ax_ts.grid(True, alpha=0.3)
        ax_ts.set_xlim(time_index[0], time_index[-1])
        ax_ts.tick_params(axis='x', labelrotation=45, labelsize=12)
        ax_ts.tick_params(axis='y', labelsize=12)

        # ── Middle panel: continuous RMSD ───────────────────────────────────
        draw_volcanic_stripes(ax_rmsd)
        ax_rmsd.grid(True, alpha=0.3)
        ax_rmsd.set_axisbelow(True)
        half_w = window_size // 2
        n_t = preds.shape[1]
        for method in methods_plot_order:
            color = colors[method]
            preds = np.array(timeseries[model][method]['preds'])
            rmsd = np.array([
                np.sqrt(((preds[:, max(0, t - half_w):min(n_t, t + half_w + 1)]
                          - truth_gm[None, max(0, t - half_w):min(n_t, t + half_w + 1)]) ** 2).mean())
                for t in range(n_t)
            ]) / norm_const

            ax_rmsd.plot(time_index, rmsd * 100., color=color,
                         linewidth=1.2, alpha=alphas.get(method, 1.0), zorder=1)

        ax_rmsd.set_title(f'{display_model_name(model)}: GM {clim_vars_on_fig[var]}, 3yr RMSD, %',
                        fontsize=14)
        ax_rmsd.grid(True, alpha=0.3)
        ax_rmsd.tick_params(axis='x', labelrotation=45, labelsize=10)
        ax_rmsd.tick_params(axis='y', labelsize=10)

        # ── Bottom panel: staircase RMSD ────────────────────────────────────
        draw_volcanic_stripes(ax_stair)

        all_bounds = [time_index[0]] + interval_bounds + [time_index[-1]]
        interval_edges = list(zip(all_bounds[:-1], all_bounds[1:]))

        for method in methods_plot_order:
            color = colors[method]
            preds = np.array(timeseries[model][method]['preds'])

            stair_x, stair_y = [], []
            for t_start, t_end in interval_edges:
                mask = (time_index >= t_start) & (time_index < t_end)
                if not mask.any():
                    continue
                rmsd_interval = np.sqrt(
                    ((preds[:, mask] - truth_gm[None, mask]) ** 2).mean()
                ) / norm_const
                stair_x += [t_start, t_end]
                stair_y += [rmsd_interval, rmsd_interval]

            ax_stair.plot(stair_x, np.array(stair_y) * 100., color=color,
                          linewidth=1.5, alpha=alphas.get(method, 1.0),
                          drawstyle='default', zorder=1)

        for b in interval_bounds:
            ax_rmsd.axvline(b, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)
            ax_stair.axvline(b, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)

        #ax_stair.set_ylabel('RMSD by intervals (%)', fontsize=10)
        ax_stair.set_title(f'{display_model_name(model)}: GM {clim_vars_on_fig[var]}, accumulated RMSD, %',
                        fontsize=14)
        ax_stair.grid(True, alpha=0.3)
        ax_stair.tick_params(axis='x', labelrotation=45, labelsize=10)
        ax_stair.tick_params(axis='y', labelsize=10)

    # Hide unused axis triplets
    for j in range(n_models, n_rows * n_cols):
        row_base = (j // n_cols) * 3
        col = j % n_cols
        for k in range(3):
            axes[row_base + k, col].set_visible(False)

    # Shared legend
    legend_handles = []
    for method in method_names:
        color = colors[method]
        alpha = alphas.get(method, 1.0)
        if method in errorbar_methods:
            legend_handles.append(
                mlines.Line2D([], [], color=color, marker='|',
                              markersize=6, markeredgecolor='black',
                              markeredgewidth=0.5, linewidth=1.0,
                              label=method_names_on_fig[method], alpha=alpha)
            )
        else:
            legend_handles.append(
                mlines.Line2D([], [], color=color, linewidth=8,
                              alpha=alpha, label=method_names_on_fig[method])
            )
    legend_handles.append(
        mlines.Line2D([], [], color='black', linewidth=2.0,
                      linestyle='--', label='Ground truth')
    )
    legend_handles.append(
        mpatches.Patch(color=volcanic_color, alpha=0.35, label='Volcanic event')
    )

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncols=5,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    return fig, axes

def plot_timeseries(timeseries, esms, method_names, colors, clim_vars_on_fig,
                    method_names_on_fig, var,
                    panel_height_ratios=(2, 1, 1)):
    n_models = len(esms)
    n_cols = 4
    n_rows = math.ceil(n_models / n_cols)
    window_size = 12 * 3

    volcanic_color = '#b0c4de'

    # Keep x independent because row 3 uses categorical x; do not share y-axis
    # so each panel has independent y-scaling (especially for bottom row bar charts).
    # Add a spacer row between rows 2 and 3 so the shared x-axis of row 2 is visually separated.
    spacer_height = 0.55
    tick_fontsize = 10
    height_pattern = [panel_height_ratios[0], panel_height_ratios[1], spacer_height, panel_height_ratios[2]]
    fig, axes = plt.subplots(
        n_rows * 4, n_cols,
        figsize=(5 * n_cols, sum(panel_height_ratios) / 5 * 10 * n_rows),
        sharex=False,
        sharey=False,
        squeeze=False,
        gridspec_kw={'height_ratios': height_pattern * n_rows},
    )

    method_order_on_fig = method_names
    alphas = {'RegGMST': 1, 'LR': 1, 'LIM': 1, 'LIM-opt': 1,
              'PullbackDMDc': 1, 'PullbackDMDc-2d': 1, 'PullbackDMDc-3d': 1}
    mid_linewidth = 1.0
    methods_plot_order = [m for m in method_order_on_fig if m in method_names]
    errorbar_methods = {'LIM'}
    markers = {'LIM': None, 'LIM-opt': None}
    # Use Taylor-diagram markers on the middle-panel method curves.
    middle_markers = {
        method: method_markers.get(method, None)
        for method in methods_plot_order
    }

    top_bounds = []
    middle_max_bounds = []

    def draw_volcanic_stripes(ax):
        for v_start, v_end, _ in volcanic_events:
            ax.axvspan(
                pd.Timestamp(f'{v_start}-01-01'),
                pd.Timestamp(f'{v_end}-01-01'),
                color=volcanic_color, alpha=0.35, zorder=0, linewidth=0,
            )

    ci_rows = []
    for j, model in enumerate(esms):
        print(f'\nPlotting model: {model}', flush=True)

        row_base = (j // n_cols) * 4
        col = j % n_cols

        ax_ts    = axes[row_base,     col]
        ax_rmsd  = axes[row_base + 1, col]
        ax_bar   = axes[row_base + 3, col]

        # Share x between first and second panels within each model triplet.
        ax_rmsd.sharex(ax_ts)
        # Share y only for top two rows within each row block; keep bars independent.
        if col > 0:
            ax_ts.sharey(axes[row_base, 0])
            ax_rmsd.sharey(axes[row_base + 1, 0])

        truth_gm   = timeseries[model][method_names[0]]['truth']
        time_index = timeseries[model][method_names[0]]['time']
        norm_const = truth_gm.std()

        # ── Top panel: time series ──────────────────────────────────────────
        draw_volcanic_stripes(ax_ts)
        ax_ts.grid(True, alpha=0.3)
        ax_ts.set_axisbelow(True)

        for method in methods_plot_order:
            color = colors[method]
            preds = np.array(timeseries[model][method]['preds'])

            pred_mean = np.percentile(preds, 50, axis=0)
            pred_min  = np.percentile(preds,  0, axis=0)
            pred_max  = np.percentile(preds, 100, axis=0)
            alpha     = alphas.get(method, 1.0)
            alpha_top = 1

            ax_ts.fill_between(time_index, pred_min, pred_max,
                               color=color, alpha=alpha_top, zorder=1)

            if method != 'LIM':
                top_bounds.append((float(np.min([pred_min.min(), pred_max.min()])), float(np.max([pred_min.max(), pred_max.max()]))))

        ax_ts.plot(time_index, truth_gm, ':', color='black', linewidth=2.0, zorder=2)
        ax_ts.set_title(f'{display_model_name(model)}', fontsize=14, pad=8)
        ax_ts.set_xlim(time_index[0], time_index[-1])
        ax_ts.tick_params(axis='y', labelsize=tick_fontsize)
        if col == 0:
            ax_ts.set_ylabel(f'GM {clim_vars_on_fig[var]}', fontsize=11)
            # Prune boundary ticks to avoid overlap with the RMSD panel when hspace=0.
            ax_ts.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
        else:
            ax_ts.tick_params(axis='y', left=False, labelleft=False)

        # Hide top-panel x ticks so row 2 carries the shared x-axis labels.
        ax_ts.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labelsize=tick_fontsize)

        # ── Middle panel: continuous RMSD ───────────────────────────────────
        draw_volcanic_stripes(ax_rmsd)
        ax_rmsd.grid(True, alpha=0.3)
        ax_rmsd.set_axisbelow(True)

        half_w = window_size // 2
        n_t = len(time_index)
        for method in methods_plot_order:
            color = colors[method]
            preds = np.array(timeseries[model][method]['preds'])
            rmsd = np.array([
                np.sqrt(((preds[:, max(0, t - half_w):min(n_t, t + half_w + 1)]
                          - truth_gm[None, max(0, t - half_w):min(n_t, t + half_w + 1)]) ** 2).mean())
                for t in range(n_t)
            ]) / norm_const

            ax_rmsd.plot(time_index, rmsd * 100., color=color,
                         linewidth=mid_linewidth,
                         linestyle='-',
                         marker=middle_markers[method],
                         markevery=120,
                         markersize=5,
                         markerfacecolor=color,
                         markeredgecolor=color,
                         markeredgewidth=0.0,
                         alpha=alphas.get(method, 1.0), zorder=1)

            if method in {'LIM', 'LIM-opt'}:
                continue
            middle_max_bounds.append(float(np.max(rmsd * 100.)))

        ax_rmsd.tick_params(axis='x', labelrotation=45, labelsize=tick_fontsize)
        ax_rmsd.tick_params(axis='y', labelsize=tick_fontsize)
        ax_rmsd.set_xlim(time_index[0], time_index[-1])
        ax_rmsd.set_xlabel('Year', fontsize=10)
        if col == 0:
            ax_rmsd.set_ylabel('3yr RMSD (%)', fontsize=10)
            # Prune boundary ticks to avoid overlap with the top panel when hspace=0.
            ax_rmsd.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
        else:
            ax_rmsd.tick_params(axis='y', left=False, labelleft=False)

        
        # ── Bottom panel: grouped bar chart with bootstrap CIs ──────────────
        rows = draw_rmsd_bars(ax_bar, timeseries[model], methods_plot_order, truth_gm, time_index, var,
                              seed=20260 + j)
        for row in rows:
            row['model'] = model
        ci_rows += rows
        if col == 0:
            ax_bar.set_ylabel('RMSD by period (%)', fontsize=10)
        

        # Spacer row is intentionally blank.
        axes[row_base + 2, col].set_visible(False)

    # Hide unused axis triplets
    for j in range(n_models, n_rows * n_cols):
        row_base = (j // n_cols) * 4
        col = j % n_cols
        for k in range(4):
            axes[row_base + k, col].set_visible(False)

    def _expand_bounds(bounds, pad_frac=0.08):
        if not bounds:
            return None
        low = min(v[0] for v in bounds)
        high = max(v[1] for v in bounds)
        span = high - low
        pad = span * pad_frac if span > 0 else 1.0
        return low - pad, high + pad

    def _expand_upper_bound(values, pad_frac=0.08):
        if not values:
            return None
        upper = max(values)
        pad = upper * pad_frac if upper > 0 else 1.0
        return 0.0, upper + pad

    top_ylim = _expand_bounds(top_bounds)
    middle_ylim = _expand_upper_bound(middle_max_bounds)
    if top_ylim is not None:
        for j in range(n_models):
            row_base = (j // n_cols) * 4
            col = j % n_cols
            axes[row_base, col].set_ylim(*top_ylim)
    if middle_ylim is not None:
        for j in range(n_models):
            row_base = (j // n_cols) * 4
            col = j % n_cols
            axes[row_base + 1, col].set_ylim(*middle_ylim)

    # Shared legend
    legend_handles = []
    for method in method_names:
        color = colors[method]
        alpha = alphas.get(method, 1.0)
        legend_handles.append(
            mlines.Line2D(
                [], [],
                color=color,
                linewidth=3.0,
                linestyle='-',
                marker=method_markers.get(method, None),
                markersize=6,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.0,
                alpha=alpha,
                label=method_names_on_fig[method],
            )
        )
    legend_handles.append(
        mlines.Line2D([], [], color='black', linewidth=2.0, linestyle=':', label='Ground truth')
    )
    legend_handles.append(
        mlines.Line2D([], [], color='black', linestyle='none', marker='|', markersize=10,
                      markeredgewidth=1.0, label='95% CI (member bootstrap)')
    )
    if os.getenv('PBDMDC_SIG_MARKERS', '1') != '0':
        legend_handles.append(
            mlines.Line2D([], [], color='black', linestyle='none', marker='$*$', markersize=8,
                          label='PullbackDMDc significantly lower RMSD')
        )

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncols=4,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.03),
    )
    # Minimize the vertical gap so first two rows visually stack as a shared-x pair.
    fig.tight_layout(rect=[0, 0.06, 1, 1], h_pad=0.0)
    fig.subplots_adjust(hspace=0.0, wspace=0.10)

    return fig, axes, ci_rows

if __name__ == '__main__':

    for lag in requested_lags(lags):
        for start_year, end_year in requested_year_ranges(year_ranges):

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

            artifact_results_dir = os.path.join(artifact_root, 'evaluation_results', f'results{year_suffix}{lag_suffix}')
            pdf_results_dir = os.path.join(pdf_root, f'results{year_suffix}{lag_suffix}')
            os.makedirs(artifact_results_dir, exist_ok=True)
            os.makedirs(pdf_results_dir, exist_ok=True)

            for var in clim_vars:
                print(f'\n=== Plotting variable: {var} ===', flush=True)

                # Load saved timeseries
                in_path = os.path.join(artifact_results_dir, f'gm_timeseries_{var}.pkl')
                with open(in_path, 'rb') as f:
                    timeseries = pkl.load(f)

                fig, axes, ci_rows = plot_timeseries(timeseries, esms, method_names, colors, clim_vars_on_fig, method_names_on_fig, var)

                out_csv = os.path.join(pdf_results_dir, f'gm_{var}_rmsd_by_period_ci.csv')
                pd.DataFrame(ci_rows, columns=ci_columns).to_csv(out_csv, index=False)
                print(f'Saved: {out_csv}', flush=True)

                    # for ax_col in axes[:, j]:
                    #     ax_col.set_ylim(y_min, y_max)

                out_fig = os.path.join(pdf_results_dir, f'gm_{var}.pdf')
                plt.savefig(out_fig, dpi=300, bbox_inches='tight',transparent=True)
                plt.show()
                plt.close()
                print(f'Saved: {out_fig}', flush=True)
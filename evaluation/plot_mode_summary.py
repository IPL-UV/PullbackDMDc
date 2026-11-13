import os
import pickle

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from evaluation.utils.members import (SELECTED_HANDLE, acf_selected_index, mark_selected, method_files,
                                      selected_index)
from evaluation.utils.style import ESM_COLORS, TIMESCALE_BANDS, display_model_name, set_font_sizes
from evaluation.utils.timescales import zero_crossing_summary
from utils.params import artifact_root, esms, lag_suffix_for, method_markers, pdf_root, predictions_dir, year_suffix_for

# Prints at ~7/8/9/10 pt at \textwidth (figure saves ~11.9 in wide).
FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE = 13, 15, 17, 19
set_font_sizes(FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE)



def report_timescale_threshold_table(row1_timescales, top_n, obs_model='20CRv3', threshold_mode=2, mode_num=3):
    obs_members = row1_timescales.get(obs_model, [])
    if not obs_members:
        print(f'No {obs_model} timescale data available for reporting.')
        return

    # Collect per-mode values across any available obs members.
    obs_mode_values = []
    for mode_idx in range(top_n):
        vals = []
        for member_vals in obs_members:
            if member_vals.size <= mode_idx:
                continue
            v = float(member_vals[mode_idx])
            if np.isfinite(v):
                vals.append(v)
        obs_mode_values.append(np.asarray(vals, dtype=float))

    print('\n20CRv3 timescales (years) by mode:')
    for mode_idx, vals in enumerate(obs_mode_values, start=1):
        if vals.size == 0:
            print(f'  Mode {mode_idx}: NaN')
        elif vals.size == 1:
            print(f'  Mode {mode_idx}: {vals[0]:.3f}')
        else:
            members_txt = ', '.join(f'{x:.3f}' for x in vals)
            print(f'  Mode {mode_idx}: mean={np.mean(vals):.3f} (members: {members_txt})')

    threshold_idx = threshold_mode - 1
    if threshold_idx < 0 or threshold_idx >= top_n:
        print(f'Invalid threshold mode {threshold_mode}; expected 1..{top_n}.')
        return

    threshold_vals = obs_mode_values[threshold_idx]
    if threshold_vals.size == 0:
        print(f'Cannot build threshold table: 20CRv3 Mode {threshold_mode} is NaN.')
        return

    mode1_threshold = np.nan
    if top_n >= 1 and obs_mode_values[0].size > 0:
        mode1_threshold = float(np.mean(obs_mode_values[0]))

    mode2_threshold = np.nan
    if top_n >= 2 and obs_mode_values[1].size > 0:
        mode2_threshold = float(np.mean(obs_mode_values[1]))

    print(f'\nPercent of finite mode {mode_num} timescales per model above 20CRv3 thresholds:')
    pct_header = ('Model', f'% >= Mode 1', f'% >= Mode 2', 'N finite')
    pct_col_widths = [17, 14, 14, 10]
    print(
        f'{pct_header[0]:<{pct_col_widths[0]}}'
        f'{pct_header[1]:<{pct_col_widths[1]}}'
        f'{pct_header[2]:<{pct_col_widths[2]}}'
        f'{pct_header[3]:<{pct_col_widths[3]}}'
    )
    print('-' * sum(pct_col_widths))

    for model_name in esms:
        finite_vals = []
        for member_vals in row1_timescales.get(model_name, []):
            ts = float(member_vals[mode_num - 1]) if member_vals.size >= mode_num else np.nan
            if np.isfinite(ts):
                    finite_vals.append(ts)

        n_finite = len(finite_vals)
        if n_finite == 0:
            pct_mode1_txt = 'NaN'
            pct_mode2_txt = 'NaN'
        else:
            vals = np.asarray(finite_vals, dtype=float)
            if np.isfinite(mode1_threshold):
                pct_mode1_txt = f'{100.0 * np.mean(vals >= mode1_threshold):.1f}'
            else:
                pct_mode1_txt = 'NaN'

            if np.isfinite(mode2_threshold):
                pct_mode2_txt = f'{100.0 * np.mean(vals >= mode2_threshold):.1f}'
            else:
                pct_mode2_txt = 'NaN'

        print(
            f'{model_name:<{pct_col_widths[0]}}'
            f'{pct_mode1_txt:<{pct_col_widths[1]}}'
            f'{pct_mode2_txt:<{pct_col_widths[2]}}'
            f'{n_finite:<{pct_col_widths[3]}}'
        )

    threshold = float(np.mean(threshold_vals))
    print(f'\nThreshold = 20CRv3 Mode {threshold_mode} timescale = {threshold:.3f} years')

    rows = []
    for model_name in esms:
        for member_idx, member_vals in enumerate(row1_timescales.get(model_name, []), start=1):
            for mode_idx in range(top_n):
                if member_vals.size <= mode_idx:
                    continue
                ts = float(member_vals[mode_idx])
                if np.isfinite(ts) and ts >= threshold:
                    rows.append((model_name, member_idx, mode_idx + 1, ts))

    print(f'\nModes with timescale >= 20CRv3 Mode {threshold_mode} (years):')
    if not rows:
        print('  None')
        return

    header = ('Model', 'Member', 'Mode', 'Timescale (years)')
    col_widths = [17, 8, 6, 18]
    print(
        f'{header[0]:<{col_widths[0]}}'
        f'{header[1]:<{col_widths[1]}}'
        f'{header[2]:<{col_widths[2]}}'
        f'{header[3]:<{col_widths[3]}}'
    )
    print('-' * sum(col_widths))
    for model_name, member_idx, mode_idx, ts in rows:
        print(
            f'{model_name:<{col_widths[0]}}'
            f'{member_idx:<{col_widths[1]}}'
            f'{mode_idx:<{col_widths[2]}}'
            f'{ts:<{col_widths[3]}.3f}'
        )


def gather_row2_row3_metrics(base_pred_dir, clim_var, chosen_method, top_n, smooth_sigma):
    smoothed_var_data = {model: [] for model in esms}
    forced_fraction_data = {model: [] for model in esms}

    for chosen_model in esms:
        model_dir = os.path.join(base_pred_dir, clim_var, chosen_model)
        if not os.path.isdir(model_dir):
            print(f'Skipping missing model directory: {model_dir}')
            continue

        dmd_files = sorted([
            os.path.join(model_dir, fn)
            for fn in os.listdir(model_dir)
            if fn.split('_')[-1].split('.')[0] == chosen_method
        ])

        if not dmd_files:
            print(f'No {chosen_method} files for {clim_var} / {chosen_model}')
            continue

        for dmd_file in dmd_files:
            with open(dmd_file, 'rb') as fh:
                dmd = pickle.load(fh)
            dmd.compute_modes()

            rotated = dmd.compute_rotated_modes(top_n=top_n)
            eigvals = rotated['eigvals']
            modes = rotated['spatial_patterns']
            internal_ts = rotated['internal_time_series']
            labels = rotated['labels']
            plot_n = min(top_n, eigvals.size)

            if eigvals.size == 0:
                continue

            mode_vars = [np.nan] * top_n
            for k in range(plot_n):
                z_smooth = gaussian_filter1d(internal_ts[:, k], sigma=smooth_sigma)
                if labels[k][1] == 'complex':
                    mode_vars[k] = float(np.var(z_smooth))
                else:
                    spatial_norm_sq = float(np.sum(modes[:, k] ** 2))
                    mode_vars[k] = spatial_norm_sq * float(np.var(z_smooth))

            smoothed_var_data[chosen_model].append(np.asarray(mode_vars, dtype=float))

            with np.errstate(divide='ignore', invalid='ignore'):
                ff_vals = np.var(rotated['forced_time_series'], axis=0) / np.var(
                    rotated['internal_time_series'] + rotated['forced_time_series'], axis=0
                )
            forced_fraction = np.full(top_n, np.nan, dtype=float)
            n_ff = min(top_n, ff_vals.size)
            forced_fraction[:n_ff] = ff_vals[:n_ff]
            forced_fraction[~np.isfinite(forced_fraction)] = np.nan
            forced_fraction_data[chosen_model].append(forced_fraction)

    return smoothed_var_data, forced_fraction_data


def row2_reference(row2_smoothed_var, obs_model='20CRv3'):
    """Row 2 reference level: the larger of the reanalysis modes 1 and 2."""
    return float(np.nanmax(row2_smoothed_var[obs_model][0][:2]))


def row2_exceedance_table(row2_smoothed_var, ref, top_n, obs_model='20CRv3'):
    """Per ESM: members whose smoothed internal variance reaches ref, by mode and over modes 1-2 / 1-4."""
    rows = []
    for model, member_sets in row2_smoothed_var.items():
        if model == obs_model or not member_sets:
            continue
        reach = np.vstack(member_sets)[:, :top_n] >= ref  # NaN compares False
        row = {'model': model, 'n_members': reach.shape[0]}
        row.update({f'mode{k + 1}': int(reach[:, k].sum()) for k in range(top_n)})
        row['any_mode_1_2'] = int(reach[:, :2].any(axis=1).sum())
        row[f'any_mode_1_{top_n}'] = int(reach.any(axis=1).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary_figure(
    row1_timescales,
    row2_smoothed_var,
    row3_forced_fraction,
    clim_var,
    chosen_method,
    results_dir,
    smooth_sigma,
    selected_acf,
    selected_sorted,
):
    """selected_acf / selected_sorted: model -> member index of the member shown in the mode figures, in
    ACF-pickle order (row 1) and sorted-file order (rows 2-3)."""
    top_n = 4
    rng = np.random.default_rng(0)

    obs_model = '20CRv3'
    method_marker = method_markers.get('PullbackDMDc-3d', 'o')
 
    positions = np.arange(1, len(esms) + 1)
    tick_labels = [display_model_name(model) for model in esms]
    row2_ref = row2_reference(row2_smoothed_var, obs_model)

    fig, axes = plt.subplots(
        3,
        top_n,
        figsize=(12, 2.7 * 3),
        constrained_layout=True,
        sharex='row',
    )

    # PCs project the sqrt(cos lat)-weighted anomalies onto unit-norm EOFs, so they carry the field's units.
    field_unit = {'tas_ocean': 'K', 'tas': 'K', 'psl': 'Pa'}[clim_var]
    row_specs = [
        {
            'title': 'Dominant Time Scale\n(years)',
            'data': row1_timescales,
            'selected': selected_acf,
            'xscale': 'linear',
            'xlabel': 'Years',
            'valid': lambda x: np.isfinite(x) & (x > 0),
            'xlim': None,
        },
        {
            'title': f'Variance of Smoothed\nInternal Variability ({field_unit}$^2$)',
            'data': row2_smoothed_var,
            'selected': selected_sorted,
            # Log for wide-spanning members; tas_ocean is shown linear and capped at 100.
            'xscale': 'linear' if clim_var == 'tas_ocean' else 'log',
            'xlabel': rf'Var of smoothed $w\,\tilde{{z}}(t)$ [{field_unit}$^2$]',
            'valid': lambda x: np.isfinite(x) & (x > 0),
            'xlim': (0.0, 100.0) if clim_var == 'tas_ocean' else None,
        },
        {
            'title': 'Fraction of Variance\nForced Timeseries',
            'data': row3_forced_fraction,
            'selected': selected_sorted,
            'xscale': 'linear',
            'xlabel': 'Fraction',
            'valid': lambda x: np.isfinite(x),
            'xlim': (0.0, 1.0),
        },
    ]

    for row_idx, row_cfg in enumerate(row_specs):
        for mode_idx in range(top_n):
            ax = axes[row_idx, mode_idx]

            if row_idx == 0:
                for band_name, band_cfg in TIMESCALE_BANDS.items():
                    ax.axvspan(
                        band_cfg['xlim'][0], band_cfg['xlim'][1],
                        color=band_cfg['color'], alpha=band_cfg['alpha'], zorder=0,
                    )
                    # label_x = np.sqrt(band_cfg['xlim'][0] * band_cfg['xlim'][1])
                    # ax.text(
                    #     label_x, 0.97, band_name,
                    #     ha='center', va='top', fontsize=9,
                    #     color=band_cfg['color'],
                    #     transform=ax.get_xaxis_transform(),
                    # )

            for pos, chosen_model in zip(positions, esms):
                metric_members = row_cfg['data'].get(chosen_model, [])
                values = np.array(
                    [metric_by_mode[mode_idx] for metric_by_mode in metric_members if metric_by_mode.size > mode_idx],
                    dtype=float,
                )
                valid = row_cfg['valid'](values)
                if not valid.any():
                    continue

                jitter = rng.uniform(-0.12, 0.12, size=valid.sum())
                marker = method_marker

                ax.scatter(
                    values[valid],
                    np.full(valid.sum(), pos) + jitter,
                    color=ESM_COLORS.get(chosen_model, 'tab:gray'),
                    marker=marker,
                    s=36,
                    alpha=1.0 if chosen_model == obs_model else 0.65,
                    linewidths=0.9 if chosen_model == obs_model else 0.3,
                    edgecolor='none',
                    zorder=3,
                )

                sel = row_cfg['selected'].get(chosen_model)
                if sel is not None and metric_members[sel].size > mode_idx:
                    sel_value = np.array([metric_members[sel][mode_idx]], dtype=float)
                    sel_valid = row_cfg['valid'](sel_value)
                    if sel_valid[0]:
                        mark_selected(ax, sel_value[0], pos, 0.18)

            if row_idx == 1:
                ax.axvline(row2_ref, ls='--', color='black', lw=1.0, zorder=2)

            ax.set_xscale(row_cfg['xscale'])
            if row_cfg['xlim'] is not None:
                ax.set_xlim(*row_cfg['xlim'])
            ax.grid(True, axis='x', which='both', alpha=0.25)
            ax.set_yticks(positions)
            if mode_idx == 0:
                ax.set_yticklabels(tick_labels)
            else:
                ax.set_yticklabels([])

            if row_idx == 0:
                ax.set_title(f'Mode {mode_idx + 1}', fontsize=FS_TITLE)
            if mode_idx == 0:
                ax.set_ylabel(row_cfg['title'], fontsize=FS_LABEL)

    member_handle = plt.scatter(
        [], [], marker=method_marker, s=24,
        color='gray', alpha=0.65, edgecolors='white', linewidths=0.3,
        label='Individual member',
    )
    obs_handle = plt.scatter(
        [], [], marker=method_marker, s=28,
        color='black', edgecolors='none', linewidths=0.9,
        label='20CRv3 member',
    )

    ref_handle = mlines.Line2D([], [], ls='--', color='black', lw=1.0, label=f'{obs_model} mode 1 (Row 2)')
    fig.legend(handles=[SELECTED_HANDLE, ref_handle], loc='upper center', bbox_to_anchor=(0.5, 0.0),
               ncol=2, frameon=False, fontsize=FS_TICK)

    out_path = os.path.join(
        results_dir,
        f'mode_summary_three_rows_{clim_var}_{chosen_method}_sigma{smooth_sigma}.pdf',
    )
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    lag = 3
    start_year, end_year = 1850, 2014

    lag_suffix, year_suffix = lag_suffix_for(lag), year_suffix_for((start_year, end_year))

    clim_vars = ['tas', 'tas_ocean','psl']
    
    top_n = 4
    dt_years = 1.0 / 12.0
    smooth_sigma = 12 * 8 # eight years

    base_pred_dir = predictions_dir(year_suffix, lag_suffix)
    artifact_results_dir = os.path.join(artifact_root, 'evaluation_results', f'results{year_suffix}{lag_suffix}')
    pdf_results_dir = os.path.join(pdf_root, f'results{year_suffix}{lag_suffix}')
    os.makedirs(artifact_results_dir, exist_ok=True)
    os.makedirs(pdf_results_dir, exist_ok=True)

    for clim_var in clim_vars:
        if clim_var == 'psl':
            chosen_method = 'PullbackDMDc-2d'
        else:
            chosen_method = 'PullbackDMDc-3d'
        acf_path = os.path.join(artifact_results_dir, f'acfs_{clim_var}_{chosen_method}.pkl')
        if not os.path.exists(acf_path):
            print(f'Skipping {clim_var}: missing ACF file {acf_path}')
            continue

        with open(acf_path, 'rb') as fh:
            acf_data = pickle.load(fh)

        row1_timescales = {}
        for chosen_model in esms:
            member_acfs = acf_data.get(chosen_model, [])
            row1_timescales[chosen_model] = zero_crossing_summary(
                member_acfs=member_acfs,
                top_n=top_n,
                dt_years=dt_years,
            )

        report_timescale_threshold_table(
            row1_timescales=row1_timescales,
            top_n=top_n,
            obs_model='20CRv3',
            threshold_mode=2,
        )

        row2_smoothed_var, row3_forced_fraction = gather_row2_row3_metrics(
            base_pred_dir=base_pred_dir,
            clim_var=clim_var,
            chosen_method=chosen_method,
            top_n=top_n,
            smooth_sigma=smooth_sigma,
        )

        # Member shown in the mode figures: ACF rows follow os.listdir order, rows 2-3 sorted file order.
        selected_acf, selected_sorted = {}, {}
        for chosen_model in esms:
            model_dir = os.path.join(base_pred_dir, clim_var, chosen_model)
            if acf_data.get(chosen_model):
                selected_acf[chosen_model] = acf_selected_index(
                    model_dir, chosen_method, chosen_model, acf_data[chosen_model],
                    top_n=200 if clim_var == 'psl' else 20)
            files = method_files(model_dir, chosen_method, sort=True)
            assert len(files) == len(row2_smoothed_var[chosen_model]), f'{chosen_model}: member count mismatch'
            selected_sorted[chosen_model] = selected_index(files, chosen_model)

        out_path = plot_summary_figure(
            row1_timescales=row1_timescales,
            row2_smoothed_var=row2_smoothed_var,
            row3_forced_fraction=row3_forced_fraction,
            clim_var=clim_var,
            chosen_method=chosen_method,
            results_dir=pdf_results_dir,
            smooth_sigma=smooth_sigma,
            selected_acf=selected_acf,
            selected_sorted=selected_sorted,
        )
        print(f'Saved: {out_path}')

        ref = row2_reference(row2_smoothed_var)
        exceedance = row2_exceedance_table(row2_smoothed_var, ref, top_n)
        print(f'{clim_var}: members with smoothed internal variance >= 20CRv3 mode 1 ({ref:.4g})\n'
              f'{exceedance.to_string(index=False)}', flush=True)
        exceedance.to_csv(os.path.join(pdf_results_dir, f'mode_summary_row2_exceedance_{clim_var}.csv'), index=False)


if __name__ == '__main__':
    main()

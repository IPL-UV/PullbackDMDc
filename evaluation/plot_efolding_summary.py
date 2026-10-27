import os
import pickle
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from evaluation.utils.members import (SELECTED_HANDLE, acf_selected_index, mark_selected, method_files,
                                      selected_index)
from evaluation.utils.timescales import efolding_time, zero_crossing_time, eig_decay_time_years
from utils.params import artifact_root, esms, pdf_root, predictions_dir, requested_clim_vars

CLIM_VAR_LABELS = {'tas_ocean': 'OSAT', 'psl': 'SLP', 'tas': 'SAT'}

# (key, legend label, scatter style, vertical offset within a dataset row)
QUANTITIES = [
    ('efold', 'e-folding time of internal ACF', dict(marker='o'), 0.25),
    ('zero4', '4 × first zero crossing of internal ACF', dict(marker='s', facecolors='none'), 0.0),
    ('eigdecay', 'DMD eigenvalue decay time', dict(marker='x'), -0.25),
]


def chosen_method_for(clim_var):
    return 'PullbackDMDc-2d' if clim_var == 'psl' else 'PullbackDMDc-3d'


def acf_timescales(acf_data, top_n, dt_years):
    """Per dataset: e-folding and 4 x zero-crossing times, arrays of shape (n_members, top_n), in years."""
    out = {}
    for model_name, member_acfs in acf_data.items():
        efold = np.full((len(member_acfs), top_n), np.nan)
        zero4 = np.full((len(member_acfs), top_n), np.nan)
        for i, member_acf in enumerate(member_acfs):
            for k in range(min(top_n, member_acf.shape[0])):
                efold[i, k] = efolding_time(member_acf[k], dt_years)
                zero4[i, k] = 4.0 * zero_crossing_time(member_acf[k], dt_years)
        out[model_name] = {'efold': efold, 'zero4': zero4}
    return out


def eig_decay_times(model_dir, method, top_n):
    """DMD decay time (years) of the top_n rotated modes for every member, shape (n_members, top_n)."""
    files = sorted(f for f in os.listdir(model_dir) if f.split('_')[-1].split('.')[0] == method)
    decay = np.full((len(files), top_n), np.nan)
    for i, f_name in enumerate(files):
        with open(os.path.join(model_dir, f_name), 'rb') as f:
            dmd = pickle.load(f)
        dmd.compute_modes()
        with warnings.catch_warnings():
            # top_n may cut a conjugate pair; the slot then holds the pair's leading rotation.
            warnings.filterwarnings('ignore', message='top_n cuts off a complex-conjugate pair')
            eigvals = dmd.compute_rotated_modes(top_n=top_n)['eigvals']
        decay[i, :eigvals.size] = eig_decay_time_years(eigvals, dmd.lag)
    return decay


def plot_timescale_grid(timescales_by_var, clim_vars, top_n, timescale_bands, esm_colors,
                        model_display_names, obs_model):
    n_rows = len(clim_vars)
    fig, axes = plt.subplots(
        n_rows,
        top_n,
        figsize=(12, 3.2 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )

    model_labels = [m for m in esms if m in timescales_by_var[clim_vars[0]]]
    model_tick_labels = [model_display_names.get(m, m) for m in model_labels]
    positions = np.arange(1, len(model_labels) + 1)
    rng = np.random.default_rng(0)

    for row_idx, clim_var in enumerate(clim_vars):
        timescales_by_model = timescales_by_var[clim_var]

        for mode_idx in range(top_n):
            ax = axes[row_idx, mode_idx]

            for band_cfg in timescale_bands.values():
                ax.axvspan(band_cfg['xlim'][0], band_cfg['xlim'][1],
                           color=band_cfg['color'], alpha=band_cfg['alpha'], zorder=0)

            for pos, model_name in zip(positions, model_labels):
                color = esm_colors.get(model_name, 'tab:gray')
                alpha = 1.0 if model_name == obs_model else 0.6
                for key, _, style, offset in QUANTITIES:
                    values = timescales_by_model[model_name][key][:, mode_idx]
                    values = values[np.isfinite(values) & (values > 0.0)]
                    if values.size == 0:
                        continue
                    jitter = rng.uniform(-0.06, 0.06, size=values.size)
                    color_kw = {'edgecolors': color} if style.get('facecolors') == 'none' else {'color': color}
                    ax.scatter(values, np.full(values.size, pos + offset) + jitter,
                               s=14, alpha=alpha, linewidths=0.8, zorder=3, **style, **color_kw)
                    sel = timescales_by_model[model_name].get('selected', {}).get(key)
                    if sel is not None:
                        sel_value = timescales_by_model[model_name][key][sel, mode_idx]
                        if np.isfinite(sel_value) and sel_value > 0.0:
                            mark_selected(ax, sel_value, pos + offset, 0.1)

            if row_idx == 0:
                ax.set_title(f'Mode {mode_idx + 1}', fontsize=13)

            ax.set_yticks(positions)
            ax.set_yticklabels(model_tick_labels)
            ax.set_ylim(positions[0] - 0.5, positions[-1] + 0.5)
            ax.set_xscale('log')
            ax.grid(True, axis='x', which='major', alpha=0.3)
            ax.grid(True, axis='x', which='minor', alpha=0.15)
            for pos in positions[:-1]:
                ax.axhline(pos + 0.5, color='lightgray', linewidth=0.6, zorder=1)

            if row_idx == n_rows - 1:
                for band_name, band_cfg in timescale_bands.items():
                    ax.text(np.sqrt(band_cfg['xlim'][0] * band_cfg['xlim'][1]), 0.01, band_name,
                            ha='center', va='bottom', fontsize=9, color=band_cfg['color'],
                            transform=ax.get_xaxis_transform())

        axes[row_idx, 0].set_ylabel(CLIM_VAR_LABELS[clim_var], fontsize=12)

    for ax in axes[-1, :]:
        ax.set_xlabel('Years', fontsize=12)

    legend_handles = [
        Line2D([0], [0], linestyle='None', color='black', markerfacecolor='none' if style.get('facecolors') == 'none' else 'black',
               marker=style['marker'], markersize=6, label=label)
        for _, label, style, _ in QUANTITIES
    ] + [SELECTED_HANDLE]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle('Time scales of the leading dynamic modes', fontsize=18)
    return fig


def summary_rows(timescales_by_var, top_n):
    rows = []
    for clim_var, timescales_by_model in timescales_by_var.items():
        for model_name, quantities in timescales_by_model.items():
            for key, _, _, _ in QUANTITIES:
                for mode_idx in range(top_n):
                    values = quantities[key][:, mode_idx]
                    finite = values[np.isfinite(values)]
                    q25, median, q75 = (np.percentile(finite, [25, 50, 75]) if finite.size else (np.nan,) * 3)
                    rows.append({'var': clim_var, 'dataset': model_name, 'mode': mode_idx + 1, 'quantity': key,
                                 'n': values.size, 'n_nan': int(values.size - finite.size),
                                 'median': median, 'q25': q25, 'q75': q75})
    return rows


def main():
    # Main run only: lag 3, 1850-2014.
    year_suffix, lag_suffix = '', ''
    clim_vars = requested_clim_vars(['tas', 'psl', 'tas_ocean'])
    top_n = 4
    dt_years = 1.0 / 12.0

    artifact_results_dir = os.path.join(artifact_root, 'evaluation_results', f'results{year_suffix}{lag_suffix}')
    pdf_results_dir = os.path.join(pdf_root, f'results{year_suffix}{lag_suffix}')
    os.makedirs(pdf_results_dir, exist_ok=True)
    base_pred_dir = predictions_dir(year_suffix, lag_suffix)

    timescale_bands = {
        'ENSO': {'xlim': (2.0, 7.0), 'color': 'tab:cyan', 'alpha': 0.12},
        'PDO': {'xlim': (10.0, 30.0), 'color': 'tab:purple', 'alpha': 0.10},
    }
    esm_colors = {
        '20CRv3': 'black',
        'CESM2': '#7f7f7f',
        'MPI-ESM1-2-LR': '#8c564b',
        'CanESM5': '#2ca02c',
        'MIROC6': '#0b1f5b',
    }
    model_display_names = {'MPI-ESM1-2-LR': 'MPI-ESM'}
    obs_model = '20CRv3'

    timescales_by_var = {}
    for clim_var in clim_vars:
        method = chosen_method_for(clim_var)
        acf_path = os.path.join(artifact_results_dir, f'acfs_{clim_var}_{method}.pkl')
        if not os.path.exists(acf_path):
            print(f'Skipping {clim_var}: missing file {acf_path}')
            continue
        with open(acf_path, 'rb') as f:
            acf_data = pickle.load(f)

        timescales_by_model = acf_timescales(acf_data, top_n, dt_years)
        for model_name in esms:
            if model_name not in timescales_by_model:
                continue
            print(f'{clim_var} {model_name}: eigenvalues', flush=True)
            model_dir = os.path.join(base_pred_dir, clim_var, model_name)
            timescales_by_model[model_name]['eigdecay'] = eig_decay_times(model_dir, method, top_n)
            # Member shown in the mode figures: ACF rows follow os.listdir order, eig_decay_times sorted order.
            acf_idx = acf_selected_index(model_dir, method, model_name, acf_data[model_name],
                                         top_n=200 if clim_var == 'psl' else 20)
            timescales_by_model[model_name]['selected'] = {
                'efold': acf_idx, 'zero4': acf_idx,
                'eigdecay': selected_index(method_files(model_dir, method, sort=True), model_name)}
        timescales_by_var[clim_var] = timescales_by_model

    available_clim_vars = [cv for cv in clim_vars if cv in timescales_by_var]
    if not available_clim_vars:
        print('No requested climate variables had ACF files. Nothing to plot.')
        return

    summary = pd.DataFrame(summary_rows(timescales_by_var, top_n))
    out_csv = os.path.join(pdf_results_dir, 'acf_timescales_summary.csv')
    summary.to_csv(out_csv, index=False)
    print(f'Saved: {out_csv}')

    # How well the ACF e-folding time matches the eigenvalue decay time (1 = AR(1)-like mode).
    ratio = summary.pivot_table(index=['var', 'dataset', 'mode'], columns='quantity', values='median')
    ratio = (ratio['efold'] / ratio['eigdecay']).unstack('mode').round(2)
    print('\nMedian e-folding time / median eigenvalue decay time:')
    print(ratio.to_string())

    fig = plot_timescale_grid(timescales_by_var, available_clim_vars, top_n, timescale_bands,
                              esm_colors, model_display_names, obs_model)
    out_path = os.path.join(pdf_results_dir, f'acf_efolding_{"_".join(available_clim_vars)}.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()

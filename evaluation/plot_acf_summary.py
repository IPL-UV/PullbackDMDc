import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from utils.params import artifact_root, esms, method_markers, pdf_root

CLIM_VAR_LABELS = {'tas_ocean': 'OSAT', 'psl': 'SLP', 'tas': 'SAT'}
 

def build_suffixes(lag, start_year, end_year):
    if lag == 3:
        lag_suffix = ''
    elif lag == 12:
        lag_suffix = '_lag12'
    else:
        raise ValueError(f'Unsupported lag: {lag}')

    if (start_year, end_year) == (1950, 2014):
        year_suffix = '_tier1'
    elif (start_year, end_year) == (1850, 2014):
        year_suffix = ''
    else:
        raise ValueError(f'Unsupported year range: {(start_year, end_year)}')

    return lag_suffix, year_suffix


def first_zero_crossing_time(acf_values, dt_years):
    for idx in range(1, len(acf_values)):
        if acf_values[idx] <= 0.0:
            prev_idx = idx - 1
            prev_value = acf_values[prev_idx]
            next_value = acf_values[idx]
            if np.isclose(prev_value, next_value):
                return idx * dt_years
            weight = (0.0 - prev_value) / (next_value - prev_value)
            weight = np.clip(weight, 0.0, 1.0)
            return (prev_idx + weight) * dt_years
    return np.nan


def compute_member_mode_timescales(member_acfs, top_n, dt_years):
    member_timescales = []
    for member_acf in member_acfs:
        mode_timescales = []
        for mode_idx in range(top_n):
            crossing_years = first_zero_crossing_time(member_acf[mode_idx], dt_years)
            mode_timescales.append(4.0 * crossing_years if np.isfinite(crossing_years) else np.nan)
        member_timescales.append(mode_timescales)
    return np.asarray(member_timescales, dtype=float)


def load_timescale_summaries(clim_vars, results_dir, top_n, dt_years):
    summary_by_var = {}

    for clim_var in clim_vars:
        if clim_var  == 'psl':
            chosen_method = 'PullbackDMDc-2d'
        else:
            chosen_method = 'PullbackDMDc-3d'
        in_path = os.path.join(results_dir, f'acfs_{clim_var}_{chosen_method}.pkl')
        if not os.path.exists(in_path):
            print(f'Skipping {clim_var}: missing file {in_path}')
            continue

        with open(in_path, 'rb') as file_handle:
            acf_data = pickle.load(file_handle)

        summary_by_model = {}
        for model_name in esms:
            model_member_acfs = acf_data.get(model_name, [])
            summary_by_model[model_name] = compute_member_mode_timescales(
                member_acfs=model_member_acfs,
                top_n=top_n,
                dt_years=dt_years,
            )

        summary_by_var[clim_var] = summary_by_model

    return summary_by_var


def plot_mode_timescale_scatter_grid(
    summary_by_var,
    clim_vars,
    top_n,
    timescale_bands,
    esm_colors,
    model_display_names,
    obs_model,
    clim_var_to_marker,
):
    n_rows = len(clim_vars)
    fig, axes = plt.subplots(
        n_rows,
        top_n,
        figsize=(12, 2.7 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )

    ref_summary = summary_by_var[clim_vars[0]]
    model_labels = [model_name for model_name in esms if model_name in ref_summary]
    model_tick_labels = [model_display_names.get(model_name, model_name) for model_name in model_labels]
    positions = np.arange(1, len(model_labels) + 1)
    rng = np.random.default_rng(0)

    for row_idx, clim_var in enumerate(clim_vars):
        summary_by_model = summary_by_var[clim_var]

        for mode_idx in range(top_n):
            ax = axes[row_idx, mode_idx]

            for band_name, band_cfg in timescale_bands.items():
                ax.axvspan(
                    band_cfg['xlim'][0],
                    band_cfg['xlim'][1],
                    color=band_cfg['color'],
                    alpha=band_cfg['alpha'],
                    zorder=0,
                )

            for pos, model_name in zip(positions, model_labels):
                values = summary_by_model[model_name][:, mode_idx]
                values = values[np.isfinite(values) & (values > 0.0)]
                if values.size == 0:
                    continue

                jitter = rng.uniform(-0.08, 0.08, size=values.size)
                alpha = 1.0 if model_name == obs_model else 0.6

                ax.scatter(
                    values,
                    np.full(values.size, pos) + jitter,
                    color=esm_colors.get(model_name, 'tab:gray'),
                    s=24,
                    alpha=alpha,
                    marker=clim_var_to_marker[clim_var],
                    zorder=3,
                )

            if row_idx == 0:
                ax.set_title(f'Mode {mode_idx + 1}', fontsize=13)

            ax.set_yticks(positions)
            ax.set_yticklabels(model_tick_labels)
            ax.set_xscale('log')
            ax.grid(True, axis='x', which='major', alpha=0.3)
            ax.grid(True, axis='x', which='minor', alpha=0.15)

            if row_idx == n_rows - 1:
                for band_name, band_cfg in timescale_bands.items():
                    label_x = np.sqrt(band_cfg['xlim'][0] * band_cfg['xlim'][1])
                    ax.text(
                        label_x,
                        0.02,
                        band_name,
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        color=band_cfg['color'],
                        transform=ax.get_xaxis_transform(),
                    )

        axes[row_idx, 0].set_ylabel(CLIM_VAR_LABELS[clim_var], fontsize=12)

    for ax in axes[-1, :]:
        ax.set_xlabel('Years', fontsize=12)

    fig.suptitle('Dominant time scale', fontsize=18)
    return fig


def main():
    lag = 3
    start_year, end_year = 1850, 2014

    lag_suffix, year_suffix = build_suffixes(lag=lag, start_year=start_year, end_year=end_year)

    # Requested order: row 1 = tas (SAT), row 2 = psl (SLP)
    clim_vars = ['tas', 'psl']
    top_n = 4
    dt_years = 1.0 / 12.0

    artifact_results_dir = os.path.join(artifact_root, 'evaluation_results', f'results{year_suffix}{lag_suffix}')
    pdf_results_dir = os.path.join(pdf_root, f'results{year_suffix}{lag_suffix}')
    os.makedirs(artifact_results_dir, exist_ok=True)
    os.makedirs(pdf_results_dir, exist_ok=True)

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
    clim_var_to_method = {cv: ('PullbackDMDc-2d' if cv == 'psl' else 'PullbackDMDc-3d') for cv in clim_vars}
    clim_var_to_marker = {cv: method_markers.get(m, 'o') for cv, m in clim_var_to_method.items()}

    summary_by_var = load_timescale_summaries(
        clim_vars=clim_vars,
        results_dir=artifact_results_dir,
        top_n=top_n,
        dt_years=dt_years,
    )

    available_clim_vars = [clim_var for clim_var in clim_vars if clim_var in summary_by_var]
    if not available_clim_vars:
        print('No requested climate variables had valid ACF files. Nothing to plot.')
        return

    fig = plot_mode_timescale_scatter_grid(
        summary_by_var=summary_by_var,
        clim_vars=available_clim_vars,
        top_n=top_n,
        timescale_bands=timescale_bands,
        esm_colors=esm_colors,
        model_display_names=model_display_names,
        obs_model=obs_model,
        clim_var_to_marker=clim_var_to_marker,
    )

    clim_var_suffix = '_'.join(available_clim_vars)
    out_path = os.path.join(
        pdf_results_dir,
        f'acf_zero_crossing_{clim_var_suffix}_with_rotation.pdf',
    )
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from evaluation.utils.members import SELECTED_HANDLE, acf_selected_index, mark_selected
from evaluation.utils.style import CLIM_VARS_ON_FIG, ESM_COLORS, MODEL_DISPLAY_NAMES, TIMESCALE_BANDS, set_font_sizes
from evaluation.utils.timescales import zero_crossing_summary
from utils.params import artifact_root, esms, lag_suffix_for, method_markers, pdf_root, predictions_dir, year_suffix_for

# Prints at ~7/8/9/10 pt when included at \textwidth (figure saves ~11.9 in wide).
FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE = 13, 15, 17, 19
set_font_sizes(FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE)


def load_timescale_summaries(clim_vars, results_dir, top_n, dt_years, base_pred_dir):
    """Per var: timescales by model, and the row of the member shown in the mode figures."""
    summary_by_var = {}
    selected_by_var = {}

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
        selected_by_model = {}
        for model_name in esms:
            model_member_acfs = acf_data.get(model_name, [])
            summary_by_model[model_name] = np.asarray(zero_crossing_summary(
                member_acfs=model_member_acfs,
                top_n=top_n,
                dt_years=dt_years,
            ), dtype=float)
            if model_member_acfs:
                selected_by_model[model_name] = acf_selected_index(
                    os.path.join(base_pred_dir, clim_var, model_name), chosen_method, model_name,
                    model_member_acfs, top_n=200 if clim_var == 'psl' else 20)

        summary_by_var[clim_var] = summary_by_model
        selected_by_var[clim_var] = selected_by_model

    return summary_by_var, selected_by_var


def plot_mode_timescale_scatter_grid(
    summary_by_var,
    selected_by_var,
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

                sel = selected_by_var[clim_var].get(model_name)
                if sel is not None:
                    sel_value = summary_by_model[model_name][sel, mode_idx]
                    if np.isfinite(sel_value) and sel_value > 0.0:
                        mark_selected(ax, sel_value, pos, 0.15)

            if row_idx == 0:
                ax.set_title(f'Mode {mode_idx + 1}', fontsize=FS_TITLE)

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
                        fontsize=FS_TICK,
                        color=band_cfg['color'],
                        transform=ax.get_xaxis_transform(),
                    )

        axes[row_idx, 0].set_ylabel(CLIM_VARS_ON_FIG[clim_var], fontsize=FS_LABEL)

    for ax in axes[-1, :]:
        ax.set_xlabel('Years', fontsize=FS_LABEL)

    fig.legend(handles=[SELECTED_HANDLE], loc='upper center', bbox_to_anchor=(0.5, 0.0), frameon=False,
               fontsize=FS_TICK)
    return fig


def main():
    lag = 3
    start_year, end_year = 1850, 2014

    lag_suffix, year_suffix = lag_suffix_for(lag), year_suffix_for((start_year, end_year))

    # Requested order: row 1 = tas (SAT), row 2 = psl (SLP)
    clim_vars = ['tas', 'psl']
    top_n = 4
    dt_years = 1.0 / 12.0

    artifact_results_dir = os.path.join(artifact_root, 'evaluation_results', f'results{year_suffix}{lag_suffix}')
    pdf_results_dir = os.path.join(pdf_root, f'results{year_suffix}{lag_suffix}')
    os.makedirs(artifact_results_dir, exist_ok=True)
    os.makedirs(pdf_results_dir, exist_ok=True)

    obs_model = '20CRv3'
    clim_var_to_method = {cv: ('PullbackDMDc-2d' if cv == 'psl' else 'PullbackDMDc-3d') for cv in clim_vars}
    clim_var_to_marker = {cv: method_markers.get(m, 'o') for cv, m in clim_var_to_method.items()}

    summary_by_var, selected_by_var = load_timescale_summaries(
        clim_vars=clim_vars,
        results_dir=artifact_results_dir,
        top_n=top_n,
        dt_years=dt_years,
        base_pred_dir=predictions_dir(year_suffix, lag_suffix),
    )

    available_clim_vars = [clim_var for clim_var in clim_vars if clim_var in summary_by_var]
    if not available_clim_vars:
        print('No requested climate variables had valid ACF files. Nothing to plot.')
        return

    fig = plot_mode_timescale_scatter_grid(
        summary_by_var=summary_by_var,
        selected_by_var=selected_by_var,
        clim_vars=available_clim_vars,
        top_n=top_n,
        timescale_bands=TIMESCALE_BANDS,
        esm_colors=ESM_COLORS,
        model_display_names=MODEL_DISPLAY_NAMES,
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

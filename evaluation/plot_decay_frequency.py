import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from evaluation.utils.members import SELECTED_HANDLE, mark_selected, selected_index
from evaluation.utils.style import CLIM_VARS_ON_FIG, METHOD_NAMES_ON_FIG, display_model_name, set_font_sizes
from evaluation.utils.timescales import eig_decay_time_years, eig_frequency_per_year
from utils.params import esms, colors, lag_suffix_for, method_markers, pdf_root, predictions_dir, year_suffix_for

# Prints at ~7/8/9/10 pt when included at \textwidth (figure saves ~11.9 in wide).
FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE = 13, 15, 17, 19
set_font_sizes(FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE)


def remove_outliers_iqr(values, iqr_multiplier=1.5):
    """Filter outliers using an IQR rule; return values unchanged for tiny samples."""
    if values.size < 4:
        return values

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return values

    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    return values[(values >= lower) & (values <= upper)]


# def select_top_unique_eigs(eigs, top_n=4, tol=1e-10):
#     selected = []
#     for lam in eigs:
#         if len(selected) >= top_n:
#             break

#         is_conjugate_duplicate = False
#         if np.abs(np.imag(lam)) > tol:
#             for chosen in selected:
#                 if np.abs(lam - np.conjugate(chosen)) < 1e-8:
#                     is_conjugate_duplicate = True
#                     break

#         if not is_conjugate_duplicate:
#             selected.append(lam)

#     return np.array(selected)


def load_eig_data(clim_var, methods, top_n, base_pred_dir):
    """Return dict: model -> method -> list of (decay_time, frequency) arrays."""
    clim_dir = os.path.join(base_pred_dir, clim_var)
    data = {model: {method: {'decay': [], 'freq': [], 'files': []} for method in methods} for model in esms}

    for model in esms:
        model_dir = os.path.join(clim_dir, model)
        if not os.path.isdir(model_dir):
            continue

        for f_name in os.listdir(model_dir):
            method = f_name.split('_')[-1].split('.')[0]
            if method not in methods:
                continue

            f_path = os.path.join(model_dir, f_name)
            with open(f_path, 'rb') as f:
                dmd = pickle.load(f)

            eigs = dmd.eigvals if hasattr(dmd, 'eigvals') else np.linalg.eigvals(dmd.A)
            # eigs = select_top_unique_eigs(eigs, top_n=top_n)
            if eigs.size == 0:
                continue

            # Keep mode columns consistent across files by sorting from the
            # largest-magnitude eigenvalue downward before taking mode 1, 2, ...
            order = np.argsort(np.abs(eigs))[::-1]
            eigs = eigs[order]

            decay_times = np.real(eig_decay_time_years(eigs, dmd.lag))
            frequencies = np.real(eig_frequency_per_year(eigs, dmd.lag))

            data[model][method]['decay'].append(decay_times)
            data[model][method]['freq'].append(frequencies)
            data[model][method]['files'].append(f_name)

    return data


METRICS = [('decay', 'Decay time (yr)', 'log'), ('freq', 'Frequency (1/yr)', None)]


def draw_panel(ax, data, metric_key, mode_idx, methods, model_order, rng):
    """Scatter one metric of one mode: y = model (methods offset), x = metric value per member."""
    half = (len(methods) - 1) / 2.0
    method_offsets = {m: (i - half) * 0.18 for i, m in enumerate(methods)}
    positions = np.arange(len(model_order))

    for method in methods:
        offset = method_offsets[method]
        x_vals, y_vals = [], []

        for pos, model in zip(positions, model_order):
            member_arrays = data[model][method][metric_key]
            if not member_arrays:
                continue
            values = np.array([arr[mode_idx] for arr in member_arrays
                               if len(arr) > mode_idx and np.isfinite(arr[mode_idx])])
            # Use stronger filtering for frequency to suppress long tails.
            iqr_multiplier = 1.5 if metric_key == 'decay' else 1.0
            values = remove_outliers_iqr(values, iqr_multiplier=iqr_multiplier)
            if values.size == 0:
                continue
            jitter = rng.uniform(-0.06, 0.06, size=values.size)
            x_vals.append(values)
            y_vals.append(np.full(values.size, pos + offset) + jitter)

            # Member shown in the mode figures, if it survived the filters above.
            sel = selected_index(data[model][method]['files'], model)
            if sel is not None and len(member_arrays[sel]) > mode_idx:
                sel_value = member_arrays[sel][mode_idx]
                if np.isfinite(sel_value) and np.isin(sel_value, values):
                    mark_selected(ax, sel_value, pos + offset, 0.07)

        if x_vals:
            ax.scatter(
                np.concatenate(x_vals),
                np.concatenate(y_vals),
                marker=method_markers.get(method, 'o'),
                color=colors.get(method, 'tab:gray'),
                s=3,
                alpha=0.75,
                zorder=3,
                label=method,
            )

    ax.set_yticks(positions)
    ax.grid(True, axis='x', which='major', alpha=0.3)
    ax.grid(True, axis='x', which='minor', alpha=0.15)


def plot_all_vars(data_by_var, clim_vars, methods, model_order, top_n):
    """Rows = modes; one column pair (decay, frequency) per variable, variables left to right."""
    fig = plt.figure(figsize=(12.0, 2.1 * top_n), layout='constrained')
    # First block carries the model tick labels, so give it extra width to keep panels equal.
    subfigs = fig.subfigures(1, len(clim_vars), width_ratios=[1.3] + [1.0] * (len(clim_vars) - 1),
                             wspace=0.02, squeeze=False)[0]
    tick_labels = [display_model_name(m) for m in model_order]

    first_axes = None
    for v_idx, (clim_var, subfig) in enumerate(zip(clim_vars, subfigs)):
        axes = subfig.subplots(top_n, len(METRICS), sharex='col', sharey=True, squeeze=False)
        rng = np.random.default_rng(0)
        for col, (metric_key, x_label, xscale) in enumerate(METRICS):
            for mode_idx in range(top_n):
                ax = axes[mode_idx, col]
                draw_panel(ax, data_by_var[clim_var], metric_key, mode_idx, methods, model_order, rng)
                if xscale == 'log':
                    ax.set_xscale('log')
                if v_idx == 0 and col == 0:
                    ax.set_yticklabels(tick_labels, fontsize=FS_TICK)
                    ax.set_ylabel(f'Mode {mode_idx + 1}', fontsize=FS_LABEL)
                else:
                    ax.tick_params(axis='y', labelleft=False)
            axes[-1, col].set_xlabel(x_label, fontsize=FS_LABEL)
        subfig.suptitle(CLIM_VARS_ON_FIG.get(clim_var, clim_var), fontsize=FS_TITLE)
        if first_axes is None:
            first_axes = axes

    handles, labels = first_axes[0, 0].get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    fig.legend(
        [h for h, _ in unique] + [SELECTED_HANDLE],
        [METHOD_NAMES_ON_FIG.get(l, l) for _, l in unique] + [SELECTED_HANDLE.get_label()],
        title='Method',
        loc='upper center',
        ncol=len(unique) + 1,
        frameon=True,
        facecolor=None,
        edgecolor='#9e9e9e',
        framealpha=1.0,
        bbox_to_anchor=(0.5, 0.0),
        markerscale=3,
        fontsize=FS_TICK,
        title_fontsize=FS_TICK,
    )

    return fig


if __name__ == '__main__':
    lag = 3
    start_year, end_year = 1850, 2014

    lag_suffix, year_suffix = lag_suffix_for(lag), year_suffix_for((start_year, end_year))

    clim_vars = ['psl', 'tas', 'tas_ocean']  # left to right in the figure
    methods = ['LIM', 'PullbackDMDc', 'PullbackDMDc-2d', 'PullbackDMDc-3d']
    top_n = 4

    base_pred_dir = predictions_dir(year_suffix, lag_suffix)
    results_dir = os.path.join(pdf_root, f'results{year_suffix}{lag_suffix}')
    os.makedirs(results_dir, exist_ok=True)

    data_by_var = {cv: load_eig_data(cv, methods, top_n, base_pred_dir) for cv in clim_vars}

    model_order = [m for m in esms if any(
        data_by_var[clim_vars[0]][m][method]['decay'] for method in methods
    )]
    if not model_order:
        raise ValueError('No eigenvalue data found for the requested configuration.')

    fig = plot_all_vars(data_by_var, clim_vars, methods, model_order, top_n)
    out_fig = os.path.join(results_dir, f'decay_frequency_top{top_n}_all.pdf')
    fig.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_fig}', flush=True)

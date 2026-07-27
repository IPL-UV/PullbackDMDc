import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

CLIM_VAR_LABELS = {'tas_ocean': 'OSAT', 'psl': 'SLP', 'tas': 'SAT'}

from utils.params import esms, colors, method_markers, predictions_dir

method_names_on_fig = {
    'RegGMST': 'RegGMST',
    'LR': 'Linear reg.',
    'LIM': 'LIM',
    'LIM-opt': 'LIM-opt',
    'PullbackDMDc': 'PullbackDMDc',
    'PullbackDMDc-2d': 'PullbackDMDc(2d)',
    'PullbackDMDc-3d': 'PullbackDMDc(3d)',
}

model_display_names = {'MPI-ESM1-2-LR': 'MPI-ESM'}


def display_model_name(model_name):
    return model_display_names.get(model_name, model_name)


def compute_yearly_decay_time(lambdas, tau):
    return -tau / (12 * np.log(np.abs(lambdas)))


def compute_yearly_frequency(lambdas, tau):
    omegas = np.angle(lambdas) / tau
    return (6 * np.abs(omegas)) / np.pi


def parse_ensemble_member_from_filename(f_name, esm_name):
    if esm_name == '20CRv3':
        return ''
    return f_name.split('_')[-4]


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
    data = {model: {method: {'decay': [], 'freq': []} for method in methods} for model in esms}

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

            decay_times = np.real(compute_yearly_decay_time(eigs, dmd.lag))
            frequencies = np.real(compute_yearly_frequency(eigs, dmd.lag))

            data[model][method]['decay'].append(decay_times)
            data[model][method]['freq'].append(frequencies)

    return data


def plot_combined_transposed(data, methods, model_order, top_n, title):
    """Row 1=decay time, row 2=frequency; cols=modes; y-axis=models, x-axis=metric value."""
    metrics = [
        ('decay', 'Decay Time (years)', 'log'),
        ('freq',  'Frequency (1/years)', None),
    ]
    n_rows = len(metrics)
    n_cols = top_n
    rng = np.random.default_rng(0)

    half = (len(methods) - 1) / 2.0
    method_offsets = {m: (i - half) * 0.18 for i, m in enumerate(methods)}

    positions = np.arange(len(model_order))
    tick_labels = [display_model_name(m) for m in model_order]

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 2.8 * n_rows),
        sharex='row',
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14)

    for row_idx, (metric_key, row_label, xscale) in enumerate(metrics):
        for mode_idx in range(top_n):
            ax = axes[row_idx, mode_idx]

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

            if xscale == 'log':
                ax.set_xscale('log')

            ax.set_yticks(positions)
            ax.set_yticklabels(tick_labels, fontsize=9)
            ax.grid(True, axis='x', which='major', alpha=0.3)
            ax.grid(True, axis='x', which='minor', alpha=0.15)

            if row_idx == 0:
                ax.set_title(f'Mode {mode_idx + 1}', fontsize=12)

            if mode_idx == 0:
                ax.set_ylabel(row_label, fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    fig.legend(
        [h for h, _ in unique],
        [method_names_on_fig.get(l, l) for _, l in unique],
        title='Method',
        loc='lower center',
        ncol=len(unique),
        frameon=True,
        facecolor=None,
        edgecolor='#9e9e9e',
        framealpha=1.0,
        bbox_to_anchor=(0.5, -0.12),
        markerscale=3,
        fontsize=10,
        title_fontsize=11,
    )

    return fig


if __name__ == '__main__':
    lag = 3
    start_year, end_year = 1850, 2014

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

    clim_vars = ['tas', 'psl', 'tas_ocean']
    methods = ['LIM', 'PullbackDMDc', 'PullbackDMDc-2d', 'PullbackDMDc-3d']
    top_n = 4

    base_pred_dir = predictions_dir(year_suffix, lag_suffix)
    results_dir = os.path.join('.', f'evaluation_results/results{year_suffix}{lag_suffix}')
    os.makedirs(results_dir, exist_ok=True)

    data_by_var = {cv: load_eig_data(cv, methods, top_n, base_pred_dir) for cv in clim_vars}

    model_order = [m for m in esms if any(
        data_by_var[clim_vars[0]][m][method]['decay'] for method in methods
    )]
    if not model_order:
        raise ValueError('No eigenvalue data found for the requested configuration.')

    for clim_var in clim_vars:
        clim_var = [clim_var]
        clim_var_suffix = '_'.join(clim_var)
        title = f'Decay Time & Frequency – {" ".join(CLIM_VAR_LABELS.get(cv, cv) for cv in clim_var)}'
        fig = plot_combined_transposed(
            data=data_by_var[clim_var[0]],
            methods=methods,
            model_order=model_order,
            top_n=top_n,
            title=title,
        )
        out_fig = os.path.join(results_dir, f'decay_frequency_top{top_n}_{clim_var_suffix}.pdf')
        fig.savefig(out_fig, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_fig}', flush=True)
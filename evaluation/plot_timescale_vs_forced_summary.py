import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from utils.params import esms, method_markers, predictions_dir


def first_zero_crossing_time(acf_values, dt):
    for idx in range(1, len(acf_values)):
        if acf_values[idx] <= 0.0:
            prev_idx = idx - 1
            prev_value = acf_values[prev_idx]
            next_value = acf_values[idx]
            if np.isclose(prev_value, next_value):
                return idx * dt
            weight = (0.0 - prev_value) / (next_value - prev_value)
            weight = np.clip(weight, 0.0, 1.0)
            return (prev_idx + weight) * dt
    return np.nan


def compute_zero_crossing_summary(member_acfs, top_n, dt_years):
    summaries = []
    for member_acf in member_acfs:
        mode_vals = []
        for mode_idx in range(top_n):
            crossing = first_zero_crossing_time(member_acf[mode_idx], dt_years)
            mode_vals.append(4.0 * crossing if np.isfinite(crossing) else np.nan)
        summaries.append(np.asarray(mode_vals, dtype=float))
    return summaries


def build_suffixes(lag, start_year, end_year):
    if lag == 3:
        lag_suffix = ''
    elif lag == 12:
        lag_suffix = '_lag12'
    elif lag == 6:
        lag_suffix = '_lag6'
    elif lag == 1:
        lag_suffix = '_lag1'
    else:
        raise ValueError(f'Unsupported lag={lag}')

    if (start_year, end_year) == (1850, 2014):
        year_suffix = ''
    elif (start_year, end_year) == (1950, 2014):
        year_suffix = '_tier1'
    else:
        raise ValueError(f'Unsupported year range: {(start_year, end_year)}')

    return lag_suffix, year_suffix


def gather_forced_fraction(base_pred_dir, clim_var, chosen_method, top_n):
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

            forced_fraction = (
                np.var(rotated['forced_time_series'], axis=0)
                / np.var(rotated['internal_time_series'] + rotated['forced_time_series'], axis=0)
            )
            forced_fraction_data[chosen_model].append(forced_fraction)

    return forced_fraction_data


def plot_forced_fraction_vs_timescale(
    row1_timescales,
    row3_forced_fraction,
    clim_var,
    chosen_method,
    results_dir,
    smooth_sigma,
):
    top_n = 4

    timescale_bands = {
        'ENSO': {'xlim': (2.0, 7.0),   'color': 'tab:cyan',   'alpha': 0.12},
        'PDO':  {'xlim': (10.0, 30.0), 'color': 'tab:purple', 'alpha': 0.10},
    }
    esm_colors = {
        '20CRv3': 'black', 'CESM2': '#7f7f7f',
        'MPI-ESM1-2-LR': '#8c564b', 'CanESM5': '#2ca02c', 'MIROC6': '#0b1f5b',
    }
    model_display_names = {'MPI-ESM1-2-LR': 'MPI-ESM'}
    obs_model = '20CRv3'
    method_marker = method_markers.get('PullbackDMDc-3d', 'o')

    fig, axes = plt.subplots(1, top_n, figsize=(12, 3.2), constrained_layout=True, sharex='row', sharey='row')

    legend_handles = []
    for mode_idx in range(top_n):
        ax = axes[mode_idx]
        for band_cfg in timescale_bands.values():
            ax.axhspan(*band_cfg['xlim'], color=band_cfg['color'], alpha=band_cfg['alpha'], zorder=0)

        for model in esms:
            ts_members = row1_timescales.get(model, [])
            ff_members = row3_forced_fraction.get(model, [])
            pairs = [(float(ts[mode_idx]), float(ff[mode_idx]))
                     for ts, ff in zip(ts_members, ff_members)
                     if ts.size > mode_idx and ff.size > mode_idx]
            pairs = [(x, y) for x, y in pairs if np.isfinite(x) and x > 0 and np.isfinite(y)]
            if not pairs:
                continue
            xs, ys = zip(*pairs)
            is_obs = model == obs_model
            sc = ax.scatter(ys, xs,  # forced fraction on x, timescale on y
                            color=esm_colors.get(model, 'tab:gray'), marker=method_marker,
                            s=108 if is_obs else 36,
                            alpha=1.0 if is_obs else 0.65,
                            linewidths=0.9 if is_obs else 0.3, edgecolor='none', zorder=3,
                            label=model_display_names.get(model, model))
            if mode_idx == 0:
                legend_handles.append(sc)

        ax.set_xlim(0.0, 1.0)
        ax.grid(True, which='both', alpha=0.25)
        ax.set_title(f'Mode {mode_idx + 1}', fontsize=13)
        ax.set_xlabel('Forced Fraction', fontsize=10)
        if mode_idx == 0:
            ax.set_ylabel('Time Scale (years)', fontsize=12)

    axes[0].legend(handles=legend_handles, fontsize=8, loc='upper left',
                   framealpha=0.7, title='Model', title_fontsize=8)

    fig.suptitle('OSAT: Forced Fraction vs. Time Scale', fontsize=16)

    out_path = os.path.join(
        results_dir,
        f'forced_fraction_vs_timescale_{clim_var}_{chosen_method}_sigma{smooth_sigma}.pdf',
    )
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    lag = 3
    start_year, end_year = 1850, 2014

    lag_suffix, year_suffix = build_suffixes(lag=lag, start_year=start_year, end_year=end_year)

    clim_vars = ['tas_ocean']
    chosen_method = 'PullbackDMDc-3d'
    top_n = 4
    dt_years = 1.0 / 12.0
    smooth_sigma = 12 * 8  # kept for output filename consistency

    base_pred_dir = predictions_dir(year_suffix, lag_suffix)
    results_dir = os.path.join('.', f'evaluation_results/results{year_suffix}{lag_suffix}')
    os.makedirs(results_dir, exist_ok=True)

    for clim_var in clim_vars:
        acf_path = os.path.join(results_dir, f'acfs_{clim_var}_{chosen_method}.pkl')
        if not os.path.exists(acf_path):
            print(f'Skipping {clim_var}: missing ACF file {acf_path}')
            continue

        with open(acf_path, 'rb') as fh:
            acf_data = pickle.load(fh)

        row1_timescales = {}
        for chosen_model in esms:
            member_acfs = acf_data.get(chosen_model, [])
            row1_timescales[chosen_model] = compute_zero_crossing_summary(
                member_acfs=member_acfs,
                top_n=top_n,
                dt_years=dt_years,
            )

        row3_forced_fraction = gather_forced_fraction(
            base_pred_dir=base_pred_dir,
            clim_var=clim_var,
            chosen_method=chosen_method,
            top_n=top_n,
        )

        out_path = plot_forced_fraction_vs_timescale(
            row1_timescales=row1_timescales,
            row3_forced_fraction=row3_forced_fraction,
            clim_var=clim_var,
            chosen_method=chosen_method,
            results_dir=results_dir,
            smooth_sigma=smooth_sigma,
        )
        print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
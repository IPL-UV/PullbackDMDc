import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import dpss

from utils.params import esms, predictions_dir


def parse_ensemble_member_from_filename(f_name, esm_name):
    if esm_name == '20CRv3':
        return ''
    return f_name.split('_')[-4]

# ---------------------------------------------------------------
# Multitaper PSD
# ---------------------------------------------------------------
def multitaper_psd(x, fs, NW=3, detrend='constant'):
    """
    Multitaper power spectral density estimate using Slepian (DPSS) tapers.

    Parameters
    ----------
    x       : 1-D array, the time series (monthly, length T)
    fs      : sampling frequency in cycles per year (= 12 for monthly data)
    NW      : time-bandwidth product.
                NW=2 -> finest resolution (2*NW/T cy/yr) but only K=3 tapers,
                        noisier; can hint at 40-50 yr variability.
                NW=3 -> good default for climate: K=5 tapers, resolves PDO band.
                NW=4 -> smoother, similar resolution to 20-yr Welch segments.
    detrend : 'constant' removes the mean; 'linear' removes a linear trend.

    Returns
    -------
    freqs : 1-D array, frequency axis in cycles per year (0 … fs/2)
    psd   : 1-D array, PSD in [PC_units² / (cy yr⁻¹)], normalised so that
            np.trapz(psd, freqs) ≈ np.var(x)  (Parseval satisfied).

    Notes
    -----
    Resolution bandwidth = 2*NW / (N * dt) cycles/year.
    For a 165-year monthly record:
        NW=2 -> 0.024 cy/yr  (period ~42 yr resolvable)
        NW=3 -> 0.036 cy/yr  (period ~28 yr resolvable)
        NW=4 -> 0.048 cy/yr  (period ~21 yr resolvable)
    All are better than Welch with 20-yr segments (0.050 cy/yr).
    """
    N = len(x)
    K = int(2 * NW - 1)   # number of tapers: 3, 5, or 7 for NW=2,3,4

    if detrend == 'constant':
        x = x - np.mean(x)
    elif detrend == 'linear':
        x = x - np.polyval(np.polyfit(np.arange(N), x, 1), np.arange(N))

    tapers, ratios = dpss(N, NW, Kmax=K, return_ratios=True)
    # ratios are spectral concentration eigenvalues; all ≈1 for the first K
    # tapers when NW >= 2.5, which is what makes MTM variance reduction reliable

    tapered      = tapers * x[np.newaxis, :]            # (K, N)
    eigenspectra = np.abs(np.fft.rfft(tapered, axis=1)) ** 2 / fs
    # simple average over tapers — adaptive weighting gives nearly identical
    # results when all concentration eigenvalues are close to 1
    psd   = np.mean(eigenspectra, axis=0)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)             # cy/yr

    # Drop the zero-frequency (DC) bin so period-axis transforms 1/f stay finite.
    return freqs[1:], psd[1:]


# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
lag = 3
start_year, end_year = 1850, 2014
lag_suffix  = '' if lag == 3 else '_lag12'
year_suffix = '' if (start_year, end_year) == (1850, 2014) else '_tier1'

clim_vars        = ['tas_ocean']
clim_vars_on_fig = {'tas_ocean': 'OSAT', 'psl': 'SLP', 'tas': 'SAT'}
chosen_method    = 'PullbackDMDc-3d'

base_pred_dir = predictions_dir(year_suffix, lag_suffix)
results_dir   = os.path.join('.', f'evaluation_results/results{year_suffix}{lag_suffix}')

top_n = 4
fs    = 12.0        # monthly data -> 12 samples / year
NW    = 3           # time-bandwidth product; change to 2 for finer resolution

esm_colors = {
    '20CRv3':        'black',
    'CESM2':         '#7f7f7f',
    'MPI-ESM1-2-LR': '#8c564b',
    'CanESM5':       '#2ca02c',
    'MIROC6':        '#0b1f5b',
}
model_display_names = {'MPI-ESM1-2-LR': 'MPI-ESM'}

# Match the member selection used in plot_modes.py
esm_member_selection = {
    '20CRv3': '',
    'MPI-ESM1-2-LR': 'r47i1p1f1',
    'CESM2': '1281.005i1p1f1',
    'CanESM5': 'r16i1p1f1',
    'MIROC6': 'r35i1p1f1',
}

# Use distinct line styles so ESM curves are distinguishable even in grayscale.
esm_linestyles = {
    '20CRv3': '-',
    'CESM2': '--',
    'MPI-ESM1-2-LR': '-.',
    'CanESM5': ':',
    'MIROC6': (0, (3, 1, 1, 1)),
}

# Frequency-band shading (expressed as cy/yr)
freq_bands = {
    'ENSO': {'flim': (1. / 7.,  1. / 2.),  'color': 'tab:cyan',   'alpha': 0.15},
    'PDO':  {'flim': (1. / 30., 1. / 10.), 'color': 'tab:purple',  'alpha': 0.12},
}

os.makedirs(results_dir, exist_ok=True)

# ---------------------------------------------------------------
# Main loop over climate variables
# ---------------------------------------------------------------
for clim_var in clim_vars:

    # psd_data[model] = list over members of (top_n, n_freqs) arrays
    psd_data  = {}
    freq_axis = None   # shared across all models/members once set

    for chosen_model in esms:
        model_dir = os.path.join(base_pred_dir, clim_var, chosen_model)
        if not os.path.isdir(model_dir):
            print(f'Skipping missing model directory: {model_dir}')
            psd_data[chosen_model] = []
            continue

        dmd_files = sorted([
            os.path.join(model_dir, fn)
            for fn in os.listdir(model_dir)
            if fn.split('_')[-1].split('.')[0] == chosen_method
        ])

        selected_member = esm_member_selection.get(chosen_model, '')
        dmd_files = [
            f_path
            for f_path in dmd_files
            if parse_ensemble_member_from_filename(os.path.basename(f_path), chosen_model) == selected_member
        ]

        if not dmd_files:
            print(
                f'No {chosen_method} file for {clim_var} / {chosen_model} '
                f'member={selected_member!r}'
            )
            psd_data[chosen_model] = []
            continue

        member_psds = []

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

            mode_psds = [None] * top_n
            for k in range(plot_n):
                freqs, psd = multitaper_psd(internal_ts[:, k], fs=fs, NW=NW)
                if labels[k][1] == 'complex':
                    mode_psds[k] = psd
                else:
                    spatial_norm_sq = float(np.sum(modes[:, k] ** 2))
                    # multiply by ||w||^2 so that ∫ PSD df = ||w||^2 * var(z)
                    # = variance of the reconstructed field in PC space
                    mode_psds[k] = spatial_norm_sq * psd

            if freq_axis is None:
                freq_axis = freqs   # cy/yr, same for every file

            member_psds.append(np.stack(mode_psds))   # (top_n, n_freqs)

        psd_data[chosen_model] = member_psds

    # ---------------------------------------------------------------
    # Plot: one panel per mode
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(
        1, top_n,
        figsize=(4.5 * top_n, 4.5),
        constrained_layout=True,
        sharey=True,
    )

    for mode_idx in range(top_n):
        ax = axes[mode_idx]

        # frequency-band shading
        for band_name, band_cfg in freq_bands.items():
            ax.axvspan(
                band_cfg['flim'][0], band_cfg['flim'][1],
                color=band_cfg['color'], alpha=band_cfg['alpha'], zorder=0,
            )
            label_f = np.sqrt(band_cfg['flim'][0] * band_cfg['flim'][1])
            ax.text(
                label_f, 0.97, band_name,
                ha='center', va='top', fontsize=12,
                color=band_cfg['color'],
                transform=ax.get_xaxis_transform(),
            )

        for chosen_model in esms:
            color        = esm_colors.get(chosen_model, 'tab:gray')
            linestyle    = esm_linestyles.get(chosen_model, '-')
            display_name = model_display_names.get(chosen_model, chosen_model)
            if not psd_data.get(chosen_model):
                continue

            # One line per selected ensemble member (matching plot_modes selection).
            psd = psd_data[chosen_model][0][mode_idx]
            ax.plot(
                freq_axis, psd,
                color=color, linestyle=linestyle, lw=2.0, alpha=0.95, zorder=3,
                label=display_name,
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'Mode {mode_idx + 1}', fontsize=17)
        ax.set_xlabel('Frequency (cycles / year)', fontsize=15)
        ax.grid(True, which='both', alpha=0.2)
        ax.tick_params(labelsize=14)

        # secondary x-axis: period in years
        ax2 = ax.secondary_xaxis(
            'top',
            functions=(lambda f: 1.0 / f, lambda p: 1.0 / p),
        )
        ax2.set_xlabel('Period (years)', fontsize=14)
        ax2.tick_params(labelsize=13)

        if mode_idx == 0:
            ax.set_ylabel(r'PSD  [PC units² / (cy yr$^{-1}$)]', fontsize=15)

    # shared legend placed below the figure
    legend_handles = [
        plt.Line2D([], [], color=esm_colors.get(m, 'tab:gray'),
                   linestyle=esm_linestyles.get(m, '-'),
                   lw=2, label=model_display_names.get(m, m))
        for m in esms
        if psd_data.get(m)
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(esms),
        fontsize=14,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.14),
    )

    # resolution annotation in figure title
    T_years  = (end_year - start_year + 1)
    res_cyyr = 2 * NW / T_years
    res_yr   = 1.0 / res_cyyr

    fig.suptitle(
        (f'{clim_vars_on_fig[clim_var]}: Power spectral density  -  '
         f'Multitaper MTM, NW={NW}, K={2*NW-1} tapers  -  '
         f'resolution {res_cyyr:.3f} cy/yr  ({res_yr:.0f} yr period)'),
        fontsize=17,
    )

    out_path = os.path.join(results_dir, f'psd_mtm_{clim_var}_{chosen_method}.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.utils.style import METHOD_NAMES_ON_FIG, set_font_sizes
from param_sensitivity.compute_sweep import METHOD, SENS_ROOT
from utils.params import n_eofs, pdf_root, transition_time

FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE = 8, 9, 10, 11
set_font_sizes(FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE)

SWEEPS = ['N', 'lag', 'history']
SWEEP_TITLE = {'N': 'Truncation', 'lag': 'Lag', 'history': 'Forcing history'}
# (var, label, colour, marker): Okabe-Ito colours, one marker per variable for grayscale.
SERIES = [('tas', 'SAT', '#0072B2', 'o'),
          ('tas_ocean', 'OSAT', '#D55E00', 's'),
          ('psl', 'SLP', '#009E73', '^')]


def load():
    files = glob.glob(os.path.join(SENS_ROOT, 'results', 'main', '*.csv'))
    df = pd.concat([pd.read_csv(f) for f in sorted(files)], ignore_index=True)
    df = df[df.method == df['var'].map(METHOD)].copy()
    df['hard_fail'] = (df.lead_abs >= 1) | df.nonfinite.fillna(False).astype(bool) | df.fit_error.notna()
    df['rmsd'] = df.td_rmse * 100  # Taylor distance to (1, 0), % of the truth std
    return df


def is_main(df):
    return (df.N == df['var'].map(n_eofs)) & (df.lag == 3) & (df.history == transition_time)


def sweeps(df):
    # Rows of each one-at-a-time sweep keyed by x; the main config belongs to every sweep.
    main_n, main_h = df['var'].map(n_eofs), df.history == transition_time
    return {'N': df[(df.lag == 3) & main_h].assign(x=lambda d: d.N / d['var'].map(n_eofs)),
            'lag': df[(df.N == main_n) & main_h].assign(x=lambda d: d.lag),
            'history': df[(df.N == main_n) & (df.lag == 3)].assign(x=lambda d: d.history / 12)}


def fixed_set(d):
    # Drop members with a hard failure at any setting of the sweep, so every point uses the same members.
    bad = set(d.loc[d.hard_fail, ['var', 'esm', 'member']].itertuples(index=False, name=None))
    return d[[key not in bad for key in zip(d['var'], d.esm, d.member)]]


def weighted_quantile(x, w, qs):
    o = np.argsort(x)
    x, w = x[o], w[o]
    return np.interp(qs, (np.cumsum(w) - 0.5 * w) / w.sum(), x)


def deltas(sw):
    # Per-member RMSD minus the same member's RMSD at the main config; every ESM weighted equally.
    rows = []
    for sweep, d in sw.items():
        for var, d in fixed_set(d).groupby('var'):
            main = d[is_main(d)].set_index(['esm', 'member']).rmsd
            d = d.assign(d=d.rmsd.values - main.reindex(pd.MultiIndex.from_frame(d[['esm', 'member']])).values)
            for x, g in d.groupby('x'):
                w = 1. / g.groupby('esm').member.transform('count').values
                q = weighted_quantile(g.d.values, w, [0.25, 0.5, 0.75])
                rows.append(dict(var=var, sweep=sweep, x=x, n_members=len(g), q25=q[0], q50=q[1], q75=q[2]))
    return pd.DataFrame(rows)


def style_ax(ax, sweep):
    if sweep == 'history':
        ax.set_xscale('symlog', linthresh=1, linscale=0.5)  # 0 yr = cold start in 1850
        ticks = [0, 1, 2, 5, 10, 25, 50, 99]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlim(-0.2, 130)
        ax.set_xlabel('forcing history before 1850 (years)')
        main = transition_time / 12
    else:
        if sweep == 'N':
            ticks, labels, main = [0.1, 0.25, 0.5, 1, 2, 4], ['×0.1', '×0.25', '×0.5', '×1', '×2', '×4'], 1
            ax.set_xlabel('N / main N (SAT, OSAT: 20; SLP: 200)')
        else:
            ticks, main = [1, 3, 6, 12], 3
            labels = [str(t) for t in ticks]
            ax.set_xlabel('lag (months)')
        ax.set_xscale('log', base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    ax.minorticks_off()
    ax.axvline(main, color='0.6', lw=0.8, ls='--', zorder=0)
    ax.axhline(0, color='0.3', lw=0.8, ls=':', zorder=0)
    ax.grid(axis='y', color='0.9', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def plot(dq, out_path):
    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(10, 3.6), sharey=True)
    for k, (ax, sweep) in enumerate(zip(axes, SWEEPS)):
        for var, label, color, marker in SERIES:
            q = dq[(dq['var'] == var) & (dq.sweep == sweep)].sort_values('x')
            if q.empty:
                continue
            label = f'{label}, {METHOD_NAMES_ON_FIG[METHOD[var]]}'
            ax.fill_between(q.x, q.q25, q.q75, color=color, alpha=0.15, lw=0)
            ax.plot(q.x, q.q50, color=color, lw=2, marker=marker, ms=4, label=label)
        style_ax(ax, sweep)
        ax.set_title(f'({"abc"[k]}) {SWEEP_TITLE[sweep]}', loc='left')
    axes[0].set_ylabel('ΔRMSD (percentage points\nof ground-truth std)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(l), fontsize=FS_LABEL, frameon=False, handlelength=3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    out_dir = os.path.join(pdf_root, 'param_sensitivity')
    os.makedirs(out_dir, exist_ok=True)
    sw = sweeps(load())
    kept = pd.concat({k: fixed_set(v).groupby(['var', 'esm']).member.nunique() for k, v in sw.items()}, names=['sweep'])
    total = pd.concat({k: v.groupby(['var', 'esm']).member.nunique() for k, v in sw.items()}, names=['sweep'])
    print('Members kept / total:\n', (kept.astype(str) + '/' + total.astype(str)).unstack('esm').to_string())
    dq = deltas(sw)
    dq.round(4).to_csv(os.path.join(out_dir, 'pullback_drmsd_quantiles.csv'), index=False)
    plot(dq, os.path.join(out_dir, 'pullback_rmsd.pdf'))
    print(f'Saved pullback_rmsd.pdf to {out_dir}')

"""R2-M5 sensitivity figures: GM / polar-cap RMSD against truncation N, lag and tas_ocean domain.

Reads SENS_ROOT/results/{main,domain}/*.csv written by sweep.py and writes to pdf_root/param_sensitivity/
  pullback_rmsd.{pdf,png}   headline PullbackDMDc (-3d SAT/OSAT, -2d SLP) only, all variables on one N axis
  methods_rmsd.{pdf,png}    baselines and PullbackDMDc, one row per variable / index
  rmsd_quantiles.csv        member quartiles behind both figures
Lines are the member median over the four ESMs, bands the IQR. Members with a hard failure (unstable, nonfinite,
fit_error) for any method at any setting of a sweep are dropped from that sweep, so every line uses the same members.
The domain column is drawn if results/domain/*.csv exist (see README.md).
  python -m param_sensitivity.plot_rmsd
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.params import artifact_root, colors, n_eofs, pdf_root

SENS_ROOT = os.getenv('PBDMDC_SENS_ROOT', os.path.join(artifact_root, 'param_sensitivity'))  # as in sweep.py
RES = os.path.join(SENS_ROOT, 'results')
FIG = os.path.join(pdf_root, 'param_sensitivity')
SWEEPS = ['N', 'lag', 'domain']
SWEEP_TITLE = {'N': 'Truncation', 'lag': 'Lag', 'domain': 'Domain (OSAT only)'}
DOMAINS = ['main', '65S65N', '90S90N']
DOMAIN_LABEL = {'main': '40S–65N\n(main)', '65S65N': '65S–65N', '90S90N': '90S–90N'}
METHOD_ORDER = ['RegGMST', 'LR', 'LIM', 'LIM-opt', 'PullbackDMDc-2d', 'PullbackDMDc-3d']
# (var, index, row label, colour, linestyle): categorical slots 1-3; both SLP caps share slot 3, N cap dashed.
SERIES = [('tas', 'gm', 'SAT global mean', '#2a78d6', '-'),
          ('tas_ocean', 'gm', 'OSAT global mean', '#eb6834', '-'),
          ('psl', 's65', 'SLP S of 65°S', '#1baf7a', '-'),
          ('psl', 'n65', 'SLP N of 65°N', '#1baf7a', '--')]


def load():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, 'main', '*.csv'))
                                                   + glob.glob(os.path.join(RES, 'domain', '*.csv')))],
                   ignore_index=True)
    eig_methods = df.method.isin(['LIM', 'LIM-opt']) | df.method.str.startswith('PullbackDMDc')
    df['hard_fail'] = (((df.lead_abs >= 1) & eig_methods) | df.nonfinite.fillna(False).astype(bool)
                       | df.fit_error.notna())
    return df


def sweeps(df):
    """Rows of each one-at-a-time sweep, keyed by config value x. The main config belongs to all three."""
    main_n = df['var'].map(n_eofs)
    is_main = (df.domain == 'main') & (df.N == main_n) & (df.lag == 3)
    out = {'N': df[(df.domain == 'main') & (df.lag == 3)].assign(x=lambda d: d.N),
           'lag': df[(df.domain == 'main') & (df.N == main_n)].assign(x=lambda d: d.lag),
           'domain': df[((df.domain != 'main') | is_main) & (df['var'] == 'tas_ocean')].assign(x=lambda d: d.domain)}
    return {k: v.assign(sweep=k) for k, v in out.items()}


def fixed_set(d):
    """Drop members with a hard failure for any method at any setting of this sweep."""
    bad = set(d.loc[d.hard_fail, ['var', 'esm', 'member']].itertuples(index=False, name=None))
    return d[[key not in bad for key in zip(d['var'], d.esm, d.member)]]


def quantiles(sw):
    rows = []
    for sweep, d in sw.items():
        d = fixed_set(d)
        for var, idx_name, *_ in SERIES:
            e = d[d['var'] == var]
            if e.empty:
                continue
            rmsd = np.sqrt(e[f'{idx_name}_mse_full']) / e[f'{idx_name}_truth_std'] * 100
            for (method, x), r in rmsd.groupby([e.method, e.x]):
                q = r.quantile([0.25, 0.5, 0.75]).values
                rows.append(dict(var=var, index=idx_name, sweep=sweep, x=x, method=method, n_members=len(r),
                                 q25=q[0], q50=q[1], q75=q[2]))
    return pd.DataFrame(rows)


def xpos(sweep, x):
    return DOMAINS.index(x) if sweep == 'domain' else x


def style_ax(ax, sweep, var, relative_n=False):
    if sweep == 'domain':
        ax.set_xticks(range(len(DOMAINS)))
        ax.set_xticklabels([DOMAIN_LABEL[d] for d in DOMAINS], fontsize=8)
        ax.set_xlim(-0.4, len(DOMAINS) - 0.6)
    else:
        if relative_n:
            vals, labels, main = [0.5, 1, 2, 4], ['×0.5', '×1', '×2', '×4'], 1
            xlabel = 'N / main N (SAT, OSAT: 20; SLP: 200)'
        elif sweep == 'N':
            vals = [100, 200, 400] if var == 'psl' else [10, 20, 40, 80]
            labels, main, xlabel = [str(v) for v in vals], n_eofs[var], 'truncation N (EOFs)'
        else:
            vals, main, xlabel = [1, 3, 6, 12], 3, 'lag (months)'
            labels = [str(v) for v in vals]
        ax.set_xscale('log', base=2)
        ax.set_xticks(vals)
        ax.set_xticklabels(labels, fontsize=8)
        ax.minorticks_off()
        ax.axvline(main, color='0.6', lw=0.8, ls='--', zorder=0)
        ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis='y', color='0.9', lw=0.6)
    ax.tick_params(labelsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def draw(ax, q, sweep, color, ls='-', label=None, band=True, scale=1.):
    px = np.array([xpos(sweep, x) for x in q.x]) / scale
    o = np.argsort(px)
    if band:
        ax.fill_between(px[o], q.q25.values[o], q.q75.values[o], color=color, alpha=0.15, lw=0)
    ax.plot(px[o], q.q50.values[o], color=color, ls=ls, lw=2, marker='o', ms=4, label=label)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=150)
    plt.close(fig)


def plot_pullback(qt):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, sweep in zip(axes, SWEEPS):
        for var, idx_name, label, color, ls in SERIES:
            q = qt[(qt['var'] == var) & (qt['index'] == idx_name) & (qt.sweep == sweep)
                   & qt.method.str.startswith('PullbackDMDc')]
            if not q.empty:
                draw(ax, q, sweep, color, ls, label, band=ls == '-', scale=n_eofs[var] if sweep == 'N' else 1.)
        style_ax(ax, sweep, None, relative_n=sweep == 'N')
        ax.set_title(SWEEP_TITLE[sweep], fontsize=10)
    axes[0].set_ylabel('RMSD (% of truth std)', fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=9, frameon=False, handlelength=3.5)
    fig.suptitle('PullbackDMDc: member median (line) and IQR (band) over the four ESMs; '
                 'dashed vertical = main setting', fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    save(fig, 'pullback_rmsd')


def plot_methods(qt):
    fig, axes = plt.subplots(len(SERIES), 3, figsize=(11, 2.6 * len(SERIES)), sharey='row')
    for r, (var, idx_name, label, *_) in enumerate(SERIES):
        for c, sweep in enumerate(SWEEPS):
            ax = axes[r, c]
            q = qt[(qt['var'] == var) & (qt['index'] == idx_name) & (qt.sweep == sweep)]
            if q.empty:
                ax.set_visible(False)
                continue
            for m in [m for m in METHOD_ORDER if m in set(q.method)]:
                draw(ax, q[q.method == m], sweep, colors[m], label=m)
            style_ax(ax, sweep, var)
            if r == 0 or (sweep == 'domain'):
                ax.set_title(SWEEP_TITLE[sweep], fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{label}\nRMSD (% std)', fontsize=9)
    h, l = {}, {}
    for ax in axes.ravel():  # union of methods over rows (-3d and -2d never share a row)
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            h.setdefault(ll, hh)
    labels = [m for m in METHOD_ORDER if m in h]
    fig.legend([h[m] for m in labels], labels, loc='lower center', ncol=len(labels), fontsize=9, frameon=False)
    fig.suptitle('Member median (line) and IQR (band) over the four ESMs; dashed vertical = main setting',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    save(fig, 'methods_rmsd')


def main():
    os.makedirs(FIG, exist_ok=True)
    sw = sweeps(load())
    kept = pd.concat({k: fixed_set(v).groupby(['var', 'esm']).member.nunique() for k, v in sw.items()}, names=['sweep'])
    total = pd.concat({k: v.groupby(['var', 'esm']).member.nunique() for k, v in sw.items()}, names=['sweep'])
    print('Members kept / total:\n', (kept.astype(str) + '/' + total.astype(str)).unstack('esm').to_string())

    qt = quantiles(sw)
    qt.round(4).to_csv(os.path.join(FIG, 'rmsd_quantiles.csv'), index=False)
    plot_pullback(qt)
    plot_methods(qt)
    print(f'Saved figures and rmsd_quantiles.csv to {FIG}')


if __name__ == '__main__':
    main()

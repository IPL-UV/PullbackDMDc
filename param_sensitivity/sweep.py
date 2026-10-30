"""R2-M5 parameter sweep: refit every method for one (var, ESM) at several truncations / lags and score it.

Methods are fitted from the precomputed EOF pickles exactly as in experiments.py, but nothing is pickled:
each fit is scored in PC space and one CSV row per (member, method, N, lag, history) is written, with
  - GM / polar-cap per-period MSE against the ensemble mean (as plot_gm_timeseries; psl truth smoothed 3 yr),
  - trend-map corr / normalised rmse over 1985-2014 (as compute_trend_maps, unsmoothed truth),
  - failure diagnostics (leading eigenvalue, e-folding time, nonfinite, fit errors).
Flags are derived from these columns later in plot_rmsd.py, so the rules can change without refitting.

Run from the repo root, e.g.
  python -m param_sensitivity.sweep --var tas --esm CESM2 --truncations 10,20,40,80 --lags 1,3,6,12 \
      --out $PBDMDC_ARTIFACT_ROOT/param_sensitivity/results/main/sweep_tas_CESM2.csv
"""
import argparse
import os
import pickle as pkl
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from evaluation.plot_gm_timeseries import is_in_volcanic
from evaluation.utils.metrics import global_mean, regional_mean, smooth_truth, coord_to_dtindex, lat_flat_from_pars
from evaluation.utils.taylor_diagram import TaylorDiagramEstimator
from evaluation.utils.trends import compute_linear_trend, window_mask
from utils.data_utils import load_data, load_forcings, load_forcings_pullback, load_forcings_2d, load_forcings_3d, load_truth
from utils.lim import LIM
from utils.lim_opt import LIM_opt
from utils.lr import LR
from utils.params import artifact_root, base_path, n_eofs, regions, transition_time
from utils.pullback_dmdc import PullbackDMDc
from utils.reg_gmst import RegGMST

SENS_ROOT = os.getenv('PBDMDC_SENS_ROOT', os.path.join(artifact_root, 'param_sensitivity'))
TREND_WINDOW = (1985, 2014)
PER_DECADE = 120.
SHORT_LEAD_MONTHS = 24.  # 'short_lead' flag threshold (2 yr)


def headline_pb(var):
    return 'PullbackDMDc-2d' if var == 'psl' else 'PullbackDMDc-3d'


def period_masks(time_index):
    t = np.array(time_index)
    volc = np.array([is_in_volcanic(x) for x in t])
    early = (t >= pd.Timestamp('1850-01-01')) & (t < pd.Timestamp('1980-01-01')) & ~volc
    late = (t >= pd.Timestamp('1980-01-01')) & (t < pd.Timestamp('2015-01-01')) & ~volc
    return {'full': np.ones_like(volc), 'volc': volc, 'early': early, 'late': late}


def load_truth_indices(var, esm, means_root):
    """Truth series (GM + polar caps, psl smoothed) and truth trend map, as in the main evaluation scripts."""
    truth_ds, lats = load_truth(var, esm, means_root, return_lats=True)
    trend_truth = truth_ds.stack(z=('lat', 'lon')).values.astype(np.float64)
    nan_cols = np.isnan(trend_truth[0, :])
    years = truth_ds['time'].dt.year.values
    tmask = window_mask(years, TREND_WINDOW)
    truth_trend = compute_linear_trend(trend_truth[:, ~nan_cols][tmask]) * PER_DECADE

    series_ds = smooth_truth(truth_ds, window_years=3) if var == 'psl' else truth_ds
    series = {'gm': np.ma.getdata(global_mean(series_ds, lats))}
    valid_lat = np.repeat(lats[:, None], truth_ds.sizes['lon'], axis=1).ravel()[~nan_cols]
    for r, (lo, hi) in regions.items():
        if ((valid_lat >= lo) & (valid_lat <= hi)).any():  # e.g. no polar caps in the 40S-65N tas_ocean domain
            series[r] = np.ma.getdata(regional_mean(series_ds, lats, lo, hi))
    time_index = coord_to_dtindex(truth_ds.time)
    return series, truth_trend, nan_cols, tmask, period_masks(time_index)


def index_weights(pars, index_names):
    """Linear functionals w so that index(field_flat) = field_flat @ w (global_mean_flat / regional_mean_flat)."""
    r = pars['resc_coef_flat']
    lat_flat = lat_flat_from_pars(pars)
    out = {}
    for name in index_names:
        sel = np.ones_like(r, dtype=bool) if name == 'gm' else (lat_flat >= regions[name][0]) & (lat_flat <= regions[name][1])
        w = np.where(sel, r, 0.)
        out[name] = w / (r[sel] ** 2).sum()
    return out


def lead_eig(op, lag):
    """Leading (least-damped) eigenvalue of a propagator over `lag` months and its diagnostics."""
    lam = np.linalg.eigvals(op)
    lead = lam[np.argmax(np.abs(lam))]
    mod = np.abs(lead)
    efold = -lag / np.log(mod) if 0 < mod < 1 else np.inf
    return {'lead_abs': mod, 'lead_imag': float(np.imag(lead)), 'lead_real_part': float(np.real(lead)),
            'lead_efold_months': efold}


def fit_method(method, eof_dict, N, lag, history, forcings, gmst):
    if method.startswith('PullbackDMDc'):
        short, long_ = forcings[method][history]
        m = PullbackDMDc(truncation=N, lag=lag, transition_time=history)
        m.fit(None, short, long_, precomputed_eofs=eof_dict)
        diag = lead_eig(m.A, lag)
    elif method == 'LIM':
        m = LIM(truncation=N, lag=lag)
        m.fit(None, precomputed_eofs=eof_dict)
        diag = lead_eig(m.A, lag)
    elif method == 'LIM-opt':
        m = LIM_opt(truncation=N, lag=lag)
        m.fit(None, precomputed_eofs=eof_dict)
        diag = lead_eig(m.M, lag)
    elif method == 'LR':
        m = LR(N)
        m.fit(None, forcings['LR'], precomputed_eofs=eof_dict)
        diag = {}
    elif method == 'RegGMST':
        m = RegGMST(N)
        m.fit(None, gmst['data'], gmst['resc'], precomputed_eofs=eof_dict)
        diag = {}
    return np.real(m.fr_pcs), m.eofs, diag


def score_member(eof_path, tas_path, var, methods, configs, forcings, truth, tde):
    series, truth_trend, nan_cols, tmask, pmasks = truth
    with open(eof_path, 'rb') as f:
        eof_dict = pkl.load(f)
    pars = eof_dict['data_pars']
    assert np.array_equal(pars['nan_mask'].ravel(), nan_cols), f'{eof_path}: nan mask differs from truth'
    weights = index_weights(pars, series.keys())
    gmst = None
    if 'RegGMST' in methods:
        data_tas, tas_pars = load_data(tas_path, 'tas')
        gmst = {'data': data_tas, 'resc': tas_pars['resc_coef_flat']}
    member = os.path.basename(eof_path)[:-4]

    rows = []
    for N, lag, history in configs:
        for method in methods:
            if history != transition_time and not method.startswith('PullbackDMDc'):
                continue  # baselines do not use the forcing history
            row = {'member': member, 'method': method, 'N': N, 'lag': lag, 'history': history, 'fit_error': ''}
            try:
                fr_pcs, eofs, diag = fit_method(method, eof_dict, N, lag, history, forcings, gmst)
            except np.linalg.LinAlgError as e:
                row['fit_error'] = f'LinAlgError: {e}'
                rows.append(row)
                continue
            row.update(diag)
            row['nonfinite'] = not np.isfinite(fr_pcs).all()
            if row['nonfinite']:
                rows.append(row)
                continue
            for name, w in weights.items():
                pred = fr_pcs @ (eofs @ w)
                ref = series[name]
                row[f'{name}_corr'] = np.corrcoef(pred, ref)[0, 1]
                for p, mask in pmasks.items():
                    row[f'{name}_mse_{p}'] = ((pred[mask] - ref[mask]) ** 2).mean()
                row[f'{name}_truth_std'] = ref.std()
            trend = compute_linear_trend(fr_pcs[tmask]) @ eofs * PER_DECADE
            td = tde.compute(truth_trend, trend)
            row['trend_corr'], row['trend_rmse'] = float(td['corr']), float(td['rmse'])
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--var', required=True)
    ap.add_argument('--esm', required=True)
    ap.add_argument('--truncations', default='', help='comma list; empty = main value only')
    ap.add_argument('--lags', default='3', help='comma list; non-3 lags use the main truncation')
    ap.add_argument('--histories', default='',
                    help='comma list of forcing histories before 1850 (months) at the main truncation and lag; '
                         'replaces the truncation / lag configs')
    ap.add_argument('--methods', default='', help='comma list; default = headline PullbackDMDc + 4 baselines')
    ap.add_argument('--eof-root', default=os.path.join(artifact_root, 'eofs'))
    ap.add_argument('--eof-root-large', default=os.path.join(SENS_ROOT, 'psl_n400', 'eofs'),
                    help='EOF store used when N exceeds the stored count (param_sensitivity/psl_eofs_n400.sbatch)')
    ap.add_argument('--means-root', default=os.path.join(artifact_root, 'means'))
    ap.add_argument('--tas-root', default=os.path.join(base_path, 'ensembles'), help='raw tas files for RegGMST')
    ap.add_argument('--domain', default='main', help='label written to the CSV')
    ap.add_argument('--max-members', type=int, default=0, help='>0: smoke test on the first members only')
    ap.add_argument('--n-jobs', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    start = time.time()
    var, esm = args.var, args.esm
    n_main = n_eofs[var]
    truncs = [int(x) for x in args.truncations.split(',') if x] or [n_main]
    lags = [int(x) for x in args.lags.split(',') if x]
    histories = [int(x) for x in args.histories.split(',') if x]
    configs = ([(n_main, 3, h) for h in histories] if histories else
               [(N, 3, transition_time) for N in truncs] + [(n_main, lag, transition_time) for lag in lags if lag != 3])
    methods = [m for m in args.methods.split(',') if m] or [headline_pb(var), 'LIM', 'LIM-opt', 'LR', 'RegGMST']

    forcings = {'LR': load_forcings(mean_center=True, rescale=True)}
    for m, loader in [('PullbackDMDc', load_forcings_pullback), ('PullbackDMDc-2d', load_forcings_2d),
                      ('PullbackDMDc-3d', load_forcings_3d)]:
        if m in methods:
            forcings[m] = {h: loader(h) for h in {c[2] for c in configs}}

    truth = load_truth_indices(var, esm, args.means_root)
    tde = TaylorDiagramEstimator(centered=False, normalized=True)

    # Configs above the stored EOF count use the larger store (same SVD, more EOFs).
    groups = {}
    for cfg in configs:
        groups.setdefault(cfg[0] > 200 and bool(args.eof_root_large), []).append(cfg)

    eof_dir = os.path.join(args.eof_root, var, esm)
    files = sorted(f for f in os.listdir(eof_dir) if f.endswith('.pkl') and 'historical' in f
                   and not ('CESM2' in f and 'cmip6' not in f))
    if args.max_members > 0:
        files = files[:args.max_members]
    print(f'{var} {esm} [{args.domain}]: {len(files)} members, configs {configs}, methods {methods}', flush=True)

    rows = []
    for large, cfgs in groups.items():
        root = args.eof_root_large if large else args.eof_root
        jobs = (delayed(score_member)(os.path.join(root, var, esm, f),
                                      os.path.join(args.tas_root, 'tas', esm, f.replace(var, 'tas', 1)[:-4] + '.nc'),
                                      var, methods, cfgs, forcings, truth, tde) for f in files)
        for member_rows in Parallel(n_jobs=args.n_jobs)(jobs):
            rows.extend(member_rows)

    df = pd.DataFrame(rows)
    df.insert(0, 'domain', args.domain)
    df.insert(0, 'esm', esm)
    df.insert(0, 'var', var)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'Saved {args.out} ({len(df)} rows) in {time.time() - start:.1f} s', flush=True)


if __name__ == '__main__':
    main()

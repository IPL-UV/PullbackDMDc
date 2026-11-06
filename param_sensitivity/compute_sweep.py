import argparse
import os
import pickle as pkl

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from evaluation.utils.metrics import smooth_truth
from evaluation.utils.taylor_diagram import Ensemble_PCA_TD_Estimator
from utils.data_utils import load_forcings_2d, load_forcings_3d, load_truth
from utils.params import artifact_root, n_eofs, transition_time, STORED_EOFS
from utils.pullback_dmdc import PullbackDMDc

SENS_ROOT = os.getenv('PBDMDC_SENS_ROOT', os.path.join(artifact_root, 'param_sensitivity'))
# Headline PullbackDMDc variant for each variable.
METHOD = {'tas': 'PullbackDMDc-3d', 'tas_ocean': 'PullbackDMDc-3d', 'psl': 'PullbackDMDc-2d'}


def load_truth_field(var, esm, means_root):
    # Ensemble-mean field as in compute_taylor (psl smoothed over 3 yr).
    truth_ds = load_truth(var, esm, means_root)
    if var == 'psl':
        truth_ds = smooth_truth(truth_ds, window_years=3)
    truth = truth_ds.stack(z=('lat', 'lon')).values.astype(np.float64)
    nan_cols = np.isnan(truth[0, :])
    return truth[:, ~nan_cols], nan_cols


def score_member(eof_path, var, configs, forcings, truth, nan_cols):
    with open(eof_path, 'rb') as f:
        eof_dict = pkl.load(f)
    assert np.array_equal(eof_dict['data_pars']['nan_mask'].ravel(), nan_cols), f'{eof_path}: nan mask differs'
    tde = Ensemble_PCA_TD_Estimator(centered=True, normalized=True)
    rows = []
    for N, lag, history in configs:
        row = {'member': os.path.basename(eof_path)[:-4], 'method': METHOD[var], 'N': N, 'lag': lag,
               'history': history, 'fit_error': ''}
        try:
            m = PullbackDMDc(truncation=N, lag=lag, transition_time=history)
            m.fit(None, *forcings[history], precomputed_eofs=eof_dict)
        except np.linalg.LinAlgError as e:
            row['fit_error'] = f'LinAlgError: {e}'
            rows.append(row)
            continue
        row['lead_abs'] = np.abs(np.linalg.eigvals(m.A)).max()
        fr_pcs = np.real(m.fr_pcs)
        row['nonfinite'] = not np.isfinite(fr_pcs).all()
        if not row['nonfinite']:
            # Taylor point as in compute_taylor / load_truncated_prediction; td_rmse = distance to (1, 0).
            svals = np.sqrt((m.eofs ** 2).sum(axis=-1))
            td = tde.compute(X=truth, P_Y_list=[fr_pcs * svals], V_Y_list=[m.eofs / svals[:, None]])[0][0]
            row.update(td_rmse=td['rmse'], td_corr=td['corr'], td_std=td['std_Y'])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--var', required=True)
    ap.add_argument('--esm', required=True)
    ap.add_argument('--truncations', default='')
    ap.add_argument('--lags', default='')
    ap.add_argument('--histories', default='', help='forcing history before 1850, months')
    ap.add_argument('--eof-root', default=os.path.join(artifact_root, 'eofs'))
    ap.add_argument('--eof-root-large', default=os.path.join(SENS_ROOT, 'psl_n400', 'eofs'))
    ap.add_argument('--means-root', default=os.path.join(artifact_root, 'means'))
    ap.add_argument('--n-jobs', type=int, default=8)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    var, esm, n_main = args.var, args.esm, n_eofs[args.var]
    ints = lambda s: [int(x) for x in s.split(',') if x]
    # One-at-a-time sweeps around the main config (N = n_eofs[var], lag 3, full forcing history).
    configs = [(n_main, 3, transition_time)]
    configs += [(N, 3, transition_time) for N in ints(args.truncations)]
    configs += [(n_main, lag, transition_time) for lag in ints(args.lags)]
    configs += [(n_main, 3, h) for h in ints(args.histories)]
    configs = sorted(set(configs))

    loader = load_forcings_2d if var == 'psl' else load_forcings_3d
    forcings = {h: loader(h) for h in {c[2] for c in configs}}
    truth, nan_cols = load_truth_field(var, esm, args.means_root)

    files = sorted(f for f in os.listdir(os.path.join(args.eof_root, var, esm)) if f.endswith('.pkl')
                   and 'historical' in f and not ('CESM2' in f and 'cmip6' not in f))
    print(f'{var} {esm}: {len(files)} members, {len(configs)} configs', flush=True)

    rows = []
    for large in (False, True):  # N above the stored EOF count comes from the larger store
        cfgs = [c for c in configs if (c[0] > STORED_EOFS) == large]
        root = args.eof_root_large if large else args.eof_root
        if cfgs:
            jobs = (delayed(score_member)(os.path.join(root, var, esm, f), var, cfgs, forcings, truth, nan_cols)
                    for f in files)
            rows += [r for member_rows in Parallel(n_jobs=args.n_jobs)(jobs) for r in member_rows]

    df = pd.DataFrame(rows)
    df.insert(0, 'esm', esm)
    df.insert(0, 'var', var)
    out = args.out or os.path.join(SENS_ROOT, 'results', 'main', f'sweep_{var}_{esm}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f'Saved {out} ({len(df)} rows)')


if __name__ == '__main__':
    main()

"""Run with: python -m tests.test_bootstrap"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.utils.bootstrap import member_bootstrap_rmsd, paired_difference


def old_bar_value(preds, truth, mask, norm_const):
    return np.sqrt(((preds[:, mask] - truth[None, mask]) ** 2).mean()) / norm_const * 100.


def make_case(seed=0, n_members=30, n_time=240):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n_time).cumsum()
    preds = truth[None, :] + rng.normal(scale=0.5, size=(n_members, n_time))
    mask = rng.random(n_time) > 0.4
    idx = np.random.default_rng(1).integers(0, n_members, size=(500, n_members))
    return preds, truth, mask, truth.std(), idx


def test_point_matches_old_formula():
    preds, truth, mask, norm_const, idx = make_case()
    point, _ = member_bootstrap_rmsd(preds, truth, mask, norm_const, idx)
    assert np.isclose(point, old_bar_value(preds, truth, mask, norm_const))


def test_identical_methods_give_zero_difference():
    preds, truth, mask, norm_const, idx = make_case()
    _, boot_a = member_bootstrap_rmsd(preds, truth, mask, norm_const, idx)
    _, boot_b = member_bootstrap_rmsd(preds.copy(), truth, mask, norm_const, idx)
    assert np.all(boot_a - boot_b == 0)
    lo, hi, ref_better = paired_difference(boot_a, boot_b)
    assert lo == 0 and hi == 0 and not ref_better


def test_much_larger_error_is_significant():
    preds, truth, mask, norm_const, idx = make_case()
    bad = truth[None, :] + 3.0 * (preds - truth[None, :])
    _, boot_ref = member_bootstrap_rmsd(preds, truth, mask, norm_const, idx)
    _, boot_bad = member_bootstrap_rmsd(bad, truth, mask, norm_const, idx)
    assert paired_difference(boot_bad, boot_ref)[2]
    assert not paired_difference(boot_ref, boot_bad)[2]


def test_same_seed_same_ci():
    preds, truth, mask, norm_const, _ = make_case()
    cis = []
    for _ in range(2):
        idx = np.random.default_rng(20260).integers(0, preds.shape[0], size=(500, preds.shape[0]))
        _, boot = member_bootstrap_rmsd(preds, truth, mask, norm_const, idx)
        cis.append(np.percentile(boot, [2.5, 97.5]))
    assert np.array_equal(cis[0], cis[1])


def test_draw_rmsd_bars_smoke():
    from evaluation.plot_gm_timeseries import draw_rmsd_bars
    time_index = pd.date_range('1850-01-01', '2014-12-01', freq='MS')
    rng = np.random.default_rng(3)
    truth = rng.normal(size=len(time_index)).cumsum() * 0.01
    names = [f'member_{i:02d}' for i in range(10)]
    methods = ['LR', 'LIM', 'PullbackDMDc-3d']
    scales = {'LR': 0.5, 'LIM': 0.3, 'PullbackDMDc-3d': 0.1}
    timeseries_model = {
        m: {'preds': list(truth[None, :] + rng.normal(scale=scales[m], size=(10, len(time_index)))),
            'truth': truth, 'time': time_index, 'data_names': names}
        for m in methods
    }
    fig, ax = plt.subplots()
    rows = draw_rmsd_bars(ax, timeseries_model, methods, truth, time_index, 'tas_ocean', seed=20260)
    plt.close(fig)
    assert len(rows) == len(methods) * 3
    for row in rows:
        assert row['ci_lo'] <= row['rmsd_pct'] <= row['ci_hi']
        assert row['n_members'] == 10
    # LR and LIM are much worse than the reference in every period.
    assert all(row['ref_significantly_better'] for row in rows if row['method'] in ('LR', 'LIM'))
    assert all(row['ref_significantly_better'] is None for row in rows if row['method'] == 'PullbackDMDc-3d')


if __name__ == '__main__':
    tests = [test_point_matches_old_formula, test_identical_methods_give_zero_difference,
             test_much_larger_error_is_significant, test_same_seed_same_ci, test_draw_rmsd_bars_smoke]
    for test in tests:
        test()
        print(f'ok  {test.__name__}')
    print('All bootstrap tests passed.')

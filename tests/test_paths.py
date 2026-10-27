"""Run with: python -m tests.test_paths"""
import os
from contextlib import contextmanager

from utils.params import (artifact_root, method_names, predictions_dir, run_suffixes, results_dir,
                          guard_main_results_write, year_suffix_for, eofs_dir)

PATH_ENV_VARS = ['PBDMDC_TRUNCATION', 'PBDMDC_RUN_TAG', 'PBDMDC_METHODS', 'PBDMDC_EOF_STORE']


@contextmanager
def env(**values):
    """Clear the path-related env vars, set the given ones, and restore everything afterwards."""
    saved = {name: os.environ.get(name) for name in PATH_ENV_VARS + list(values)}
    try:
        for name in saved:
            os.environ.pop(name, None)
        for name, value in values.items():
            os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_defaults_match_old_paths():
    with env():
        assert run_suffixes(3, (1850, 2014)) == ('', '')
        assert run_suffixes(12, (1950, 2014)) == ('_tier1', '_lag12')
        assert run_suffixes(6, (1850, 2014)) == ('', '_lag6')
        assert predictions_dir('', '') == os.path.join(artifact_root, 'models')
        # Old literal from evaluation/compute_taylor_vis.py with year_suffix = lag_suffix = ''.
        assert results_dir('', '') == os.path.join(artifact_root, 'evaluation_results', 'results')


def test_truncation_suffix():
    with env(PBDMDC_TRUNCATION='40'):
        assert run_suffixes(3, (1850, 2014)) == ('', '_n40')
        assert run_suffixes(6, (1850, 2014)) == ('', '_lag6_n40')
    with env(PBDMDC_TRUNCATION='20'):
        # Equal to the default truncation, but a sweep must never resolve to a main directory.
        assert run_suffixes(3, (1850, 2014)) == ('', '_n20')


def test_run_tag_suffix():
    with env(PBDMDC_RUN_TAG='fsmip'):
        assert run_suffixes(3, (1850, 2014)) == ('', '_fsmip')
    with env(PBDMDC_RUN_TAG='fsmip', PBDMDC_TRUNCATION='40'):
        assert run_suffixes(3, (1850, 2014)) == ('', '_n40_fsmip')


def test_guard_main_results_write():
    with env(PBDMDC_METHODS='LIM-opt'):
        try:
            guard_main_results_write('', method_names)
        except RuntimeError:
            pass
        else:
            raise AssertionError('guard should refuse a method subset in the main results dir')
        guard_main_results_write('_n40', method_names)
    with env():
        guard_main_results_write('', method_names)


def test_unsupported_year_range():
    try:
        year_suffix_for((1900, 2014))
    except ValueError:
        pass
    else:
        raise AssertionError('year_suffix_for should reject (1900, 2014)')


def test_eofs_dir():
    with env():
        assert eofs_dir('', 200) == os.path.join(artifact_root, 'eofs')
        assert eofs_dir('', 400).endswith('eofs_n400')


def print_resolved_dirs():
    with env():
        for year_range in [(1850, 2014), (1950, 2014)]:
            year_suffix, run_suffix = run_suffixes(3, year_range)
            print(f'  {year_range}: {predictions_dir(year_suffix, run_suffix)}')
            print(f'  {year_range}: {results_dir(year_suffix, run_suffix)}')


if __name__ == '__main__':
    tests = [test_defaults_match_old_paths, test_truncation_suffix, test_run_tag_suffix,
             test_guard_main_results_write, test_unsupported_year_range, test_eofs_dir]
    for test in tests:
        test()
        print(f'ok  {test.__name__}')
    print('Resolved dirs with no env vars set (lag 3):')
    print_resolved_dirs()
    print('All path tests passed.')

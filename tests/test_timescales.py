"""Run with: python -m tests.test_timescales"""
import numpy as np

from evaluation.utils.timescales import efolding_time, zero_crossing_time, eig_decay_time_years


def test_ar1_efolding():
    phi = 0.9
    acf = phi ** np.arange(200)
    expected = -1.0 / np.log(phi)  # ~9.49
    assert abs(efolding_time(acf, 1) - expected) / expected < 0.01


def test_cosine_zero_crossing():
    period = 48
    acf = np.cos(2 * np.pi * np.arange(200) / period)
    assert abs(4 * zero_crossing_time(acf, 1) - period) / period < 0.02


def test_never_crossing_is_nan():
    acf = np.linspace(1.0, 0.5, 100)
    assert np.isnan(efolding_time(acf, 1))
    assert np.isnan(zero_crossing_time(acf, 1))


def test_eig_decay_time():
    lag, tau = 3, 2.5
    eigval = np.exp(-lag / (12.0 * tau)) * np.exp(1j * 0.4)  # the angle must not matter
    assert np.isclose(eig_decay_time_years(eigval, lag), tau)


if __name__ == '__main__':
    tests = [test_ar1_efolding, test_cosine_zero_crossing, test_never_crossing_is_nan, test_eig_decay_time]
    for test in tests:
        test()
        print(f'ok  {test.__name__}')
    print('All timescale tests passed.')

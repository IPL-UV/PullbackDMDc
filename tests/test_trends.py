import numpy as np

from evaluation.compute_trend_taylor_viz import compute_linear_trend
from evaluation.utils.trends import parse_window, window_mask
from utils.data_utils import undo_lat_scaling


def test_linear_trend_exact():
    t = np.arange(120, dtype=float)
    a = np.array([1.0, -2.0, 0.5])
    b = np.array([0.01, -0.3, 2.0])
    field = a[None, :] + b[None, :] * t[:, None]
    assert np.allclose(compute_linear_trend(field), b, atol=1e-12)


def test_window_mask():
    years = np.repeat(np.arange(1850, 2015), 12)
    mask = window_mask(years, parse_window('1985-2014'))
    assert mask.sum() == 30 * 12
    assert np.flatnonzero(mask)[0] == (1985 - 1850) * 12
    assert np.flatnonzero(mask)[-1] == years.size - 1
    for bad in ('2014-1985', '1985', 'abc-2014'):
        try:
            parse_window(bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad!r} should raise')


def test_undo_lat_scaling_roundtrip():
    rng = np.random.default_rng(0)
    lat = np.linspace(-80, 80, 5)
    nlon = 8
    field = rng.normal(size=(5, nlon))
    scaled = field * np.sqrt(np.cos(np.deg2rad(lat)))[:, None]
    out = undo_lat_scaling(scaled.reshape(-1, 1), lat, nlon)[:, :, 0]
    assert np.allclose(out, field)

    nan_mask = np.zeros((5, nlon), dtype=bool)
    nan_mask[0, :3] = True
    nan_mask[4, 5] = True
    out = undo_lat_scaling(scaled[~nan_mask][:, None], lat, nlon, nan_mask)[:, :, 0]
    assert np.all(np.isnan(out[nan_mask]))
    assert np.allclose(out[~nan_mask], field[~nan_mask])


def test_pc_space_trend_equals_field_trend():
    rng = np.random.default_rng(1)
    pcs = rng.normal(size=(360, 6)).cumsum(axis=0)
    eofs = rng.normal(size=(6, 40))
    assert np.allclose(compute_linear_trend(pcs) @ eofs, compute_linear_trend(pcs @ eofs), atol=1e-10)


if __name__ == '__main__':
    test_linear_trend_exact()
    test_window_mask()
    test_undo_lat_scaling_roundtrip()
    test_pc_space_trend_equals_field_trend()
    print('test_trends: all passed')

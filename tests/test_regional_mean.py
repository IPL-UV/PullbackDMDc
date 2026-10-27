"""Run with: python -m tests.test_regional_mean"""
import numpy as np
import xarray as xr

from evaluation.utils.metrics import (global_mean, global_mean_flat, lat_flat_from_pars,
                                      regional_mean, regional_mean_flat)
from utils.params import regions

LATS = np.array([-80., -50., 0., 50., 80.])
NLON = 8
T = 6
BANDS = [regions['n65'], regions['s65'], (-60, 60)]


def make_case(field):
    """
    field: (T, nlat, nlon) in physical units. Returns the same data in the convention of
    load_data: scaled by sqrt(cos lat), flattened over valid points, plus pars and the gridded masked array.
    """
    nan_mask = np.zeros((LATS.size, NLON), dtype=bool)
    nan_mask[1, 2:5] = True
    nan_mask[3, 0] = True
    nan_mask[4, 7] = True

    resc_coef = np.sqrt(np.cos(LATS[:, None] / 180. * np.pi))
    scaled = field * resc_coef
    data_flat = scaled[:, ~nan_mask]
    resc_coef_flat = np.repeat(resc_coef, NLON, axis=-1)[~nan_mask]
    gridded = np.ma.masked_array(scaled, mask=np.broadcast_to(nan_mask, scaled.shape))
    pars = {'lat': xr.DataArray(LATS, dims='lat'), 'nlon': NLON, 'nan_mask': nan_mask,
            'resc_coef_flat': resc_coef_flat}
    return data_flat, resc_coef_flat, gridded, pars


def lat_dependent_field():
    rng = np.random.default_rng(0)
    return (LATS[None, :, None] / 10. + np.arange(T)[:, None, None]
            + rng.normal(size=(T, LATS.size, NLON)))


def test_constant_field():
    data_flat, resc_coef_flat, gridded, pars = make_case(np.full((T, LATS.size, NLON), 3.5))
    lat_flat = lat_flat_from_pars(pars)
    for band in BANDS:
        assert np.allclose(regional_mean_flat(data_flat, resc_coef_flat, lat_flat, *band), 3.5)
        assert np.allclose(regional_mean(gridded, LATS, *band), 3.5)


def test_flat_matches_gridded():
    data_flat, resc_coef_flat, gridded, pars = make_case(lat_dependent_field())
    lat_flat = lat_flat_from_pars(pars)
    for band in BANDS:
        flat = regional_mean_flat(data_flat, resc_coef_flat, lat_flat, *band)
        grid = regional_mean(gridded, LATS, *band)
        assert flat.shape == (T,)
        assert np.allclose(flat, grid)


def test_whole_globe_matches_global_mean():
    data_flat, resc_coef_flat, gridded, pars = make_case(lat_dependent_field())
    lat_flat = lat_flat_from_pars(pars)
    assert np.allclose(regional_mean_flat(data_flat, resc_coef_flat, lat_flat, -90, 90),
                       global_mean_flat(data_flat, resc_coef_flat))
    assert np.allclose(regional_mean(gridded, LATS, -90, 90), global_mean(gridded, LATS))


def test_lat_flat_ordering():
    _, resc_coef_flat, _, pars = make_case(lat_dependent_field())
    lat_flat = lat_flat_from_pars(pars)
    pars_np = dict(pars, lat=LATS)
    assert np.array_equal(lat_flat, lat_flat_from_pars(pars_np))
    # Same flat ordering as resc_coef_flat in load_data.
    assert np.allclose(np.sqrt(np.cos(lat_flat / 180. * np.pi)), resc_coef_flat)


def test_empty_band_raises():
    data_flat, resc_coef_flat, gridded, pars = make_case(lat_dependent_field())
    lat_flat = lat_flat_from_pars(pars)
    for call in (lambda: regional_mean_flat(data_flat, resc_coef_flat, lat_flat, 85, 90),
                 lambda: regional_mean(gridded, LATS, 85, 90)):
        try:
            call()
        except AssertionError:
            pass
        else:
            raise AssertionError('empty latitude band should raise')


if __name__ == '__main__':
    tests = [test_constant_field, test_flat_matches_gridded, test_whole_globe_matches_global_mean,
             test_lat_flat_ordering, test_empty_band_raises]
    for test in tests:
        test()
        print(f'ok  {test.__name__}')
    print('All regional mean tests passed.')

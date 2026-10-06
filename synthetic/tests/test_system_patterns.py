import numpy as np

from ablation_data import B_HAT, M, W_INV_REF, W_REF
from system_patterns import CO2_ZONAL_FORCING, co2_forcing_pattern, lat_grid


def test_pattern_orthogonality_and_forcing_overlaps():
    assert np.allclose(np.linalg.norm(W_REF, axis=0), 1)
    assert np.abs(W_INV_REF @ W_REF - np.eye(len(W_REF))).max() < 1e-12
    assert np.abs(W_REF[:, :3].T @ W_REF[:, 3:]).max() < 1e-12
    gram = np.column_stack([W_REF[:, :3], B_HAT])
    gram = gram.T @ gram
    expected = {(0, 1): 0.396, (0, 2): 0.329, (1, 2): 0.278, (3, 0): 0.650, (3, 1): 0.489, (3, 2): 0.478}
    for (i, j), value in expected.items():
        assert abs(gram[i, j] - value) < 5e-4, (i, j, gram[i, j])


def test_co2_forcing_pattern_shape():
    b = B_HAT
    assert np.all(b > 0) and np.allclose(b, b[::-1], atol=1e-12)
    assert np.argmax(b) in (M // 2 - 1, M // 2)
    ratio = CO2_ZONAL_FORCING[0] / CO2_ZONAL_FORCING[-1]
    assert abs(ratio / (2.50 / 1.54) - 1) < 0.02, ratio
    raw = co2_forcing_pattern(lat_grid(M))
    assert np.allclose(b, raw / np.linalg.norm(raw), atol=1e-15)

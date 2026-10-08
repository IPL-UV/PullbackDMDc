"""A run is a pure function of its config: same seeds in, same numbers out.

Every draw in the package goes through `np.random.default_rng` with a seed that lives in `Config`
(`pattern_seed`, `eigenvalue_seed`, `noise_seed`), and the fits are closed-form linear algebra, so
`results/<run>/config.json` is enough to reproduce a run exactly. These tests hold that property down:
the first two check that the seeds are wired up and load-bearing, and the third pins the actual draws,
so a change of seed, of numpy's generator stream, or of the QR sign convention in `make_W` fails here
rather than silently moving every number in the paper.

Not covered, because it is a property of the environment rather than of the code: `--n-jobs 1` makes
joblib fall back to SequentialBackend, which skips `inner_max_num_threads=1`, so a serial run can differ
from a parallel one in the last bits of each BLAS reduction. See README.md.
"""

import numpy as np

from ablation_data import REFERENCE, equal_budget, make_dataset, sweep, total_snr_sweep
from config import DEFAULT, with_overrides
from support import SMALL


def test_the_same_config_gives_the_same_data_twice():
    """Two independent builds of one sweep level agree bit for bit, noise included."""
    first, second = (sweep("total_snr", [1 / 3], DEFAULT, **SMALL)[0] for _ in range(2))
    assert np.array_equal(first.y, second.y)
    assert np.array_equal(first.forced, second.forced)
    assert np.array_equal(first.internal, second.internal)  # the only part that is drawn
    assert np.array_equal(first.system.W, second.system.W)
    assert np.array_equal(first.system.lam_c, second.system.lam_c)


def test_the_noise_seed_reaches_the_sweep():
    """A different noise_seed moves the realizations, and nothing else.

    This is what keeps `sweep`'s `seed=cfg.noise_seed` honest: `make_dataset`'s own default is 0, which
    is also DEFAULT.noise_seed, so dropping the wiring would leave every other test passing.
    """
    base = sweep("total_snr", [1 / 3], DEFAULT, **SMALL)[0]
    other = sweep("total_snr", [1 / 3], with_overrides(DEFAULT, ["noise_seed=1"]), **SMALL)[0]
    assert not np.allclose(base.internal, other.internal)
    # the forced response and the system are noise-free, so they must not move with the noise seed
    assert np.array_equal(base.forced, other.forced)
    assert np.array_equal(base.y, other.y)
    assert np.array_equal(base.system.W, other.system.W)


def test_common_random_numbers_across_seeds_stay_paired():
    """Realization r is the same draw at every level of a sweep, whichever seed is used.

    test_ablation_data.test_common_random_numbers pins this for the default seed; the point here is that
    it is a property of the construction rather than of seed 0.
    """
    cfg = with_overrides(DEFAULT, ["noise_seed=7"])
    levels = total_snr_sweep(snrs=[1 / 3, 3], cfg=cfg, **SMALL)
    scaled = [ds.internal / np.sqrt(ds.system.s1_sq) for ds in levels]
    assert np.abs(scaled[0] - scaled[1]).max() < 1e-10


def test_the_seeded_draws_are_pinned():
    """Golden values for the two structural draws, so an RNG-stream change cannot pass silently.

    The existing pattern tests assert norms and orthogonality, which hold for ANY seed. These do not:
    lam_c comes from eigenvalue_seed=20 and the complement columns of W from pattern_seed=22, through
    make_W's Haar rotation. Tolerance 1e-12 -- these are exact draws, not accumulated arithmetic.
    """
    assert np.allclose(REFERENCE.lam_c[:4],
                       [0.028007596263, 0.046114670980, 0.012171969258, 0.052260835172], atol=1e-12)
    assert np.allclose(REFERENCE.W[:4, 3],
                       [-0.022419741326, 0.028828478080, -0.072772254117, -0.342604485870], atol=1e-12)
    # a scalar over the whole complement block: catches a reordering that leaves column 3 alone.
    # The Frobenius norm would not -- the block is orthonormal, so it is sqrt(17) for every seed.
    assert abs(float(REFERENCE.W[:, 3:].sum()) - (-3.9646728595672527)) < 1e-12


def test_a_dataset_is_reproducible_from_its_system():
    """make_dataset is pure in (system, budget, n_realizations, seed) -- no hidden global state."""
    first = make_dataset(REFERENCE, equal_budget, seed=3, n_realizations=2)
    second = make_dataset(REFERENCE, equal_budget, seed=3, n_realizations=2)
    assert np.array_equal(first.internal, second.internal)
    assert not np.allclose(first.internal, make_dataset(REFERENCE, equal_budget, seed=4,
                                                        n_realizations=2).internal)

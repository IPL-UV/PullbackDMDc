import pathlib
import tempfile

from config import DEFAULT, config_diff, from_json, generator_kwargs, slug, to_json, with_overrides
from synthetic_generator import generate_ensemble


def test_overrides():
    cfg = with_overrides(DEFAULT, ["tau1=50", "total_snrs=[0.1, 1]", "methods=['LIM']"])
    assert cfg.tau1 == 50 and cfg.total_snrs == (0.1, 1) and cfg.methods == ("LIM",)
    assert config_diff(cfg) == "tau1=50, total_snrs=(0.1, 1), methods=(LIM,)"
    assert slug(DEFAULT) == "default" and "/" not in slug(cfg)
    hash(cfg)
    for bad in (["tau_1=50"], ["tau1"]):
        try:
            with_overrides(DEFAULT, bad)
        except (KeyError, ValueError):
            continue
        raise AssertionError(f"{bad} should raise")


def test_generator_kwargs_are_generate_ensemble_arguments():
    """Every system field reaches generate_ensemble under its own name; tau1, split and b_rot are his main block's."""
    generate_ensemble(1, **{**generator_kwargs(DEFAULT), "spin_up_factor": 1}, rng=0)
    assert (DEFAULT.tau1, DEFAULT.split, DEFAULT.b_rot) == (20, (1, 1, 1, 0.9, 0.1), 0.0)
    assert abs(DEFAULT.snr - 1 / 3) < 1e-12


def test_config_json_roundtrip():
    cfg = with_overrides(DEFAULT, ["tau1=50", "split=[1, 2, 1, 1, 0]", "lag=1"])
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "config.json"
        to_json(cfg, path)
        assert from_json(path) == cfg

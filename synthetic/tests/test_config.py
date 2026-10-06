import pathlib
import tempfile

from ablation_data import build_reference, eigenvalue
from config import DEFAULT, config_diff, from_json, slug, to_json, with_overrides


def test_overrides():
    cfg = with_overrides(DEFAULT, ["tau1_yr=50", "total_snrs=[0.1, 1]", "methods=['LIM']"])
    assert cfg.tau1_yr == 50 and cfg.total_snrs == (0.1, 1) and cfg.methods == ("LIM",)
    assert config_diff(cfg) == "tau1_yr=50, total_snrs=(0.1, 1), methods=(LIM,)"
    assert slug(DEFAULT) == "default" and "/" not in slug(cfg)
    hash(cfg)
    for bad in (["tau_1=50"], ["tau1_yr"]):
        try:
            with_overrides(DEFAULT, bad)
        except (KeyError, ValueError):
            continue
        raise AssertionError(f"{bad} should raise")
    ref = build_reference(cfg)
    assert ref.system.lam1 == eigenvalue(600)


def test_config_json_roundtrip():
    cfg = with_overrides(DEFAULT, ["tau1_yr=50", "mode_overlaps=[0.1, 0.2]", "lag=3"])
    file_cfg = with_overrides(DEFAULT, ["forcing_source=file", "forcing_file=/some/where/forcing.csv",
                                        "forcing_column=total"])
    assert (file_cfg.forcing_source, file_cfg.forcing_file, file_cfg.forcing_column) == (
        "file", "/some/where/forcing.csv", "total")
    with tempfile.TemporaryDirectory() as tmp:
        for c in (cfg, file_cfg):
            path = pathlib.Path(tmp) / "config.json"
            to_json(c, path)
            assert from_json(path) == c

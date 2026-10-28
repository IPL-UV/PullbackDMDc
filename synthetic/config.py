"""Every tunable value of the synthetic ablations, in one frozen dataclass.

The system fields are the keyword arguments of synthetic_generator.generate_ensemble, under the same names."
"""

import ast
import json
import pathlib
from dataclasses import asdict, dataclass, fields, replace
from typing import Tuple

SYNTHETIC_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = SYNTHETIC_DIR / "results"
FIGURES_DIR = SYNTHETIC_DIR / "figures"


@dataclass(frozen=True)
class Config:
    # --- system: generate_ensemble's arguments (see synthetic_generator.py) -------------------
    tau1: float = 20  # slow mode decay time (yr)
    variances: Tuple[float, ...] = (1., 1, 1, 0.99, 0.01)  # V_f : V_1 : V_p : V_c : V_eps
    delta: float = 0.2  # depth of the forcing's 1965 dip
    alpha: float = 0.0  # overlap of the complement modes with w1; 0: orthogonal
    spin_up_factor: float = 100  # spin-up in units of the slowest decay time
    w_seed: int = 22
    eig_seed: int = 20
    tau_pair: float = 1.5  # fast pair decay time (yr)
    period_pair: float = 4.0  # fast pair period (yr)
    lam_noise_max: float = 0.1  # noise-mode eigenvalues are uniform on (0, lam_noise_max)

    # --- realizations -----------------------------------------------------------------------
    noise_seed: int = 0
    n_realizations: int = 1000

    # --- sweep levels -----------------------------------------------------------------------
    total_snrs: Tuple[float, ...] = (1 / 30, 1 / 10, 1 / 3, 1, 3, 10, 30)
    partial_snr_factors: Tuple[float, ...] = (1 / 4, 1 / 2, 1, 2, 4)
    slow_timescales_yr: Tuple[float, ...] = (1, 2, 5, 10, 20, 40, 80)
    obs_noise_ratios: Tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1, 1, 10)  # V_eps / V_f
    alphas: Tuple[float, ...] = (0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.96,0.97,0.99)

    # --- methods ----------------------------------------------------------------------------
    lag: int = 1
    optlag: int = 60  # LIM-opt horizon (months)
    methods: Tuple[str, ...] = ("PullbackDMDc", "LIM", "LIM-opt", "LR")

    @property
    def snr(self):
        """The reference system's theoretical SNR."""
        return self.variances[0] / sum(self.variances[1:])


GENERATOR_KEYS = ("tau1", "variances", "delta", "alpha", "spin_up_factor", "w_seed", "eig_seed",
                  "tau_pair", "period_pair", "lam_noise_max")


def generator_kwargs(cfg):
    """cfg's system as keyword arguments of generate_ensemble."""
    return {key: getattr(cfg, key) for key in GENERATOR_KEYS}


DEFAULT = Config()
FIELDS = {f.name for f in fields(Config)}


def _freeze(value):
    return tuple(_freeze(v) for v in value) if isinstance(value, (list, tuple)) else value


def with_overrides(cfg, assignments):
    """Apply `key=value` strings; values are Python literals, lists become tuples."""
    updates = {}
    for item in assignments or ():
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep:
            raise ValueError(f"expected key=value, got {item!r}")
        if key not in FIELDS:
            raise KeyError(f"unknown config key {key!r}; valid keys: {', '.join(sorted(FIELDS))}")
        try:
            value = ast.literal_eval(raw.strip())
        except (ValueError, SyntaxError):
            value = raw.strip()  # bare strings
        updates[key] = _freeze(value)
    return replace(cfg, **updates)


def to_json(cfg, path):
    path.write_text(json.dumps(asdict(cfg), indent=2) + "\n")


def from_json(path):
    data = json.loads(path.read_text())
    unknown = set(data) - FIELDS
    if unknown:
        raise KeyError(f"unknown config keys in {path}: {sorted(unknown)}")
    return Config(**{k: _freeze(v) for k, v in data.items()})


def config_diff(cfg, base=DEFAULT):
    """Short label of the fields that differ from base, e.g. 'tau1=50, lag=3'."""
    return ", ".join(f"{f.name}={getattr(cfg, f.name)!r}".replace("'", "")
                     for f in fields(Config) if getattr(cfg, f.name) != getattr(base, f.name))


def run_name(cfg):
    """Directory name of a run and of its figures: by dip depth, e.g. 'delta0.15'."""
    return f"delta{cfg.delta:g}"


def slug(cfg):
    label = config_diff(cfg)
    return "default" if not label else "".join(c if c.isalnum() or c in "._=" else "_" for c in label.replace(", ", "__"))


def load_config(args):
    """The config an entry point runs with: results/<from_run>/config.json or DEFAULT, then --set overrides."""
    cfg = from_json(RESULTS_DIR / args.from_run / "config.json") if args.from_run else DEFAULT
    return with_overrides(cfg, args.set)

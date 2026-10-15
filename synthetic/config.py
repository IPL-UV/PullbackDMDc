"""Every tunable value of the synthetic ablations, in one frozen dataclass.

The system fields are the keyword arguments of synthetic_generator.generate_ensemble, under the same names.
Override from the command line with `--set key=value ...`, e.g.
    python run_ablation_studies.py --name slow50 --set tau1=50
    python run_ablation_studies.py --name dip1 --set depth=1                     # a much deeper dip
    python plot_generator_diagnostics.py --set "split=(1, 1, 1, 0.9, 0.5)"
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
    split: Tuple[float, ...] = (1, 1, 1, 0.9, 0.1)  # variance shares (V_f, slow, pair, noise modes, measurement)
    b_rot: float = 0.0  # fraction of the angle rotating the flat B toward w1; 1: B = w1
    depth: float = 0.25  # amplitude of the forcing's 1965 dip
    spin_up_factor: float = 100  # spin-up in units of the slowest decay time
    w_seed: int = 22
    eig_seed: int = 20
    tau_pair: float = 1.5  # fast pair decay time (yr)
    period_pair: float = 4.0  # fast pair period (yr)
    lam_noise_max: float = 0.1  # noise-mode eigenvalues are uniform on (0, lam_noise_max)

    # --- realizations -----------------------------------------------------------------------
    noise_seed: int = 0
    n_realizations: int = 100

    # --- sweep levels -----------------------------------------------------------------------
    total_snrs: Tuple[float, ...] = (1 / 30, 1 / 10, 1 / 3, 1, 3, 10, 30)
    partial_snr_factors: Tuple[float, ...] = (1 / 4, 1 / 2, 1, 2, 4)
    slow_timescales_yr: Tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100)
    obs_noise_shares: Tuple[float, ...] = (0.1, 0.2, 0.5, 1, 2, 3)  # measurement-noise share v, his __main__ values
    b_rots: Tuple[float, ...] = (0, 0.25, 0.5, 0.75, 1)

    # --- methods ----------------------------------------------------------------------------
    lag: int = 3
    optlag: int = 60  # LIM-opt horizon (months)
    methods: Tuple[str, ...] = ("PullbackDMDc", "LIM", "LIM-opt", "LR")

    @property
    def snr(self):
        """The reference system's theoretical SNR."""
        return self.split[0] / sum(self.split[1:])


GENERATOR_KEYS = ("tau1", "split", "b_rot", "depth", "spin_up_factor", "w_seed", "eig_seed",
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


def slug(cfg):
    label = config_diff(cfg)
    return "default" if not label else "".join(c if c.isalnum() or c in "._=" else "_" for c in label.replace(", ", "__"))


def load_config(args):
    """The config an entry point runs with: results/<from_run>/config.json or DEFAULT, then --set overrides."""
    cfg = from_json(RESULTS_DIR / args.from_run / "config.json") if args.from_run else DEFAULT
    return with_overrides(cfg, args.set)

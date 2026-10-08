"""Every tunable value of the synthetic ablations, in one frozen dataclass.

Override from the command line with `--set key=value ...`, e.g.
    python run_ablation_studies.py --name slow50 --set tau1_yr=50 slow_variance=0.5
    python plot_system_diagnostics.py --set tau_p_yr=5 period_p_yr=8
    python run_ablation_studies.py --name ar6 --set forcing_source=file                     # AR6 CO2 from the default file
    python run_ablation_studies.py --name exp --set forcing_source=analytic                 # plain exp ramp
    python run_ablation_studies.py --name dip1 --set gauss_dip_amp=1                        # a much deeper dip
    python plot_forcing_comparison.py --set gauss_bump_amp=0.05 gauss_bump_year=1940         # reshape the Gaussian model
    python run_ablation_studies.py --name total --set forcing_source=file forcing_column=total forcing_file=/abs/forcing.csv

Grid size M = 20 and record length N = 1980 months (Jan 1850 - Dec 2014) are structural and stay
fixed in ablation_data.py.
"""

import ast
import json
import pathlib
from dataclasses import asdict, dataclass, fields, replace
from typing import Optional, Tuple

SYNTHETIC_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = SYNTHETIC_DIR / "results"
LEGACY_KEYS = {"forced_variance"}  # dropped fields that older results/*/config.json still carry


@dataclass(frozen=True)
class Config:
    # --- timescales -------------------------------------------------------------------------
    tau1_yr: float = 20  # slow mode e-folding time
    tau_p_yr: float = 2  # oscillating pair damping time
    period_p_yr: float = 4  # oscillating pair period
    complement_eig_range: Tuple[float, float] = (0.0, 0.1)  # 17 fast modes; eig 0.1 is tau ~0.43 months

    # --- variances: internal variance of each mode group, in units of the forced variance V_f --
    # reference SNR = 1 / (slow_variance + pair_variance + complement_variance)
    slow_variance: float = 1.0  # s1^2 = slow_variance * V_f
    pair_variance: float = 1.0  # sp^2 = pair_variance * V_f / 2 per pair mode
    complement_variance: float = 1.0  # sc^2 = complement_variance * V_f / 17 per complement mode

    # --- forcing in time (CO2 zonal pattern in space) ----------------------------------------
    # Always centered on the observed interval (the record): generation, the methods' input and every plot.
    # "analytic": c + a exp((t - 2014) / forcing_efold_yr), c, a fitted to the AR6 CO2 ERF (forcing_file/column ignored)
    # "analytic_gauss": c + a exp((t - 2014) / gauss_efold_yr) + bump - dip, c, a fitted jointly with the Gaussians
    #         to the AR6 CO2 ERF (forcing_efold_yr and forcing_file/column ignored)
    # "file": column forcing_column of forcing_file, an annual CSV with a `year` column or a monthly CSV with a
    #         `time` column (YYYY-MM-01); a relative path is relative to the repo root (forcing_efold_yr ignored)
    forcing_source: str = "analytic_gauss"
    forcing_file: str = "data_preparation/AR6_ERF_1750-2019.csv"
    forcing_column: str = "co2"
    record_end_year: int = 2014  # the 165-yr record ends in December of this year (1850-2014)
    forcing_efold_yr: float = 60  # e-folding time of the exp ramp (fit: 59.8 yr)
    # analytic_gauss shape. Amplitudes are W m^-2 next to a = 1.96 (only their ratio to the exp matters, since the
    # overall scale is removed); amplitude 0 removes that Gaussian. Defaults: one dip, no bump. The joint fit to
    # AR6 CO2 is efold 63.2 yr, bump 0.048 at 1921 (sigma 17.9 yr), dip 0.137 at 1966 (sigma 14.1 yr).
    gauss_efold_yr: float = 65  # e-folding time of the Gaussian model's exp
    gauss_bump_amp: float = 0  # positive Gaussian (>= 0)
    gauss_bump_year: float = 1920
    gauss_bump_width_yr: float = 20  # sigma (> 0)
    gauss_dip_amp: float = 0.25  # negative Gaussian, subtracted (>= 0)
    gauss_dip_year: float = 1965
    gauss_dip_width_yr: float = 15  # sigma (> 0)

    # --- seeds and forcing history ----------------------------------------------------------
    pattern_seed: int = 22
    eigenvalue_seed: int = 20
    noise_seed: int = 0
    # Forcing history given to the methods, in months. `history_decay_times` e-foldings of the slow mode,
    # floored at MIN_HISTORY, so every sweep level gets a window matched to its own memory rather than one
    # fixed length that is generous at tau_1 = 1 yr and a single e-folding at 100 yr. `history` overrides it
    # with a literal month count (--set history=2400); None derives it.
    #
    # 30 is deliberately below SPINUP_DECAY_TIMES (40): the spin-up that generates the truth then reaches
    # further back than the window any method sees, so the forced response is never fully reconstructible
    # from the forcing on offer. The margin is structural, not numerical -- at 30 e-foldings the true A, B
    # already reproduce the truth to 2e-13 -- and keeping it small is what keeps the long integration cheap.
    history_decay_times: int = 30
    history: Optional[int] = None

    # --- sweep levels -----------------------------------------------------------------------
    total_snrs: Tuple[float, ...] = (1 / 30, 1 / 10, 1 / 3, 1, 3, 10, 30)
    partial_snr_factors: Tuple[float, ...] = (1 / 4, 1 / 2, 1, 2, 4)
    slow_timescales_yr: Tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100)
    mode_overlaps: Optional[Tuple[float, ...]] = None  # None: (0, 0.25, base overlap, 0.75, 0.9, 0.95)
    # cos angle(w_1, b-hat): 0 is a slow mode orthogonal to the forcing pattern, 1 is w_1 = b-hat
    b_overlaps: Optional[Tuple[float, ...]] = None  # None: (0, 0.25, 0.5, base overlap, 0.8, 0.9, 1.0)
    n_realizations: int = 100

    # --- methods ----------------------------------------------------------------------------
    lag: int = 1
    optlag: int = 60  # LIM-opt horizon (months)
    methods: Tuple[str, ...] = ("PullbackDMDc", "LIM", "LIM-opt", "LR")

    @property
    def variances(self):
        return self.slow_variance, self.pair_variance, self.complement_variance


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
    data = {k: v for k, v in json.loads(path.read_text()).items() if k not in LEGACY_KEYS}
    unknown = set(data) - FIELDS
    if unknown:
        raise KeyError(f"unknown config keys in {path}: {sorted(unknown)}")
    return Config(**{k: _freeze(v) for k, v in data.items()})


def config_diff(cfg, base=DEFAULT):
    """Short label of the fields that differ from base, e.g. 'tau1_yr=50, lag=3'."""
    return ", ".join(f"{f.name}={getattr(cfg, f.name)!r}".replace("'", "")
                     for f in fields(Config) if getattr(cfg, f.name) != getattr(base, f.name))


def slug(cfg):
    label = config_diff(cfg)
    return "default" if not label else "".join(c if c.isalnum() or c in "._=" else "_" for c in label.replace(", ", "__"))


def load_config(args):
    """The config an entry point runs with: results/<from_run>/config.json or DEFAULT, then --set overrides."""
    cfg = from_json(RESULTS_DIR / args.from_run / "config.json") if args.from_run else DEFAULT
    return with_overrides(cfg, args.set)

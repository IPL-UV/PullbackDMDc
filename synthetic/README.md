# Synthetic experiments

A 20-latitude linear system with a slow mode, a fast oscillating (ENSO-like) pair and 17 noise modes, driven by a
CO₂-like forcing and observed with measurement noise. Four estimators (`PullbackDMDc`, `LIM`, `LIM-opt`, `LR`)
are fitted to every member of an ensemble, at every level of every ablation study. Each is scored on how well
it recovers the forced response and the slow mode.

The system is defined in **`synthetic_generator.py`**, written by Andrei Gavrilov. Its docstring is the
specification. Everything else in this folder is a thin layer on top of it. Please keep his file in his style,
and put new logic in the other files.

## Run it

```bash
./run_synthetic_experiments.sh   # both ablation runs and every figure
./run_synthetic_tests.sh         # the unit tests (run_tests.py; --only <substring>, --list)
```

The drivers activate the `dmdc_variants` env, `cd` into `synthetic/` and export `OMP_NUM_THREADS=1` (the per-step
simulation loop is BLAS-thread bound) and `MPLBACKEND=Agg`. Scripts are run from inside `synthetic/`. Use
`--n-jobs 12` for ablation runs on the shared node.

```bash
python plot_generator_diagnostics.py                                    # his three figures, reference system
python plot_generator_diagnostics.py --study obs_noise --level 1       # ... at any ablation level
python run_ablation_studies.py --name slow50 --set tau1=50 --n-jobs 12  # a run with a 50-yr slow mode
python plot_ablation_run_results.py --runs baseline slow50              # overlaid on the baseline
```

`--set KEY=VALUE` overrides any `Config` field. Values are Python literals, e.g. `--set "split=(1, 1, 1, 0.9, 0.5)"`.
`--from-run NAME` starts from a saved run's `config.json` instead of the defaults.

## Files

| file | what it is |
|---|---|
| `synthetic_generator.py` | **his**: `make_setup` (W, B, forcing), `generate_ensemble` (the ensemble dict), `fit_pullback_ensemble` |
| `plot_generator_diagnostics.py` | **his** plots (`plot_setup`, `plot_ensemble_diagnostics`, `plot_pullback_results`) and a CLI → `figures/generator/<config>/` |
| `config.py` | `Config`: the system fields are `generate_ensemble`'s arguments under the same names, plus seeds, sweep levels and methods |
| `ablations.py` | `STUDIES`: how each study bends the reference; `make_level(cfg, study, level)` returns `generate_ensemble`'s dict |
| `methods.py` | the four estimators, fitted one member at a time on identity EOFs |
| `run_ablation_studies.py` | fits and scores every level → `results/<run>/` |
| `plot_ablation_run_results.py` | sweep, phase-diagram, slow-mode-shape and forcing-change figures → `figures/ablations/<run>/` |
| `plot_style.py` | shared matplotlib style of the results figures |

## Reference system (`Config` defaults)

These are the defaults of his `__main__` block:
- `tau1=20` yr;
- `split=(1, 1, 1, 0.9, 0.1)`, which gives the variance shares of the forced response, slow mode, fast pair,
  noise modes and measurement noise, so SNR = 1/3;
- `b_rot=0` (flat forcing pattern B), `depth=0.25`, forcing amplitude `a = 2`;
- `spin_up_factor=100` decay times;
- `tau_pair=1.5` yr and `period_pair=4` yr;
- 100 members, fitted at `lag=3`.

## Ablation studies

| study | levels | what changes (`generate_ensemble` argument) |
|---|---|---|
| `total_snr` | SNR 1/30 … 30 | every noise share in `split`, scaled by `snr_ref / snr` |
| `partial_snr_{slow,pair,complement}` | factor ¼ … 4 | one noise share in `split` |
| `slow_timescale_snr` | τ1 = 1 … 100 yr | `tau1`; SNR fixed, because the split is relative to each level's own V_f |
| `slow_timescale_modal_variance` | τ1 = 1 … 100 yr | `tau1`, with the noise shares rescaled by V_f(ref) / V_f(τ1) so that the absolute variances stay fixed |
| `obs_noise` | v = 0.1 (ref) … 3 | the measurement-noise share `split[4]`, internal shares fixed, so the SNR falls (his `__main__` levels) |
| `forcing_rotation` | b_rot = 0 (ref, flat B), 0.25, 0.5, 0.75, 1 (B = w1) | `b_rot`, the fraction of the angle by which B is rotated from the flat pattern to w1 |
| `joint_snr_timescale` | 7 × 7 grid | `tau1` and the `split` scale together (phase diagram) |

A variant of the whole system is a run, not a study. For example, `dip1` is `--set depth=1`.

## Outputs

`results/<run>/` holds:
- `config.json`, which fully determines the run's numbers;
- `provenance.json`, with the seeds, conventions and package versions;
- `ablations.csv`, with one row per (study, level, realization, method), the scores
  (`forced_rel_rmse`, `forced_corr`, `slow_tau_yr`, `slow_corr`, `pair_corr`, …) and `snr`/`tau1_yr`;
- `slow_modes.npz`, the fitted slow-mode patterns plus the truth per level.

## Reproducibility

All randomness is seeded:
- `w_seed` sets the noise-mode patterns.
- `eig_seed` sets the noise-mode eigenvalues.
- `noise_seed` sets the `rng` of `generate_ensemble`.

Every level draws from the same `noise_seed`. So levels with the same spin-up length share their noise draws,
which is every study except the τ1 ones. A run is a pure function of its `config.json`, given the same environment.

## Notes

- The slow mode's variance over the 165-yr record, about the record mean, is about 0.8 of its stationary value
  at τ1 = 20 yr. That is why the empirical SNR in `ensemble_diagnostics.pdf` sits slightly above the theoretical one.
- `b_rot = 1` is the reviewer's case, B = w1: then `W_inv @ B = [1, 0, …, 0]`, so the forcing drives the slow mode alone.

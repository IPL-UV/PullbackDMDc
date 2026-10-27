import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
base_path = os.getenv("PBDMDC_DATA_ROOT", "/data/databases/dmdc-variants/mmlea_v2/")
artifact_root = os.getenv("PBDMDC_ARTIFACT_ROOT", "/data/users/nate/DMDcForcedResponse")
pdf_root = os.getenv("PBDMDC_PDF_ROOT", os.path.join(REPO_ROOT, "evaluation_results"))


def _parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def requested_lags(default_lags):
    raw_lags = os.getenv("PBDMDC_LAGS")
    if not raw_lags:
        return list(default_lags)
    return _parse_int_list(raw_lags)


def requested_year_ranges(default_year_ranges):
    raw_ranges = os.getenv("PBDMDC_YEAR_RANGES")
    if not raw_ranges:
        return list(default_year_ranges)

    parsed_ranges = []
    for item in raw_ranges.split(","):
        item = item.strip()
        if not item:
            continue
        start_year, end_year = item.split("-")
        parsed_ranges.append((int(start_year), int(end_year)))
    return parsed_ranges


def requested_methods(default_methods):
    raw_methods = os.getenv("PBDMDC_METHODS")
    if not raw_methods:
        return list(default_methods)
    return [m.strip() for m in raw_methods.split(",") if m.strip()]


def requested_clim_vars(default_vars):
    raw = os.getenv("PBDMDC_CLIM_VARS")
    if not raw:
        return list(default_vars)
    return [v.strip() for v in raw.split(",") if v.strip()]


def predictions_dir(year_suffix, lag_suffix):
    return os.path.join(artifact_root, f'models{year_suffix}{lag_suffix}')


n_eofs = {'tas': 20, 'psl': 200, 'tas_masked': 20, 'tas_ocean' : 20}
clim_vars = ['psl', 'tas', 'tas_masked','tas_ocean']
method_names = ['RegGMST', 'LIM', 'LIM-opt', 'LR', 'PullbackDMDc', 'PullbackDMDc-2d', 'PullbackDMDc-3d']
esms = ['20CRv3', 'CESM2', 'MPI-ESM1-2-LR', 'CanESM5', 'MIROC6'] #['20CRv3', 'CanESM5', 'MIROC6', 'CESM2', 'MPI-ESM1-2-LR']
colors = {'LR':'tab:red',#gray',
          'RegGMST':'tab:pink',#'RegGMST':'tab:orange',
          'LIM':'tab:orange',
          'LIM-opt':'tab:olive',
          'PullbackDMDc':'tab:purple',
          'PullbackDMDc-3d':'tab:blue',#'tab:purple',
          'PullbackDMDc-2d':'tab:cyan'}#'tab:pink'}
method_markers = {'LR':'s', 'RegGMST':'^', 'LIM':'o', 'LIM-opt':'D', 'PullbackDMDc':'v', 'PullbackDMDc-3d':'>', 'PullbackDMDc-2d':'<'}
year_ranges = [(1850,2014),(1950,2014)]
lags =  [12, 3, 6, 1]
optlag = 60
transition_time = 1200-12

DEFAULT_LAG = 3
STORED_EOFS = 200  # number of EOFs written by data_preparation/compute_eofs.py by default


def year_suffix_for(year_range):
    year_range = tuple(int(y) for y in year_range)
    if year_range == (1850, 2014):
        return ''
    if year_range == (1950, 2014):
        return '_tier1'
    raise ValueError(f'Unsupported year range: {year_range}')


def lag_suffix_for(lag):
    lag = int(lag)
    return '' if lag == DEFAULT_LAG else f'_lag{lag}'


def truncation_override():
    raw = os.getenv("PBDMDC_TRUNCATION")
    return int(raw) if raw else None


def truncation_for(var):
    override = truncation_override()
    return override if override is not None else n_eofs[var]


def sweep_suffix():
    # Present iff a truncation override or a run tag is set, even if the
    # truncation equals the default, so a sweep can never resolve to a main directory.
    override = truncation_override()
    suffix = '' if override is None else f'_n{override}'
    tag = os.getenv("PBDMDC_RUN_TAG", "").strip()
    if tag:
        suffix += f'_{tag}'
    return suffix


def guard_main_results_write(run_suffix, default_methods):
    """Refuse to overwrite main-run JSON/pickles with a partial method list."""
    raw = os.getenv("PBDMDC_METHODS")
    if run_suffix == '' and raw and set(requested_methods(default_methods)) != set(default_methods):
        raise RuntimeError('PBDMDC_METHODS subset would overwrite main results; set PBDMDC_RUN_TAG.')


def run_suffixes(lag, year_range):
    """Return (year_suffix, run_suffix) with run_suffix = lag suffix + sweep suffix."""
    return year_suffix_for(year_range), lag_suffix_for(lag) + sweep_suffix()


def results_dir(year_suffix, run_suffix):
    return os.path.join(artifact_root, 'evaluation_results', f'results{year_suffix}{run_suffix}')


def eofs_dir(year_suffix, truncation=None):
    if truncation is not None and int(truncation) > STORED_EOFS:
        store = int(os.getenv("PBDMDC_EOF_STORE", "400"))
        return os.path.join(artifact_root, f'eofs{year_suffix}_n{store}')
    return os.path.join(artifact_root, f'eofs{year_suffix}')

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
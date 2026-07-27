import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from utils.params import artifact_root


def infer_output_filename(filename, input_var, output_var):
    if filename.startswith(f"{input_var}_"):
        return filename.replace(f"{input_var}_", f"{output_var}_", 1)
    root, ext = os.path.splitext(filename)
    return f"{root}_{output_var}{ext}"


def _resolve_data_var(ds, preferred_name):
    if preferred_name in ds.data_vars:
        return preferred_name
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise KeyError(
        f"Variable '{preferred_name}' not found. Available data variables: {list(ds.data_vars)}"
    )


def _find_lat_lon_names(da):
    lat_name = "lat" if "lat" in da.dims else "latitude"
    lon_name = "lon" if "lon" in da.dims else "longitude"
    if lat_name not in da.dims or lon_name not in da.dims:
        raise KeyError(
            "Could not infer lat/lon dimensions from mask data. "
            f"Found dims: {da.dims}"
        )
    return lat_name, lon_name


def _reduce_to_spatial(da):
    lat_name, lon_name = _find_lat_lon_names(da)
    extra_dims = [dim for dim in da.dims if dim not in (lat_name, lon_name)]
    if len(extra_dims) == 0:
        return da
    return da.all(dim=extra_dims)


def _rename_spatial_dims(da, target_lat_name, target_lon_name):
    src_lat_name, src_lon_name = _find_lat_lon_names(da)
    rename_map = {}
    if src_lat_name != target_lat_name:
        rename_map[src_lat_name] = target_lat_name
    if src_lon_name != target_lon_name:
        rename_map[src_lon_name] = target_lon_name
    if len(rename_map) == 0:
        return da
    return da.rename(rename_map)


def load_masks_from_mask_data(
    mask_data_dir,
    tos_mask_filename,
    tos_mask_var,
    mask_reference_year,
):
    tos_mask_path = os.path.join(mask_data_dir, tos_mask_filename)

    if not os.path.exists(tos_mask_path):
        raise FileNotFoundError(f"TOS mask file not found: {tos_mask_path}")

    ds_tos = xr.open_dataset(tos_mask_path)

    resolved_tos_var = _resolve_data_var(ds_tos, tos_mask_var)

    tos_all = ds_tos[resolved_tos_var]
    if "time" in tos_all.dims:
        if "time" not in ds_tos.coords:
            raise KeyError("TOS has a time dimension but no 'time' coordinate.")

        year_mask = ds_tos["time"].dt.year == mask_reference_year
        tos_ref_year = tos_all.where(year_mask, drop=True)
        if tos_ref_year.sizes.get("time", 0) == 0:
            raise ValueError(
                f"No tos time steps found for reference year {mask_reference_year}."
            )

        tos_reference = tos_ref_year.isel(time=0)
    else:
        tos_reference = tos_all

    land_mask = _reduce_to_spatial(tos_reference.isnull()).load()

    ds_tos.close()

    return (
        land_mask,
        tos_mask_path,
        resolved_tos_var,
    )


def make_plot(da, title, out_file):
    sample = da.isel(time=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    sample.plot(ax=ax, cmap="coolwarm", robust=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


if __name__ == '__main__':

    base_path = "/data/databases/dmdc-variants/mmlea_v2"
    output_base_path = artifact_root
    input_var = "tas"
    output_var = "tas_ocean"
    source_subdir = "tas"
    target_subdir = "tas_ocean"
    mask_data_dir = os.path.join(base_path, "mask_data")
    tos_mask_filename = "tos_Omon_MPI-ESM1-2-LR_historical_r5i1p1f1_g025_185001-201412.nc"
    tos_mask_var = "tos"
    mask_reference_year = 1850
    latitude_max = 65.0
    latitude_min = -40.0
    overwrite = False
    seed = 42

    source_root = os.path.join(base_path, "ensembles", source_subdir)
    target_root = os.path.join(output_base_path, "ensembles", target_subdir)
    proof_plot = os.path.join(os.path.dirname(__file__), "tas_ocean_random_member.png")

    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"Source directory does not exist: {source_root}")

    input_files = []
    for root, _, files in os.walk(source_root):
        for file in sorted(files):
            if file.endswith(".nc") or file.endswith(".nc4"):
                input_files.append(os.path.join(root, file))

    if len(input_files) == 0:
        raise FileNotFoundError(f"No .nc/.nc4 files found in {source_root}")

    (
        land_mask_base,
        tos_mask_path,
        resolved_tos_var,
    ) = load_masks_from_mask_data(
        mask_data_dir=mask_data_dir,
        tos_mask_filename=tos_mask_filename,
        tos_mask_var=tos_mask_var,
        mask_reference_year=mask_reference_year,
    )

    print(
        "Loaded mask inputs: "
        f"TOS={tos_mask_path} (var={resolved_tos_var}), "
        f"static mask year={mask_reference_year}, "
        f"latitude max cutoff={latitude_max}",
        f"latitude min cutoff={latitude_min}"
    )

    rng = np.random.default_rng(seed)
    random_idx = int(rng.integers(0, len(input_files)))

    print(f"Found {len(input_files)} files under {source_root}")
    for i, input_file in enumerate(input_files):
        rel_parent = os.path.relpath(os.path.dirname(input_file), source_root)
        output_name = infer_output_filename(os.path.basename(input_file), input_var, output_var)
        output_file = os.path.join(target_root, rel_parent, output_name)

        # if os.path.exists(output_file) and not overwrite:
        #     print(f"[{i+1}/{len(input_files)}] Skipping existing: {output_file}")
        #     continue

        print(f"[{i+1}/{len(input_files)}] Writing: {output_file}")
        ds = xr.open_dataset(input_file)
        if input_var not in ds:
            ds.close()
            raise KeyError(f"Variable '{input_var}' not found in {input_file}")

        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"

        land_mask = _rename_spatial_dims(land_mask_base, lat_name, lon_name)

        land_mask_aligned, _ = xr.align(land_mask, ds[input_var], join="right")
        latitude_mask = (ds[lat_name] >= latitude_min) & (ds[lat_name] <= latitude_max)

        ocean_mask = ~land_mask_aligned
        combined_mask = ocean_mask & latitude_mask

        tas_ocean = ds[input_var].where(combined_mask)
        tas_ocean.attrs = dict(ds[input_var].attrs)
        tas_ocean.name = output_var

        ds_out = ds.drop_vars(input_var)
        ds_out[output_var] = tas_ocean

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        ds_out.to_netcdf(output_file, mode="w")

        if i == random_idx:
            make_plot(
                ds_out[output_var],
                f"Random member masked field ({output_var})\n{os.path.basename(input_file)}",
                proof_plot,
            )

        ds_out.close()
        ds.close()

    print(f"Done creating tas_ocean files. Proof plot: {proof_plot}")
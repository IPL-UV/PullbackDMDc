import xcdat
import xarray as xr
import os
import glob
from utils.data_utils import load_data
import numpy as np
from utils.params import artifact_root


def compute_mean(clim_var, model, base_path="/data/databases/dmdc-variants/mmlea_v2/",start_year=1850, end_year=2014):
    input_dir = os.path.join(base_path, f"ensembles/{clim_var}/{model}")   # change this


    # ---- LOAD FILES ----
    listdir = os.listdir(input_dir)
    n_members = 0
    for i, file in enumerate(listdir, start=1):
        
        if 'historical' not in file:
            continue
        if model == 'CESM2' and model_more not in file:
            continue

        file_path = os.path.join(input_dir, file)

        if n_members == 0:
            anomalies, pars = load_data(file_path, clim_var,start_year=start_year, end_year=end_year)
        else:
            anomalies += load_data(file_path, clim_var,start_year=start_year, end_year=end_year)[0]

        n_members+=1

    anomalies = anomalies/n_members

    anomalies_3d = np.full((pars['t'], pars['nlat'], pars['nlon']), np.nan)

    anomalies_3d[:,np.logical_not(pars['nan_mask'])] = anomalies

    ds_mean = xr.Dataset(
                            {
                                clim_var: (("time", "lat", "lon"), anomalies_3d),
                            },
                            coords={
                                "time": pars['time'],
                                "lat": pars['lat'],
                                "lon": pars['lon']
                            },
                            attrs={
                                "title": f"Forced Response Estimate {model}"
                            }
                        )

    return ds_mean

    



if __name__ == '__main__':

    # ---- SETTINGS ----
    clim_vars = ['tas_ocean','psl', 'tas'] #                              
    models = ['20CRv3', 'CESM2', 'CanESM5', 'MIROC6', 'MPI-ESM1-2-LR'] #"CESM2"#"MIROC6"#"MPI-ESM" #    
    base_path = "/data/databases/dmdc-variants/mmlea_v2/"             
    output_base_path = artifact_root

    for clim_var in clim_vars:
        for model in models:
            for start_year, end_year, output_subdir in [(1850, 2014,f"means/{clim_var}/{model}"),
                                                        (1950, 2014,f"means_tier1/{clim_var}/{model}")]:
                output_dir = os.path.join(output_base_path, output_subdir)               

                if model == 'CESM2':
                    model_more = "_cmip6"
                else:
                    model_more = ""

                ds_mean = compute_mean(clim_var, model, base_path=base_path,start_year=start_year, end_year=end_year)


                # ---- SAVE ----
                os.makedirs(output_dir,exist_ok=True)            
                out_file = os.path.join(output_dir, f"{clim_var}_Amon_{model}{model_more}_ensemblemean.nc")
                ds_mean.to_netcdf(out_file, mode="w")

                print(f"Saved ensemble mean to {out_file}")
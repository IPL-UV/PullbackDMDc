import os
from subprocess import Popen

# The script takes the downloaded data and regrids it to the same grid as the models, and also selects the time interval 1850-2014. 
# The main variable is renamed to match the model variable name.
# The commands in this scipt call the package 'cdo' v. 2.0.4
# It can be installed in conda: conda install cdo

base_path = "/data/databases/dmdc-variants/mmlea_v2/"



# Save the target grid
target_grid_file=os.path.join(base_path,'cdo_grid_2.5x2.5.txt')
target_grid_example_data=os.path.join(base_path,'ensembles','tas','CanESM5','tas_Amon_CanESM5_historical_r1i1p2f1_g025_185001-201412.nc')
Popen(f'cdo -griddes {target_grid_example_data} > {target_grid_file}',shell=True).communicate()

# Source: https://www.psl.noaa.gov/data/gridded/data.20thC_ReanV3.html
# Air temperature 2m, ensemble mean, monthly, from 1806-01 to 2015-12 (2520 time points)
# Mean sea level pressure, ensemble mean, monthly, from 1806-01 to 2015-12 (2520 time points)
# https://downloads.psl.noaa.gov/Datasets/20thC_ReanV3/Monthlies/miscSI-MO/prmsl.mon.mean.nc

for var, obs_var_name, obs_var_file in [('psl','prmsl','prmsl.mon.mean.nc'),('tas','air','air.2m.mon.mean.nc')]:
    model='20CRv3'
    input_dir=os.path.join(base_path,'obs_raw')
    output_dir = os.path.join(base_path, f"ensembles/{var}/{model}")
    os.makedirs(output_dir, exist_ok=True)

    input_file=os.path.join(input_dir,obs_var_file)
    output_file = os.path.join(output_dir, f"{var}_Amon_{model}_historical_185001-201412.nc")

    # remap to a new grid
    # select the time interval 1850-2014
    # rename the main variable

    command=f'cdo -b 64 -O -chname,{obs_var_name},{var} -seldate,1850-01-01,2014-12-31 -remapcon,{target_grid_file} {input_file} {output_file}'
    print(command)
    Popen(command,shell=True).communicate()




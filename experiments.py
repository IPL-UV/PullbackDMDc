import numpy as np
import os
from utils.data_utils import load_data, load_forcings, load_forcings_pullback, load_forcings_2d, load_forcings_3d
from utils.pullback_dmdc import PullbackDMDc
from utils.lim import LIM
from utils.lim_opt import LIM_opt
from utils.lr import LR
from utils.reg_gmst import RegGMST
from utils.params import n_eofs, esms, method_names, base_path, artifact_root, transition_time, lags, year_ranges, requested_lags, requested_year_ranges, requested_methods, predictions_dir
import pickle as pkl
import time


if __name__ == '__main__':

    start = time.time()

    clim_vars = ['tas_ocean', 'psl', 'tas']

    for lag in requested_lags(lags):
        for year_range in requested_year_ranges(year_ranges):
            start_year, end_year = year_range

            if lag == 3:
                lag_suffix = ''
            else:
                lag_suffix = f'_lag{lag}'

            if (start_year, end_year) == (1950, 2014):
                year_suffix = '_tier1'
            elif (start_year, end_year) == (1850, 2014):
                year_suffix = ''
            else:
                print('year range not recognized')

            method_base_path = predictions_dir(year_suffix, lag_suffix)

            data_base_path = os.path.join(base_path,'ensembles')

            eofs_base_path = os.path.join(artifact_root, f'eofs{year_suffix}')

            for var_init in clim_vars:
                truncation = n_eofs[var_init]

                datasets = {}
                precomputed_eofs={}

                print('Loading data ... ')
                for model in esms:
                    data_path = os.path.join(data_base_path, var_init, model)

                    # Loop through each file in the input directory
                    listdir = os.listdir(data_path)
                    for i, file in enumerate(listdir, start=1):

                        # print('File {}/{}'.format(i,len(listdir)), end='\r')
                        if 'historical' not in file:
                            continue
                        if 'CESM2' in file and 'cmip6' not in file:
                            continue
                        file_path = os.path.join(data_path, file)

                        # load_data
                        data, pars = load_data(file_path, var_init, start_year=start_year, end_year=end_year,empty=True)
                        data_name=pars['data_name']
                        
                        datasets[data_name] = data
                        # load precomputed eofs
                        eofs_path=os.path.join(eofs_base_path, f"{data_name}.pkl") 
                        with open(eofs_path, "rb") as f: 
                            precomputed_eofs[data_name]=pkl.load(f)


                print('Starting experiment ... ')
                for method_name in requested_methods(method_names):
                    print(f'Running method: {method_name}')
                    for data_name, data in datasets.items():
                        print(f' on dataset: {data_name}',flush=True)

                        eof_dict=precomputed_eofs[data_name]

                        if method_name in ['PullbackDMDc']:
                            short_forcings, long_forcings = load_forcings_pullback(transition_time, start_year=start_year, end_year=end_year)
                        elif method_name in ['PullbackDMDc-2d']:
                            short_forcings, long_forcings = load_forcings_2d(transition_time, start_year=start_year, end_year=end_year)
                        elif method_name in ['PullbackDMDc-3d']:
                            short_forcings, long_forcings = load_forcings_3d(transition_time, start_year=start_year, end_year=end_year)
                        else:
                            forcings_centered_scaled = load_forcings(mean_center = True, rescale = True, start_year=start_year, end_year=end_year)

                        var_name = data_name.split('_')[0]
                        
                        if method_name in ['PullbackDMDc', 'PullbackDMDc-2d', 'PullbackDMDc-3d']:
                            my_dmdc = PullbackDMDc(truncation=truncation, lag=lag, transition_time=transition_time)
                            my_dmdc.fit(data, short_forcings, long_forcings, precomputed_eofs=eof_dict)
                        elif method_name == 'LIM':
                            my_dmdc = LIM(truncation=truncation, lag=lag)
                            my_dmdc.fit(data,precomputed_eofs=eof_dict)
                        elif method_name == 'LIM-opt':
                            my_dmdc = LIM_opt(truncation=truncation, lag=lag)
                            my_dmdc.fit(data,precomputed_eofs=eof_dict)
                        elif method_name == 'LR':
                            my_dmdc = LR(truncation)
                            my_dmdc.fit(data, forcings_centered_scaled,precomputed_eofs=eof_dict)
                        elif method_name == 'RegGMST':
                            # Load a second (tas) dataset from **the same ESM run**
                            tas_file_path=os.path.join(data_base_path, data_name.replace(var_init, 'tas'))+'.nc'
                            data_tas, tas_pars = load_data(tas_file_path, 'tas', start_year=start_year, end_year=end_year)                 
                            my_dmdc = RegGMST(truncation)
                            my_dmdc.fit(data, data_tas, tas_pars['resc_coef_flat'],precomputed_eofs=eof_dict) 
                            # A bit counterintuitive that the data_tas is already multiplied by sqrt(cos(latitude)) once, but that's how it is used in the other methods                
                        else:
                            print(f'method_name: {method_name} not recognized')
                            continue

                        method_path = os.path.join(method_base_path, f"{data_name}_{method_name}.pkl")
                        os.makedirs(os.path.dirname(method_path), exist_ok=True)
                        with open(method_path, "wb") as f:
                            pkl.dump(my_dmdc, f)

                end = time.time()
                print(f'total time: {end - start}')
import numpy as np
import os
from utils.data_utils import load_data
import pickle as pkl
import time
from eofs.standard import Eof
from utils.params import artifact_root

def run_eofs(data: np.array, truncation: int=20):
    data_mean = data.mean(axis = 0)
    data_centered = data - data_mean

    eofs_xr = Eof(data_centered, center=False, ddof=1)
    eofs = eofs_xr.eofs(neofs=truncation)
    pcs = eofs_xr.pcs(npcs=truncation)
    svals = np.sqrt(eofs_xr.eigenvalues(neigs=truncation))
    
    return data_mean, eofs, pcs, svals

if __name__ == '__main__':

    start = time.time()

    model_names = ['20CRv3', 'CanESM5', 'CESM2', 'MIROC6', 'MPI-ESM1-2-LR'] #, 
    base_path = "/data/databases/dmdc-variants/mmlea_v2/"
    output_base_path = artifact_root
    clim_vars = ['tas_ocean','psl', 'tas']

    truncation = 200

    for year_range in [(1950,2014),(1850,2014)]:
        start_year, end_year = year_range        

        if (start_year, end_year) == (1950, 2014):
            year_suffix = '_tier1'
        elif (start_year, end_year) == (1850, 2014):
            year_suffix = ''
        else:
            print('year range not recognized')


        data_base_path = os.path.join(base_path,'ensembles')


        eofs_base_path = os.path.join(output_base_path,f'eofs{year_suffix}')

        print('Truncating data ... ',flush=True)
        for var_init in clim_vars:  
            for model in model_names:
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
                    print(file_path,flush=True)

                    # load_data
                    data, pars = load_data(file_path, var_init, start_year=start_year, end_year=end_year)
                    data_name=pars['data_name']

                    data_mean, eofs, pcs, svals = run_eofs(data,truncation)
                    # saving PCs and the original names to be able to reconstruct anything else from the original file
                    truncated_data={'data_mean':data_mean,'eofs':eofs,'pcs':pcs,'svals':svals,
                                    'file_path':file_path,'var_init':var_init, 'data_pars':pars}
                    eofs_path=os.path.join(eofs_base_path, f"{data_name}.pkl") 
                    # Create missing directories, then write into file
                    os.makedirs(os.path.dirname(eofs_path), exist_ok=True)
                    with open(eofs_path, "wb") as f: pkl.dump(truncated_data, f)

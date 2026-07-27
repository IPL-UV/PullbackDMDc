import xarray as xr
import numpy as np
import os
import xcdat
import datetime
import pickle
import pandas as pd


def load_forcings(mean_center: bool = True, rescale: bool = False, start_year: int = 1850, end_year: int = 2014):
    # Load Forcings
    forcings_df = pd.read_csv('interpolatedTotalForcing.csv')

    forcings_df['time'] = pd.to_datetime(forcings_df['time']) 

    start = pd.Timestamp(f'{start_year}-01-01')
    end = pd.Timestamp(f'{end_year}-12-31')
    forcings_df = forcings_df[(forcings_df['time'] >= start) & (forcings_df['time'] <= end)].reset_index(drop=True)
    forcings  = np.array(forcings_df['total_forcing'])

    if rescale and mean_center:
        forcings  = (forcings-np.mean(forcings))/np.std(forcings)
    elif mean_center:
        forcings  = forcings-np.mean(forcings)
    elif rescale:
        forcings  = forcings/np.std(forcings)
    
    return forcings

def load_forcings_pullback(transition_time=1200, start_year: int = 1850, end_year: int = 2014):
    forcings_df = pd.read_csv('interpolatedTotalForcing.csv')

    forcings_df['time'] = pd.to_datetime(forcings_df['time']) 
    # Clip to Jan 1850 - Dec 2014
    start = pd.Timestamp(f'{start_year}-01-01')
    end = pd.Timestamp(f'{end_year}-12-31')
    short_forcings_df = forcings_df[(forcings_df['time'] >= start) & (forcings_df['time'] <= end)].reset_index(drop=True)
    short_forcings = np.array(short_forcings_df['total_forcing'])
    
    start = start - pd.DateOffset(months=transition_time)
    end = pd.Timestamp(f'{end_year}-12-31')
    long_forcings_df = forcings_df[(forcings_df['time'] >= start) & (forcings_df['time'] <= end)].reset_index(drop=True)
    long_forcings = np.array(long_forcings_df['total_forcing'])

    mean_short = np.mean(short_forcings)
    std_short = np.std(short_forcings)
        
    return (short_forcings[:,None]-mean_short)/std_short, (long_forcings[:,None]-mean_short)/std_short

def load_forcings_2d(transition_time=1200, start_year: int = 1850, end_year: int = 2014):
    forcings_df = pd.read_csv('interpolatedAllForcing.csv')

    forcings_df['time'] = pd.to_datetime(forcings_df['time']) 
    # Clip to Jan 1850 - Dec 2014
    start = pd.Timestamp(f'{start_year}-01-01')
    end = pd.Timestamp(f'{end_year}-12-31')
    short_forcings_df = forcings_df[(forcings_df['time'] >= start) & (forcings_df['time'] <= end)].reset_index(drop=True)
    short_forcings = np.stack([
                               #np.array(short_forcings_df['total_anthropogenic']+short_forcings_df['solar']),
                               np.array(short_forcings_df['co2']),
                               np.array(short_forcings_df['volcanic'])],axis=-1)
    
    start = start - pd.DateOffset(months=transition_time)
    end = pd.Timestamp(f'{end_year}-12-31')
    long_forcings_df = forcings_df[(forcings_df['time'] >= start) & (forcings_df['time'] <= end)].reset_index(drop=True)
    long_forcings = np.stack([
                               #np.array(long_forcings_df['total_anthropogenic']+short_forcings_df['solar']),
                               np.array(long_forcings_df['co2']),
                               np.array(long_forcings_df['volcanic'])],axis=-1)

    mean_short = np.mean(short_forcings,axis=0)
    std_short = np.std(short_forcings,axis=0)
        
    return (short_forcings-mean_short)/std_short, (long_forcings-mean_short)/std_short

def load_forcings_3d(transition_time=1200, start_year: int = 1850, end_year: int = 2014):
    forcings_df = pd.read_csv('interpolatedAllForcing.csv')

    forcings_df['time'] = pd.to_datetime(forcings_df['time']) 
    # Clip to Jan 1850 - Dec 2014
    start = pd.Timestamp(f'{start_year}-01-01')
    end = pd.Timestamp(f'{end_year}-12-31')
    short_forcings_df = forcings_df[(forcings_df['time'] >= start) & (forcings_df['time'] <= end)].reset_index(drop=True)
    short_forcings = np.stack([
                               np.array(short_forcings_df['co2']+short_forcings_df['nonco2_wmghg']),
                               np.array(short_forcings_df['aerosol']),
                               np.array(short_forcings_df['volcanic'])],axis=-1)
    
    start = start - pd.DateOffset(months=transition_time)
    end = pd.Timestamp(f'{end_year}-12-31')
    long_forcings_df = forcings_df[(forcings_df['time'] >= start) & (forcings_df['time'] <= end)].reset_index(drop=True)
    long_forcings = np.stack([
                               np.array(long_forcings_df['co2']+long_forcings_df['nonco2_wmghg']),
                               np.array(long_forcings_df['aerosol']),
                               np.array(long_forcings_df['volcanic'])],axis=-1)

    mean_short = np.mean(short_forcings,axis=0)
    std_short = np.std(short_forcings,axis=0)
        
    return (short_forcings-mean_short)/std_short, (long_forcings-mean_short)/std_short

def load_data(file_path, var, start_year: int = 1850, end_year: int = 2014, empty=False):
    if empty:
        # don't read the file, just return the data_name info
        split_fpath = file_path.split('/')
        data_name = os.path.join(split_fpath[-3],split_fpath[-2],split_fpath[-1][:-3])   
        data=None
        data_parameters={'data_name':data_name}
        return data, data_parameters 
    
    ds = xcdat.open_dataset(file_path)

    # Get time values (cftime objects)
    ds_times = ds.time.values
    # Build mask for Jan 1880 - Dec 2014 (inclusive)
    mask = [
        ((t.year > start_year) or (t.year == start_year and t.month >= 1)) and
        ((t.year < end_year) or (t.year == end_year and t.month <= 12))
        for t in ds_times
    ]
    mask = np.array(mask)
    if mask.sum() == 0:
        print("WARNING: No time steps found in the requested range for", file_path)
    ds = ds.isel(time=mask)

        
    # Extract numpy arrays for data and lats
    x=ds[var].to_masked_array()
    lat=ds['lat'].to_numpy()

    # Update data mask to exclude nans and big numbers for sure
    good_mask=np.logical_and(np.logical_not(x.mask),np.abs(x.data)<=1e+16)
    x=np.ma.masked_array(x.data,mask=np.logical_not(good_mask))

    # Geographical rescaling coefficient
    resc_coef=np.sqrt(np.cos(lat[:,None]/180.*np.pi))
    x=x*resc_coef

    # Computing anomalies
    t, nlat, nlon = x.data.shape
    climatology=np.concatenate(np.ma.stack(np.split(x,t//12,axis=0),axis=0).mean(axis=0,keepdims=True).repeat(t//12,axis=0),axis=0)
    anomalies=x-climatology

    # Remove pixels without data
    nan_mask = anomalies.mask
    nan_mask=np.any(nan_mask, axis=0)
    data = anomalies.data[:,np.logical_not(nan_mask)]

    split_fpath = file_path.split('/')
    data_name = os.path.join(split_fpath[-3],split_fpath[-2],split_fpath[-1][:-3])
    # grid+flatten rescaling coefficient
    resc_coef_flat=np.repeat(resc_coef,nlon,axis=-1)[np.logical_not(nan_mask)]

    data_parameters={'time':ds.time,'lat':ds.lat,'lon':ds.lon,
                    't':t,'nlat':nlat,'nlon':nlon,'data_name':data_name,
                    'nan_mask':nan_mask,
                    'resc_coef_flat':resc_coef_flat}
    return data, data_parameters
    #return data, ds, t, nlat, nlon, data_name

def load_truth(var, model, base_truth_dir, return_lats = False):
    """Load prediction and ground truth ensemble mean for a given variable/model/method."""

    # Load ground truth (ensemble mean)
    if model == 'CESM2':
        model_load = 'CESM2_cmip6'
    else:
        model_load = model
    truth_file = os.path.join(base_truth_dir, var, model, f"{var}_Amon_{model_load}_ensemblemean.nc")
    truth_ds = xr.open_dataset(truth_file)

    truth = truth_ds[var]

    if return_lats:
        return truth, truth_ds['lat'].values
    else:
        return truth

def load_prediction(var, model, method, base_pred_dir):
    """Load prediction for a given variable/model/method."""
    # Find all prediction files for this model/var
    pred_dir = os.path.join(base_pred_dir, var, model)
    preds = []
    for file in os.listdir(pred_dir):
        if f'{method}.pkl' == file.split('_')[-1]:
            f_path = os.path.join(pred_dir, file)
            with open(f_path, "rb") as f:
                fr_method = pickle.load(f)
            preds.append(fr_method.predict().data) #converting masked array -> array
    preds = np.dstack(preds)
    #time x space x member
    return preds

def load_truncated_prediction(var, model, method, base_pred_dir, return_data_names=False):
    """Load truncated prediction (list of pcs and list of eofs) for a given variable/model/method."""
    # Find all prediction files for this model/var
    pred_dir = os.path.join(base_pred_dir, var, model)
    preds_pcs = []
    preds_eofs = []
    data_names= []
    for file in os.listdir(pred_dir):
        if f'{method}.pkl' == file.split('_')[-1]:
            f_path = os.path.join(pred_dir, file)
            with open(f_path, "rb") as f:
                fr_method = pickle.load(f)              
            svals=np.sqrt((fr_method.eofs**2).sum(axis=-1))
            preds_pcs.append(fr_method.fr_pcs.real * svals)
            preds_eofs.append(fr_method.eofs / svals[:,None])
            # extracting data_name from the method filename...
            data_name=file[:-len(f'_{method}.pkl')]
            data_names.append(data_name)
    if return_data_names:
        return preds_pcs, preds_eofs, data_names
    return preds_pcs, preds_eofs

def load_truncated_predictions(var, model, methods, base_pred_dir):
    """Same as load_truncated_prediction but for the list of methods. Returns 2 dictionaries of lists, instead of 2 lists"""
    # Find all prediction files for this model/var
    pred_dir = os.path.join(base_pred_dir, var, model)
    preds_pcs = {method:[] for method in methods}
    preds_eofs = {method:[] for method in methods}
    method0=methods[0]
        
    for file in os.listdir(pred_dir):
        if f'{method0}.pkl' == file.split('_')[-1]:
            f_path0 = os.path.join(pred_dir, file)
            for method in methods:
                f_path=f_path0.replace(method0,method)
                with open(f_path, "rb") as f:
                    fr_method = pickle.load(f)            
                svals=np.sqrt((fr_method.eofs**2).sum(axis=-1))
                preds_pcs[method].append(fr_method.fr_pcs.real * svals)
                preds_eofs[method].append(fr_method.eofs / svals[:,None])            
    return preds_pcs, preds_eofs

def undo_lat_scaling(pattern, lat, nlon, nan_mask=None):
    resc_coef = np.sqrt(np.cos(np.deg2rad(lat)))[:, None]
    resc_3d = np.broadcast_to(resc_coef, (len(lat), nlon)).copy()
    pattern_3d = np.full((len(lat), nlon, pattern.shape[-1]), np.nan)
    if nan_mask is not None:
        # resc_3d[nan_mask] = np.nan
        pattern_3d[~nan_mask,:] = pattern
    else:
        pattern_3d = pattern.reshape(len(lat), nlon, pattern.shape[-1])

    return pattern_3d / resc_3d[:,:,None]
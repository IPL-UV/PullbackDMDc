import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr
import pandas as pd
from evaluation.utils.taylor_diagram_plot import TaylorDiagramPlot
import cftime
# from evaluation.taylor import TaylorDiagram


def _filter_inlier_coords(coords, fence_scale=1.5):
    if coords.shape[0] < 4:
        return coords

    q1 = np.percentile(coords, 25.0, axis=0)
    q3 = np.percentile(coords, 75.0, axis=0)
    iqr = q3 - q1
    lower = q1 - fence_scale * iqr
    upper = q3 + fence_scale * iqr
    mask = np.all((coords >= lower) & (coords <= upper), axis=1)

    if np.count_nonzero(mask) >= 3:
        return coords[mask]
    return coords


def spatial_rmse(preds, truth):
    """Spatial RMSE map (mean over time and members)."""
    # mean center across time
    # stack and compute rmse form stacked vars
    preds = preds.astype(np.float64)
    truth = truth.astype(np.float64)

    n_members=preds.shape[2]

    # stack and compute correlation between stacked vars
    truth_rep = np.repeat(truth[:,:,np.newaxis], axis = 2, repeats=n_members)

    # mean center across time
    preds_mc = preds - preds.mean(axis=0, keepdims=True)
    truth_mc = truth_rep - truth_rep.mean(axis=0, keepdims=True)

    # Do something with member_pred, e.g
    return np.sqrt(((preds_mc - truth_mc) ** 2).mean(axis=(0,2)))/np.sqrt(((truth_mc) ** 2).mean(axis=(0,2)))

def spatial_corr(preds, truth):
    """Spatial correlation map over time."""
    preds = preds.astype(np.float64)
    truth = truth.astype(np.float64)

    n_members=preds.shape[2]

    # stack and compute correlation between stacked vars
    truth_rep = np.repeat(truth[:,:,np.newaxis], axis = 2, repeats=n_members)

    # mean center across time
    preds_mc = preds - preds.mean(axis=0, keepdims=True)
    truth_mc = truth_rep - truth_rep.mean(axis=0, keepdims=True)
    
    # Numerator and denominator for correlation
    num = (preds_mc * truth_mc).sum(axis=(0,2))
    denom = np.sqrt((preds_mc**2).sum(axis=(0,2)) * (truth_mc**2).sum(axis=(0,2)))
    
    corr = num / denom
    
    return corr

def trend_map(x):
    """
    Compute linear trend at each (lat, lon), using all samples.
    
    If x has dims (time, member, lat, lon):
        members are treated as independent samples.
    
    If x has dims (time, lat, lon) (no member):
        a normal time-only trend is computed.
    """

    # Does the 'member' dimension exist?
    has_member = 'member' in x.dims

    # --- function applied to each grid cell ---
    def lintrend(ts):
        """
        ts is either:
          (time,)               -- no members
          (time, member)        -- with members
        """
        if ts.ndim == 1:
            # No members → normal regression
            T = ts.shape[0]
            y = ts
            t = np.arange(T)
        else:
            # With members → flatten (time, member)
            T, M = ts.shape
            y = ts.reshape(-1)                  # (T*M,)
            t = np.repeat(np.arange(T), M)      # (T*M,)

        # Fit linear trend (slope only)
        return np.polyfit(t, y, 1)[0]

    # --- choose core dims dynamically ---
    core_dims = [['time', 'member']] if has_member else [['time']]

    return xr.apply_ufunc(
        lintrend,
        x,
        input_core_dims=core_dims,
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
    )

def global_mean(data,lats):
    # area weighting coefficients
    sq=np.cos(np.pi*lats[:,None]/180.)
    return (data*sq).mean(axis=(-1,-2))/((data*0+1.)*sq).mean(axis=(-1,-2))

def custom_masked_array(da):
    masked_array=da.to_masked_array()
    #update mask to exclude nans and big numbers for sure
    good_mask=np.logical_and(np.logical_not(masked_array.mask),np.abs(masked_array.data)<=1e+16)
    masked_array=np.ma.masked_array(masked_array.data,mask=np.logical_not(good_mask))
    return masked_array

def global_mean(x, lats):
    """Global mean timeseries (mean over space)."""

    if isinstance(x, xr.DataArray):
        print('hi')
        x=x.to_masked_array()
    sq=np.cos(np.pi*lats[:,None]/180.)
    x=x/np.sqrt(sq) # Scale to original units, then compute weighted mean by definition
    return (x*sq).mean(axis=(-1,-2))/((x*0+1.)*sq).mean(axis=(-1,-2))

def global_mean_flat(data_flat, resc_coef_flat):
    x=data_flat/resc_coef_flat # Scale to original units, then compute weighted mean by definition
    sq=(resc_coef_flat**2)
    return (x*sq).mean(axis=(-1))/((x*0+1.)*sq).mean(axis=(-1))    


def smooth_truth(x, window_years: int = 3):
    """Apply centered rolling mean over time with a 3-year window (monthly data)."""

    if not isinstance(x, (xr.DataArray, xr.Dataset)):
        raise TypeError("smooth_truth expects an xarray DataArray or Dataset")

    window = int(window_years * 12)
    return x.rolling(time=window, center=True, min_periods=1).mean()

def nino34_index(x, lats):
    """Compute Nino3.4 index (mean over 5S-5N, 190E-240E, mean over members)."""
    coslat = np.cos(lats)
    coslat_arr = xr.DataArray(coslat, dims=['lat']).broadcast_like(x)
    x_rescaled = x*np.sqrt(coslat_arr)
    x_rescaled = x_rescaled.sel(lat=slice(-5, 5), lon=slice(190, 240))
    x_rescaled = x_rescaled.stack(z=('lat', 'lon'))
    nino = x_rescaled.mean(dim='z')/np.mean(coslat)
    return nino

def plot_spatial_map(data, title, cmap='RdBu_r', abs_max = 0):
    if abs_max == 0:
        abs_max = np.percentile(np.abs(data), 90)

    plt.figure(figsize=(8, 4))
    data.plot(cmap=cmap, vmin=-abs_max, vmax=abs_max)
    plt.title(title)

def coord_to_dtindex(coord):
    """Convert xarray/cftime time coord to pandas.DatetimeIndex, handling no-leap calendars by string conversion."""
    try:
        # Try pandas conversion
        return pd.DatetimeIndex(coord.values)
    except Exception:
        pass
    try:
        # Try xarray's .dt accessor
        return coord.dt.to_index()
    except Exception:
        pass
    try:
        # Try cftime to pandas datetime via string conversion (works for no-leap)
        if hasattr(coord.values[0], 'strftime'):
            date_strs = [t.strftime("%Y-%m-%d") for t in coord.values]
            return pd.to_datetime(date_strs)
    except Exception:
        pass
    # Fallback: return as string
    return [str(t) for t in coord.values]

def compute_taylor_std_range(td_list, percentage=95):
    std_values = [td['std_Y'] for td in td_list]
    ref_values = [td['std_X'] for td in td_list]

    min_std = 0
    max_std = max(np.percentile(std_values, percentage), max(ref_values))
    margin = 0.15  # 15% margin
    range_span = max_std - min_std

    # Add minimum range to avoid degenerate cases
    ref_scale = max(ref_values)
    if range_span < 0.01 * ref_scale:
        range_span = 0.01 * ref_scale
        max_std = max_std + range_span / 2

    return (min_std, max_std + margin * range_span + 0.2)

def compute_taylor_axis_extent(td_lists, padding=0.03, percentage=98):
    point_coords = []

    for td_list in td_lists:
        if not td_list:
            continue

        point_coords.append((td_list[0]['std_X'], 0.0))
        for sample in td_list:
            corr = float(np.clip(sample['corr'], -1.0, 1.0))
            theta = np.arccos(corr)
            std_value = float(sample['std_Y'])
            point_coords.append((std_value * np.cos(theta), std_value * np.sin(theta)))

    if not point_coords:
        return None

    coords = np.asarray(point_coords, dtype=float)
    coords = _filter_inlier_coords(coords)
    x_vals = coords[:, 0]
    y_vals = coords[:, 1]

    tail_percent = max(0.0, min(100.0, 100.0 - float(percentage))) / 2.0
    lower_percent = tail_percent
    upper_percent = 100.0 - tail_percent

    x_min = float(np.percentile(x_vals, lower_percent))
    x_max = float(np.percentile(x_vals, upper_percent))
    y_min = float(np.percentile(y_vals, lower_percent))
    y_max = float(np.percentile(y_vals, upper_percent))

    x_span = max(x_max - x_min, 1e-3)
    y_span = max(y_max - y_min, 1e-3)

    return (
        (x_min - padding * x_span, x_max + padding * x_span),
        (max(0.0, y_min - padding * y_span), y_max + padding * y_span),
    )

def add_one_custom_plot(td_list, fig, subplot=None, legend=True, percentage=95, std_range=None, legend_kwargs=None, axis_extent=None, ax=None, corr_label_pad=0.0, corr_label_fontsize=10.0):
    ref_point=td_list[0]['std_X']

    if std_range is None:
        std_range = compute_taylor_std_range(td_list, percentage=percentage)

    # add taylor diagram to plot with a shared absolute std range when provided
    taylor_fig = TaylorDiagramPlot(
        ref_point=ref_point,
        fig=fig,
        subplot=subplot,
        ax=ax,
        extend_angle=False,
        std_range=std_range,
        corr_label_pad=corr_label_pad,
        corr_label_fontsize=corr_label_fontsize,
    )
    taylor_fig.add_reference_point(ref_point,color="black",marker=".",markersize=20,linestyle="",label="Reference Data")
    taylor_fig.add_reference_line(ref_point, color="black", linestyle="--", label="_")
    taylor_fig.add_grid()
    taylor_fig.add_contours(ref_point, levels=14, cmap="viridis", linewidths=1.2)

    for sample in td_list:
        if sample['name'] is not None:
            taylor_fig.add_point(sample['std_Y'],sample['corr'],color=sample['color'],label=sample['name'],
                linestyle="",marker=sample['marker'],zorder=3, alpha=sample['alpha'], ms=sample['size'])
        else:
            taylor_fig.add_point(sample['std_Y'],sample['corr'],color=sample['color'],
                linestyle="",marker=sample['marker'],zorder=3,alpha=sample['alpha'], ms=sample['size'])                

    if axis_extent is None:
        taylor_fig.autoscale_to_points()
    else:
        taylor_fig.set_extent(axis_extent)

    if legend:
        legend_kwargs = legend_kwargs or {}
        taylor_fig.add_legend(fig, numpoints=1, prop=dict(size="medium"), **legend_kwargs)
    # Keep a strong reference so callback-based correlation tick updates
    # remain active when axis limits are modified after this function returns.
    taylor_fig.graph_axes._taylor_plot = taylor_fig
    taylor_fig.graph_axes._taylor_contours = taylor_fig.contours
    return taylor_fig.graph_axes

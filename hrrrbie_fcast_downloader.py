"""
Author: Lara Tobias-Tarsh (ltt0663@princeton.edu)
Created: 01/15/26

Script uses the HERBIE python API to download and format a variety
of datasets for forecasting with HLM. This is an early iteration which
will be cleaned up substantially in the future.

Requirements
-------------
python 3.14, herbie, numpy, xarray and backends.
It is probably best to just clone the environment I used to develop this. I have
tested it locally [and on a Tiger login node] and it seems to work fine.

```
conda env create -f met_downloader.yml
conda activate met_downloader
```

Running
-------
First, fork and clone the GitHub repo. I think having version control will be 
very useful here because the datasets will change formatting with time.

Next, skip to the section called GLOBALS, and edit the global parameters. For
hindcasting (i.e. you know the time of the event and you need to get archived 
forecasts at a given lead time), you MUST fill out the following parameters:

    - EVENT_TIME (the time at which the event you are trying to hindcast ENDED)
    - OUTPATH (where do you want to save out your downloaded data)
    - MODELS_TO_PULL (list of models you want to get, see README or the MODELS dict)

You can configure a set of dictionaries to download data from the available models.
This is currently extremely simple - you just put all the lead times (in hours before
the event you are hindcasting) in a list for that dataset. The script will check if
your request is theoretically valid and fail if it is not. You do not need to remove
a dataset from the dictionary if you do not want to download it, as the script will
just ignore it. I reccomend leaving it as a reminder of the available models.

I have set this up to just download data for grid cell over the Raritan
basin. There are some optional parameters that you can set to change these.
You can find these in the README and in the function docstring but for the 
most part there shouldn't be any reason to.


Troubleshooting
---------------
I find that these NOAA APIs can be a bit sensitive and unreliable, especially
with real time NOAA data and variable names. I wrote this in a somewhat clunky,
one function per dataset way so that we can easily troubleshoot individual datasets
without needing to understand how the code works or have much familiarity with 
operational and archived NWP data.

There are checks in place to ensure you cannot request a lead time that does not
exist for a specific model. I expect these will work 90% of the time. Sometimes
a model run might not be available, so try a different lead time as a first port 
of call or go to the NCEI NOMADS/RDA portals and see if you can find what you are
looking for. 

I will create some documentation with troubleshooting tips and try to add catch
alls for bugs in each dataset as I encounter them, but I expect that you will run 
into some weird variable/download errors at certain points. 

If you find an issue or encounter errors with a download, please open a GitHub issue, 
even if it just turns out to be a simple fix, in which case you can include the fix you used too. 
This way we can identify common errors associated with data stores and improve the code as a 
group without too much back and forth.
"""
import warnings
from datetime import datetime, timedelta
import os

import numpy as np
from herbie import FastHerbie
import xarray as xr
xr.set_options(use_new_combine_kwarg_defaults=False)

### IMPORTANT GLOBALS FOR DOWNLOAD ###
avail_models = ['hrrr', 'rap', 'gfs', 'nbm', 'rrfs', 'href', 'hiresw',
                'hrdps', 'rdps']


## come back and add the HWRF, HAFS global nest, SHIELD and TSHIELD, NAVGEM, NOGAPS, COAMPS, HRDPS

_ALLOWED_FORMATS = (
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H",
    "%Y-%m-%d %H%M",
    "%Y%m%d%H",
    "%Y%m%d %H",
)

def snap_to_prev_cycle(start_time):
    cycles = (0, 6, 12, 18)

    # If already on a cycle, no change
    if start_time.hour in cycles:
        snapped = start_time.replace(minute=0, second=0, microsecond=0)
    else:
        # cycles before current hour
        prev_cycles = [h for h in cycles if h <= start_time.hour]

        if prev_cycles:
            snapped_hour = max(prev_cycles)
            snapped = start_time.replace(
                hour=snapped_hour, minute=0, second=0, microsecond=0
            )
        else:
            # before 00 UTC → go to 18 UTC previous day
            snapped = (start_time - timedelta(days=1)).replace(
                hour=18, minute=0, second=0, microsecond=0
            )

    hour_diff = int((start_time - snapped).total_seconds() // 3600)
    return snapped, hour_diff

def _parse_time(time_str: str) -> datetime:
    """
    Safely parse time string using a whitelist of formats.
    Raises ValueError if none match.
    """
    for fmt in _ALLOWED_FORMATS:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"Unsupported datetime format: {time_str!r}. "
        f"Expected one of: {_ALLOWED_FORMATS}"
    )


def _calc_fcst_init(event_time, lead_time):
    """
    Function uses python datetime formatting to back 
    calculate the correct initialization time for some
    fixed lead time.

    NOTE: add option to forward calculate instead of back
    for operational forecasting

    Parameters
    -----------
    event_time : string
        the datetime string for when the event occurs
    lead_time : int
        the integer lead time for the forecast in hours
    
    Returns
    -------
    start_time : datetime.datetime
        the model initialization time for this event
    """
    #init_time = datetime.strptime(event_time, "%Y-%m-%d %H:%M")
    init_time = _parse_time(event_time)
    start_time = init_time - timedelta(hours=lead_time)

    return start_time

def to_360(lon):
    """
    forces longitudes to be 0-360 if they are -180 to 180

    Parameters
    -----------
    lon : arrayLike
        a list of longitudes to convert

    Returns
    --------
        : arrayLike
        the converted longitudes
    """
    return lon + 360 if lon < 0 else lon

def check_request(model, init_time, run_hours, dt, strict=True):
    """
    Function checks to see if the submitted request is valid.

    Parameters
    -----------
    model : string
        name of the model being requested
    init_time : datetime or string
        the initialization time for the model run
    run_hours : int
        how many hours to run the forecast forward
    dt : int
        interval data is being downloaded at
    """
    if isinstance(init_time, str):
        init_time = _parse_time(init_time)

    if model == 'GFS'.casefold():
        print("Validating GFS request...")
        # CHANGED: check run_hours instead of lead_time
        if run_hours > 240:
            raise ValueError(
                f"Maximum forecast hours ({run_hours}) are invalid. "
                "Ensure GFS run_hours is between 1 and 240.\n"
                f"See https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php for more"
            )
        
        
        if init_time.hour not in (0, 6, 12, 18):
            if strict:
                raise ValueError(
                    f"Initialization at {init_time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"is invalid for GFS, which must start at 0, 6, 12 or 18Z. "
                    f"See https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php for more"
                )
            else: 
                new_init, offset = snap_to_prev_cycle(init_time)
                warnings.warn(
                    f"GFS init time snapped to {new_init.strftime('%Y-%m-%d %H:%M:%S')}"
                )

        if dt % 3 != 0 and run_hours > 120:
            raise ValueError(
                f"Timestep of dt = {dt} is invalid for GFS data. "
                f"GFS has a 3 hour timestep, so ensure dt is a multiple of 3.\n"
                f"See https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php for more"
            )
        
    if model == 'HRRR'.casefold():
        print('Validating HRRR request...')
        # CHANGED: check run_hours against limits based on init hour
        if init_time.hour not in (0, 6, 12, 18) and run_hours > 18:
            raise ValueError(
                f"Maximum forecast hours ({run_hours}) are invalid for "
                f"{init_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                "Ensure HRRR run_hours is between 1 and 18 for intermediate init times.\n"
                f"HRRR analysis cycles hourly but is only available longer than 18 hours at "
                f"the 0, 6, 12 and 18Z cycles.\n"
                f"See https://rapidrefresh.noaa.gov/hrrr/ for more"
            )
        
        if run_hours > 48:
            raise ValueError(
                f"Maximum forecast hours ({run_hours}) are invalid. "
                f"Ensure HRRR run_hours is between 1 and 48.\n"
                f"See https://rapidrefresh.noaa.gov/hrrr/ for more"
            )

        if init_time.year < 2021 and run_hours > 36:
            raise ValueError(
                f"Maximum forecast hours ({run_hours}) are invalid for "
                f"{init_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                "Ensure run_hours is between 0 and 36 prior to 2021.\n"
                f"See https://rapidrefresh.noaa.gov/hrrr/ for more"
            )

def write_file(path, content):
    #os.makedirs(os.path.dirname(path), exist_ok=True)  # CHECK ON THIS
    with open(path, "w") as f:
        f.write(content)

def write_metadata(model_name, pr_ds, t2m_ds, outpath, 
                   pr_path, t2m_path, init_time, run_hours, dt):
    """
    Function writes a metadata file for use with the 
    yaml data needed for mapping indices to the links
    """
    if isinstance(init_time, str):
        init_time = _parse_time(init_time)
    
    valid_end = init_time + timedelta(hours=run_hours)
    
    template = f"""
    Metadata for {model_name} forecast
    Initialized: {init_time.strftime('%Y-%m-%d %H:%M')}
    Valid through: {valid_end.strftime('%Y-%m-%d %H:%M')}
    Run length: {run_hours}h

    path: "{outpath}"
    
    variables:
    - name: "pr"
      file: "{pr_path}"
      time_resolution: "{dt}h"
      dims: {", ".join(pr_ds['pr'].dims)}

    - name: "t2m"
      file: "{t2m_path}"
      time_resolution: "24h"
      dims: {", ".join(t2m_ds['t2m'].dims)}
"""
    init_str = init_time.strftime('%Y%m%d_%H%M')
    write_file(f"{outpath}/metadata-{model_name}_init{init_str}.yaml", template)

def download_hrrr(init_time, run_hours, outpath, dt=1, 
                  verbose=1, max_threads=50, crop_domain=True,
                  lat_max=41.024, lat_min=40.185, lon_max=-74.229, lon_min=-75.055):
    """
    Downloads and process a HRRR dataset for a fixed
    forecast period, checks units and processes to the
    required file format needed for regridding.

    Parameters
    ----------
    init_time : string or datetime
        initialization time for the model run
    run_hours : int
        how many hours to run the forecast forward
    dt : int
        timestep between forecasts
    outpath : string
        where to store file output
    crop_domain : bool
        should we crop to a regional domain?
    lat_max, lat_min : float
        the maximum and minimum latitude

    Returns
    -------
    apcp_df, t2m_df : xr.Dataset
    """
    # CHANGED: parse init_time and calculate valid times directly
    if isinstance(init_time, str):
        init_time = _parse_time(init_time)
    start_time = init_time
    valid_start = start_time + timedelta(hours=dt)
    valid_end = start_time + timedelta(hours=run_hours)
    
    if verbose:
        print(f"Downloading HRRR initialized at {start_time}")
        # CHANGED: print run_hours instead of lead_time
        print(f"Valid: {valid_start} to {valid_end} ({run_hours}h run)")
        print(f"Timestep: {dt} hourly")
        print(f"files will be saved to: {outpath} for regridding")

    os.makedirs(f"{outpath}/raw", exist_ok=True)

    FH = FastHerbie(
        DATES=[start_time],
        model='hrrr',
        product='sfc',
        # CHANGED: use run_hours instead of lead_time
        fxx=np.arange(dt, run_hours+dt, dt).tolist(),
        max_threads=max_threads,
        save_dir=f"{outpath}/raw"
    )

    apcp_a = [i.xarray(r":APCP:") for i in FH.file_exists]
    t2m_a = [i.xarray(r"TMP:2 m") for i in FH.file_exists]

    apcp_df = xr.combine_nested(apcp_a, concat_dim="valid_time", coords='different')
    t2m_df = xr.combine_nested(t2m_a, concat_dim="valid_time", coords='different')

    apcp_df = apcp_df.diff(dim='valid_time')
    apcp_df['tp'] = apcp_df['tp'] / float(dt)

    apcp_df['tp'].attrs['units'] = 'mm/hr'
    apcp_df['tp'].attrs['long_name'] = 'Hourly precipitation rate from total accumulated precipitation'
    apcp_df = apcp_df.rename({'tp' : 'pr', 'time': 'init_time'})
    apcp_df = apcp_df.rename({'valid_time': 'time'})

    t2m_df = ((t2m_df.resample(valid_time='1D',origin='start').min() + t2m_df.resample(valid_time='1D',origin='start').max()) / 2) - 273.15

    t2m_df['t2m'].attrs['units'] = 'degC'
    t2m_df['t2m'].attrs['long_name'] = "daily average 2-m air temperature (min/max method)"
    t2m_df = t2m_df.rename({'time': 'init_time'})
    t2m_df = t2m_df.rename({'valid_time': 'time'})

    if crop_domain:
        print(f'Taking regional subset with:')
        print(f'lats: {lat_max}, {lat_min}')
        print(f'lons: {lon_max}, {lon_min}')

        lon_min_360 = to_360(lon_min)
        lon_max_360 = to_360(lon_max)

        lon = apcp_df.longitude
        lat = apcp_df.latitude

        mask = (
            (lat >= lat_min) & (lat <= lat_max) &
            (lon >= lon_min_360) & (lon <= lon_max_360)
        )

        if not mask.any():
            raise ValueError("Requested lat/lon region does not intersect HRRR domain")

        y_idx, x_idx = np.where(mask.values)

        y_slice = slice(y_idx.min(), y_idx.max() + 1)
        x_slice = slice(x_idx.min(), x_idx.max() + 1)
        apcp_df = apcp_df.isel(y=y_slice, x=x_slice)
        t2m_df  = t2m_df.isel(y=y_slice, x=x_slice)

    apcp_path = f'/hrrr_pr_hrly_{valid_start.strftime("%Y%m%d")}_{valid_end.strftime("%Y%m%d")}.nc'
    t2m_path = f'/hrrr_t2m_daily_avg_{valid_start.strftime("%Y%m%d")}_{valid_end.strftime("%Y%m%d")}.nc'

    apcp_df.to_netcdf(f'{outpath}/{apcp_path}')
    t2m_df.to_netcdf(f'{outpath}/{t2m_path}')

    write_metadata(model_name='hrrr', 
                   pr_ds=apcp_df, 
                   t2m_ds=t2m_df, 
                   outpath=outpath, 
                   pr_path=apcp_path, 
                   t2m_path=t2m_path,
                   init_time=start_time,
                   run_hours=run_hours,
                   dt=dt)

    return apcp_df, t2m_df

def download_gfs(init_time, run_hours, outpath, dt=3, 
                  verbose=1, max_threads=50, crop_domain=True,
                  lat_max=41.024, lat_min=40.185, lon_max=-74.229, lon_min=-75.055):
    """
    Downloads and process a GFS dataset for a fixed
    forecast period, checks units and processes to the
    required file format needed for regridding.

    Parameters
    ----------
    init_time : string or datetime
        initialization time for the model run
    run_hours : int
        how many hours to run the forecast forward
    dt : int
        timestep between forecasts
    outpath : string
        where to store file output
    crop_domain : bool
        should we crop to a regional domain?
    lat_max, lat_min : float
        the maximum and minimum latitude for the regional crop
    lon_max, lon_min : float
        the maximum and minimum longitude for the regional crop

    Returns
    -------
    apcp_df, t2m_df : xr.Dataset
        datasets for precipitation and temperature
    """
    # CHANGED: parse init_time and calculate valid times directly
    if isinstance(init_time, str):
        init_time = _parse_time(init_time)
    start_time = init_time
    valid_start = start_time + timedelta(hours=dt)
    valid_end = start_time + timedelta(hours=run_hours)

    if verbose:
        print(f"Downloading GFS initialized at {start_time}")
        # CHANGED: print run_hours instead of lead_time
        print(f"Valid: {valid_start} to {valid_end} ({run_hours}h run)")
        print(f"Timestep: {dt} hourly")
        print(f"files will be saved to: {outpath} for regridding")

    os.makedirs(f"{outpath}/raw", exist_ok=True)

    FH = FastHerbie(
        DATES=[start_time],
        model='gfs',
        product='pgrb2.0p25',
        # CHANGED: use run_hours instead of lead_time
        fxx=np.arange(dt, run_hours+dt, dt).tolist(),
        max_threads=max_threads,
        save_dir=f"{outpath}/raw"
    )

    apcp_df = FH.xarray(r":APCP:surface:0-")
    t2m_df = FH.xarray(r":TMP:2 m above")

    apcp_df = apcp_df.swap_dims({"step": "valid_time"})
    apcp_df = apcp_df.drop_vars("step")

    t2m_df = t2m_df.swap_dims({"step": "valid_time"})
    t2m_df = t2m_df.drop_vars("step")

    apcp_df = apcp_df.diff(dim='valid_time')
    apcp_df['tp'] = apcp_df['tp'] / float(dt)

    apcp_df['tp'].attrs['units'] = 'mm/hr'
    apcp_df['tp'].attrs['long_name'] = 'Hourly precipitation rate from total accumulated precipitation'
    apcp_df = apcp_df.rename({'tp' : 'pr', 'time': 'init_time'})
    apcp_df = apcp_df.rename({'valid_time': 'time'})

    t2m_df = ((t2m_df.resample(valid_time='1D',origin='start').min() + t2m_df.resample(valid_time='1D',origin='start').max()) / 2) - 273.15

    t2m_df['t2m'].attrs['units'] = 'degC'
    t2m_df['t2m'].attrs['long_name'] = "daily average 2-m air temperature (min/max method)"
    t2m_df = t2m_df.rename({'time': 'init_time'})
    t2m_df = t2m_df.rename({'valid_time': 'time'})

    if crop_domain:
        print(f'Taking regional subset with:')
        print(f'lats: {lat_max}, {lat_min}')
        print(f'lons: {lon_max}, {lon_min}')

        lat_slice = slice(lat_max, lat_min)
        lon_slice = slice(to_360(lon_min), to_360(lon_max))

        apcp_df = apcp_df.sel(latitude=lat_slice, longitude=lon_slice)
        t2m_df = t2m_df.sel(latitude=lat_slice, longitude=lon_slice)

    apcp_path = f'/gfs_pr_hrly_{valid_start.strftime("%Y%m%d")}_{valid_end.strftime("%Y%m%d")}.nc'
    t2m_path = f'/gfs_t2m_daily_avg_{valid_start.strftime("%Y%m%d")}_{valid_end.strftime("%Y%m%d")}.nc'

    apcp_df.to_netcdf(f'{outpath}/{apcp_path}')
    t2m_df.to_netcdf(f'{outpath}/{t2m_path}')

    # CHANGED: pass init_time and run_hours to metadata
    write_metadata(model_name='gfs', 
                   pr_ds=apcp_df, 
                   t2m_ds=t2m_df, 
                   outpath=outpath, 
                   pr_path=apcp_path, 
                   t2m_path=t2m_path,
                   init_time=start_time,
                   run_hours=run_hours,
                   dt=dt)

    return apcp_df, t2m_df

def download_nbm():
    pass

def download_HIRESW():
    pass

def download_HREF():
    pass

def download_RRFS():
    pass

def download_RAP():
    pass

def download_HRDPS():
    pass

def download_RDPS():
    pass

def download_NAVGEM():
    pass

def download_NOGAPS():
    pass

def download_HAFS():
    pass

def download_driver(model_dict, outpath, req_models, crop_region,
                    lat_max, lat_min, lon_max, lon_min):
    """
    Function is a driver to download the requested HLM data.

    Parameters
    ----------
    model_dict : Dict[Dict]
        dictionary of models with INIT_TIMES, run_hours, and dt
    outpath : string
        path to save out the data
    req_models : List[string]
        the requested models to download
    crop_region : bool
        if True, crops the region to the specified bounding box
    lat_max, lat_min : float
        the maximum and minimum latitude for the regional crop
    lon_max, lon_min : float
        the maximum and minimum longitude for the regional crop
    """
    for mod_name in req_models:
        mod_opts = model_dict[mod_name]
        for init in mod_opts['INIT_TIMES']:
            check_request(mod_name, init, mod_opts['run_hours'], mod_opts['dt'])

    for mod_name in req_models:
        mod_opts = model_dict[mod_name]
        
        if mod_name == 'gfs'.casefold():
            for init in mod_opts['INIT_TIMES']:
                init_dt = _parse_time(init) if isinstance(init, str) else init
                init_str = init_dt.strftime('%Y%m%d_%H%M')
                # CHANGED: directory named by init time
                outpath_inner = f"{outpath}/gfs/init_{init_str}"

                _,_ = download_gfs(
                            init, mod_opts['run_hours'],
                            outpath_inner, dt=mod_opts['dt'], crop_domain=crop_region,
                            lat_max=lat_max, lat_min=lat_min,
                            lon_max=lon_max, lon_min=lon_min
                            )
                
        if mod_name == 'hrrr'.casefold():
            for init in mod_opts['INIT_TIMES']:
                init_dt = _parse_time(init) if isinstance(init, str) else init
                init_str = init_dt.strftime('%Y%m%d_%H%M')
                # CHANGED: directory named by init time
                outpath_inner = f"{outpath}/hrrr/init_{init_str}"
                
                _,_ = download_hrrr(
                            init, mod_opts['run_hours'], 
                            outpath_inner, dt=mod_opts['dt'], crop_domain=crop_region,
                            lat_max=lat_max, lat_min=lat_min,
                            lon_max=lon_max, lon_min=lon_min
                            )

                
    
    

#----------------------------------------------------------------------------------------------------------------
# GLOBALS

# REQUIRED SETTINGS
OUTPATH = '/home/lt0663/Documents/hlm_forecast/data/init_format'
MODELS_TO_PULL = ['gfs','hrrr']

# OPTIONAL CHANGES
CROP_REGION = True
LON_MIN, LON_MAX = -75.055, -74.229
LAT_MIN, LAT_MAX = 40.185, 41.024

# Optional reference times for hindcast validation (not used in downloads)
MET_EVENT_TIME = "2021-09-01 06:00"
HYDRO_EVENT_TIME = "2021-09-03 00:00"

# CHANGED: LEAD_TIMES replaced with INIT_TIMES + run_hours
MODELS = {
    'gfs' : {
        'INIT_TIMES' : ["2021-08-31 00:00", "2021-08-31 12:00", "2021-09-01 00:00"],
        'run_hours' : 120,
        'dt' : 1
    },

    'hrrr' : {
        'INIT_TIMES' : ["2021-09-01 00:00", "2021-09-01 12:00"],
        'run_hours' : 48,
        'dt' : 1
    }
}

# -----------------------------------------------------------------------------------------------------------------
# Code executes here

if __name__ == '__main__':
    print("Downloading data for TigerHLM forecasting!")

    # CHANGED: removed EVENT_TIME from call
    download_driver(MODELS, OUTPATH, 
                    MODELS_TO_PULL, CROP_REGION,
                    lat_max=LAT_MAX, lat_min=LAT_MIN,
                    lon_max=LON_MAX, lon_min=LON_MIN)
    
    print(f"Success! Data can be found at {OUTPATH}")
    print(f"It is highly recommended to check output! You can do this in quick_check_fcast.ipynb")
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
from herbie import Herbie, FastHerbie
import xarray as xr
xr.set_options(use_new_combine_kwarg_defaults=False)

### IMPORTANT GLOBALS FOR DOWNLOAD ###
avail_models = ['hrrr', 'rap', 'gfs', 'nbm', 'rrfs', 'href', 'hiresw',
                'hrdps', 'rdps']


# DICTIONARIES WITH MODEL REQUIREMENTS
NOAA_PARAM_DICT = {
    'GFS' : {'model' : 'gfs',
             'product' : 'pgrb2.0p25'},
    
    'HIRESW_ARW' : {'model' : 'hiresw',
                    'product' : 'arw_2p5km'},

    'HIRESW_FV3' : { 'model' : 'hiresw',
                    'product' : 'fv3_2p5km'},

    'HREF' : {'model' : 'href',
              'product' : 'mean',
              'domain' : 'conus'},
    
    'HRRR' : {'model' : 'hrrr',
              'product' : 'sfc'},

    'NAM' : {'model' : 'nam',
             'product' : 'conusnest.hiresf'},

    'NBM' : {'model' : 'nbm',
             'product' : 'co'},

    'RRFS' : {'model' : 'rrfs',
              'product' : 'prs',
              'domain' : 'conus'}

}

## come back and add the HWRF, HAFS global nest, SHIELD and TSHIELD, NAVGEM, NOGAPS, COAMPS, HRDPS

def quick_check(ds, var='tp', times=[1,10,20]):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    fig, ax = plt.subplots(1,3,figsize=(20,8),subplot_kw={'projection':ccrs.PlateCarree()})
    levels=np.arange(0.01,490,5)
    for i, idx in enumerate(times):
     dat = ds.isel(valid_time=idx)
     ax[i].contourf(dat.longitude, dat.latitude, dat[var], transform=ccrs.PlateCarree(), cmap='bone_r', levels=levels)
     ax[i].coastlines()
     ax[i].set_title(dat.valid_time.values)
    ds.to_netcdf('quick_check.nc')
    plt.show()

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


def _parse_event_time(event_time: str) -> datetime:
    """
    Safely parse event_time using a whitelist of formats.
    Raises ValueError if none match.
    """
    for fmt in _ALLOWED_FORMATS:
        try:
            return datetime.strptime(event_time, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"Unsupported datetime format: {event_time!r}. "
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
    init_time = _parse_event_time(event_time)
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

def check_request(model, lead_time, event_time, dt, strict=True):
    """
    Function checks to see if the submitted request is valid.
    Right now, this only supports hindcasts.

    Parameters
    -----------
    model : string
        name of the model being requested
    lead_time : int
        the number of hours in advance of the event requested
    event_time : string
        what time does the event occur?
    dt : int
        interval data is being downloaded at
    """
    # get the start time from the lead time
    start_time = _calc_fcst_init(event_time, lead_time)

    # now check all the parameters
    if model == 'GFS'.casefold():
        print("Validating GFS request...")
        if lead_time > 240:
            # GFS goes to 240 hours out (technically 384 but I don't think we should be trusting precip that long term)
            raise ValueError(
                f"Maximum forecast hours ({lead_time}) are invalid for "
                f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                "Ensure GFS lead_time is between 1 and 240.\n"
                f"See https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php for more"
            )
        
        #if lead_time % 6 != 0:
        if start_time.hour not in (0, 6, 12, 18):
            # If strict (default), fail unless an analysis is available at the requested initialization time
            # If not strict, find the nearest valid initialization before the requested lead time, 
            # and print a user warning stating the actual initialization time and the timestep the model will
            # begin at to achieve the requested lead time (???) this is presumably helpful behavior?
            if strict:
                raise ValueError(
                    f"Initialization at {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"is invalid for GFS, which must start at 0, 6, 12 or 18Z. "
                    f"See https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php for more"
                )
            else: 
                # get the new init and start dt
                new_init, new_start_dt = snap_to_prev_cycle(start_time)
                warnings.warn(
                    f"GFS lead time of {lead_time} will start in the middle of the run"
                    f"New init time will be {new_init.strftime('%Y-%m-%d %H:%M:%S')}"
                    f"Model will start at new simulation hour: {new_start_dt}"
                    )

        if dt % 3 != 0:
            # GFS timestep is 6 hourly so dt must be a multiple of 6
            raise ValueError(
                f"Timestep of dt = {dt} is invalid for GFS data"
                f"GFS has a 3 hour timestep, so ensure dt is a multiple of 6\n"
                f"See https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php for more"
            )
        
    if model == 'HRRR'.casefold():
        print('Validating HRRR request...')
        # HRRR intermediate cycles (01–05, 07–11, 13–17, 19–23 UTC)
        if start_time.hour not in (0, 6, 12, 18) and lead_time > 18:
            raise ValueError(
                f"Maximum forecast hours ({lead_time}) are invalid for "
                f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}."
                "Ensure HRRR lead_time is between 1 and 18 for intermediate init times.\n"
                f"HRRR analysis cycles hourly but is only available longer 18 hours at"
                f"the 0, 6, 12 and 18Z cycles\n"
                f"See https://rapidrefresh.noaa.gov/hrrr/ for more"
            )
        
        if lead_time > 48:
            raise ValueError(
                f"Maximum forecast hours ({lead_time}) are invalid for "
                f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                f"Ensure HRRR lead_time is between 1 and 48."
                f"See https://rapidrefresh.noaa.gov/hrrr/ for more"
            )

        if start_time.year < 2021 and lead_time > 36:
            raise ValueError(
                f"Maximum forecast hours ({lead_time}) are invalid for "
                f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                "Ensure lead_time is between 0 and 36 prior to 2021."
                f"See https://rapidrefresh.noaa.gov/hrrr/ for more"
            )
        # no need to check dt because HRRR is hourly

def write_file(path, content):
    #os.makedirs(os.path.dirname(path), exist_ok=True)  # CHECK ON THIS
    with open(path, "w") as f:
        f.write(content)

def write_metadata(model_name, pr_ds, t2m_ds, outpath, 
                   pr_path, t2m_path, start_time, lead_time, event_time, dt):
    """
    Function writes a metadata file for use with the 
    yaml data needed for mapping indices to the links

    Parameters
    ----------
    """
    template = f"""
    Metadata for {model_name} forecast with {lead_time}h lead time

    start: "{start_time}"
    end: "{event_time}"
    path: "{outpath}"
    
    variables:
    - name: "pr"
      file: "{pr_path}"
      time_resolution: "{dt}h"
      dims: {", ".join(pr_ds.dims)}

    - name: "t2m"
      file: "{t2m_path}"
      time_resolution: "24h"
      dims: {", ".join(t2m_ds.dims)}
"""
    write_file(f"{outpath}/metadata-{model_name}_{lead_time}h_lead.yaml", template)



def download_hrrr(event_time, lead_time, outpath, dt=1, 
                  verbose=1, max_threads=50, crop_domain=True,
                  lat_max=41.024, lat_min=40.185, lon_max=-74.229, lon_min=-75.055):
    """
    Downloads and process a HRRR dataset for a fixed
    forecast period, checks units and processes to the
    required file format needed for regridding.

    Parameters
    ----------
    event_time : string
        datetime string for when the event occurs
    lead_time : int
        the lead time for a given event
    dt : int
        timestep between forecasts
    outpath : string
        where to store file output
    crop_domain : bool
        should we crop to a regional domain?
    lat_max, lat_min : float
        the maximum and longitude to 

    Returns
    -------
    """
    # STEP 1: calculate the correct forecast hour and lead time
    # make sure to check the start time for the max_fxx
    start_time = _calc_fcst_init(event_time, lead_time)
    valid_start = start_time + timedelta(hours=dt)
    
    if verbose:
        print(f"Downloading HRRR initialized at {start_time}")
        print(f"Valid: {valid_start} to {event_time} ({lead_time - dt}h lead time)")
        print(f"Timestep: {dt} hourly")
        print(f"files will be saved to: {outpath} for regridding")

    # STEP 2: download the data with FastHerbie and concat into
    # one data set for each variable (vars are 2m temp and accumulated precip)
    os.makedirs(f"{outpath}/raw", exist_ok=True)

    FH = FastHerbie(
        DATES=[start_time],
        model='hrrr',
        product='sfc',
        fxx=np.arange(dt, lead_time+dt, dt).tolist(),  # you need to start at fxx=1 bc no precip at initialization (it accumulates in time)
        max_threads=max_threads,
        save_dir=f"{outpath}/raw"
    )

    # have herbie search the wgrib2 keys for the precip data
    apcp_a = [i.xarray(r":APCP:") for i in FH.file_exists]
    t2m_a = [i.xarray(r"TMP:2 m") for i in FH.file_exists]

    # concat into a dataframe
    apcp_df = xr.combine_nested(apcp_a, concat_dim="valid_time", coords='different')  # note that the actual variable is tp
    t2m_df = xr.combine_nested(t2m_a, concat_dim="valid_time", coords='different')    # note that the actual variable is t2m

    # STEP 3: deal with precipitation accumulation (timestep accum -> mm/h),
    # you will just have to assume that HLM starts one model timestep
    # after the atmospheric model initalization as at fxx=0 there is no 
    # accumulated precipitation
    apcp_df = apcp_df.diff(dim='valid_time')

    # convert to units of mm/hr and region crop
    apcp_df['tp'] = apcp_df['tp'] / float(dt)

    apcp_df['tp'].attrs['units'] = 'mm/hr'
    apcp_df['tp'].attrs['long_name'] = 'Hourly precipitation rate from total accumulated precipitation'
    apcp_df = apcp_df.rename({'tp' : 'pr', 'time': 'init_time'})
    apcp_df = apcp_df.rename({'valid_time': 'time'})

    # take the 24 hour average of the temperature & convert to C - NOTE: come back and fix to the min max method later
    t2m_df = ((t2m_df.resample(valid_time='1D',origin='start').min() + t2m_df.resample(valid_time='1D',origin='start').max()) / 2) - 273.15

    t2m_df['t2m'].attrs['units'] = 'degC'
    t2m_df['t2m'].attrs['long_name'] = "daily average 2-m air temperature (min/max method)"
    t2m_df = t2m_df.rename({'time': 'init_time'})
    t2m_df = t2m_df.rename({'valid_time': 'time'})

    # region subsetting (make sure the lons are 0-360)
    if crop_domain:
        print(f'Taking regional subset with:')
        print(f'lats: {lat_max}, {lat_min}')
        print(f'lons: {lon_max}, {lon_min}')

        lon_min_360 = to_360(lon_min)
        lon_max_360 = to_360(lon_max)

        # get all the coordinate points available in the dtaa
        lon = apcp_df.longitude
        lat = apcp_df.latitude

        # create a mask of the coords we want to subset to
        mask = (
            (lat >= lat_min) & (lat <= lat_max) &
            (lon >= lon_min_360) & (lon <= lon_max_360)
        )

        if not mask.any():
            raise ValueError("Requested lat/lon region does not intersect HRRR domain")

        # back calculate the indices of the bounding box
        y_idx, x_idx = np.where(mask.values)

        # construct the slice of curvilinear coordinates and slice the xarray dataframe
        y_slice = slice(y_idx.min(), y_idx.max() + 1)
        x_slice = slice(x_idx.min(), x_idx.max() + 1)
        apcp_df = apcp_df.isel(y=y_slice, x=x_slice)
        t2m_df  = t2m_df.isel(y=y_slice, x=x_slice)

    # TO DO!!! write a metadata file detailing the parameters

    # write to netcdf - filename is <PRODUCT>-<START>-<END>-<LEADTIME>.nc
    
    apcp_path = f'/hrrr_pr_hrly_{valid_start.strftime('%Y%m%d')}_{_parse_event_time(event_time).strftime('%Y%m%d')}.nc'
    t2m_path = f'/hrrr_t2m_daily_avg_{valid_start.strftime('%Y%m%d')}_{_parse_event_time(event_time).strftime('%Y%m%d')}.nc'

    apcp_df.to_netcdf(f'{outpath}/{apcp_path}')
    t2m_df.to_netcdf(f'{outpath}/{t2m_path}')

    # write the metadata file
    write_metadata(model_name='hrrr', 
                   pr_ds=apcp_df, 
                   t2m_ds=t2m_df, 
                   outpath=outpath, 
                   pr_path=apcp_path, 
                   t2m_path=t2m_path,
                   start_time=valid_start.strftime('%Y%m%d %H:%M'), 
                   lead_time=lead_time,
                   event_time=_parse_event_time(event_time).strftime('%Y%m%d  %H:%M'),
                   dt=dt)

    return apcp_df, t2m_df

def download_gfs(event_time, lead_time, outpath, dt=6, 
                  verbose=1, max_threads=50, crop_domain=True,
                  lat_max=41.024, lat_min=40.185, lon_max=-74.229, lon_min=-75.055):
    """
    Downloads and process a HRRR dataset for a fixed
    forecast period, checks units and processes to the
    required file format needed for regridding.

    Parameters
    ----------
    event_time : string
        datetime string for when the event occurs
    lead_time : int
        the lead time for a given event
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
    # STEP 1: calculate the correct forecast hour and lead time
    # make sure to check the start time for the max_fxx
    start_time = _calc_fcst_init(event_time, lead_time)
    valid_start = start_time + timedelta(hours=dt)

    if verbose:
        print(f"Downloading GFS initialized at {start_time}")
        print(f"Valid: {valid_start} to {event_time} ({lead_time - dt}h lead time)")
        print(f"Timestep: {dt} hourly")
        print(f"files will be saved to: {outpath} for regridding")

    # STEP 2: download the data with FastHerbie and concat into
    # one data set for each variable (vars are 2m temp and accumulated precip)
    os.makedirs(f"{outpath}/raw", exist_ok=True)

    FH = FastHerbie(
        DATES=[start_time],
        model='gfs',
        product='pgrb2.0p25',
        fxx=np.arange(dt, lead_time+dt, dt).tolist(),  # you need to start at fxx=1 bc no precip at initialization (it accumulates in time)
        max_threads=max_threads,
        save_dir=f"{outpath}/raw"
    )

    # have herbie search the wgrib2 keys for the precip data
    apcp_df = FH.xarray(r":APCP:")
    t2m_df = FH.xarray(r":TMP:2 m above")

    apcp_df = apcp_df.swap_dims({"step": "valid_time"})
    apcp_df = apcp_df.drop_vars("step")
    #apcp_df = apcp_df.rename({"valid_time": "time"})

    t2m_df = t2m_df.swap_dims({"step": "valid_time"})
    t2m_df = t2m_df.drop_vars("step")
    #t2m_df = t2m_df.rename({"valid_time": "time"})


    # STEP 3: deal with precipitation accumulation (timestep accum -> mm/h),
    # you will just have to assume that HLM starts one model timestep
    # after the atmospheric model initalization as at fxx=0 there is no 
    # accumulated precipitation
    apcp_df = apcp_df.diff(dim='valid_time')

    # convert to units of mm/hr and region crop
    apcp_df['tp'] = apcp_df['tp'] / float(dt)


    apcp_df['tp'].attrs['units'] = 'mm/hr'
    apcp_df['tp'].attrs['long_name'] = 'Hourly precipitation rate from total accumulated precipitation'
    apcp_df = apcp_df.rename({'tp' : 'pr', 'time': 'init_time'})
    apcp_df = apcp_df.rename({'valid_time': 'time'})

    # take the 24 hour average of the temperature & convert to C - NOTE: come back and fix to the min max method later
    t2m_df = ((t2m_df.resample(valid_time='1D',origin='start').min() + t2m_df.resample(valid_time='1D',origin='start').max()) / 2) - 273.15

    t2m_df['t2m'].attrs['units'] = 'degC'
    t2m_df['t2m'].attrs['long_name'] = "daily average 2-m air temperature (min/max method)"
    t2m_df = t2m_df.rename({'time': 'init_time'})
    t2m_df = t2m_df.rename({'valid_time': 'time'})

    # region subsetting (make sure the lons are 0-360)
    if crop_domain:
        print(f'Taking regional subset with:')
        print(f'lats: {lat_max}, {lat_min}')
        print(f'lons: {lon_max}, {lon_min}')

        lat_slice = slice(lat_max, lat_min)
        lon_slice = slice(to_360(lon_min), to_360(lon_max))

        apcp_df = apcp_df.sel(latitude=lat_slice, longitude=lon_slice)
        t2m_df = t2m_df.sel(latitude=lat_slice, longitude=lon_slice)

    # TO DO!!! write a metadata file detailing the parameters

    # write to netcdf - filename is <PRODUCT>-<START>-<END>.nc
    apcp_path = f'/gfs_pr_hrly_{valid_start.strftime('%Y%m%d')}_{_parse_event_time(event_time).strftime('%Y%m%d')}.nc'
    t2m_path = f'/gfs_t2m_daily_avg_{valid_start.strftime('%Y%m%d')}_{_parse_event_time(event_time).strftime('%Y%m%d')}.nc'

    apcp_df.to_netcdf(f'{outpath}/{apcp_path}')
    t2m_df.to_netcdf(f'{outpath}/{t2m_path}')

    # write the metadata file
    write_metadata(model_name='gfs', 
                   pr_ds=apcp_df, 
                   t2m_ds=t2m_df, 
                   outpath=outpath, 
                   pr_path=apcp_path, 
                   t2m_path=t2m_path,
                   start_time=valid_start.strftime('%Y%m%d %H:%M'), 
                   lead_time=lead_time,
                   event_time=_parse_event_time(event_time).strftime('%Y%m%d  %H:%M'),
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

def download_driver(model_dict, event_time, outpath, req_models, crop_region,
                    lat_max, lat_min, lon_max, lon_min):
    """
    Function is a driver to download the requested HLM data. I guess this
    could be multiprocessed, but I highly reccomend against this since
    Herbie itself is multithreaded and if the model is on a NOAA or NCEI
    ftp it can rate limit and block you (whoops......)

    Parameters
    ----------
    model_dict : Dict[Dict]
        the dictionary from the global MODELS (mutable so not a default argument),
        but should be a dictionary of models, with requested lead times and timesteps
    event_time : string
        time of the event that we are hindcasting for
    outpath : string
        path to save out the data
    req_models : List[string]
        the requested models to download, used to index model_dict
    crop_region : bool
        if True, crops the region to the specified bounding box
    lat_max, lat_min : float
        the maximum and minimum latitude for the regional crop
    lon_max, lon_min : float
        the maximum and minimum longitude for the regional crop

    Returns
    --------
    """
    # first check the request is valid so any partially invalid requests will fail
    for mod_name in req_models:
        # loops over all the requested models and runs the check on them
        mod_opts = model_dict[mod_name]
        [check_request(mod_name, lead, event_time, mod_opts['dt']) for lead in mod_opts['LEAD_TIMES']]

    # now basically do the same as above but actually download
    for mod_name in req_models:
        mod_opts = model_dict[mod_name]
        
        if mod_name == 'GFS'.casefold():
            for lead in mod_opts['LEAD_TIMES']:
                outpath_inner = f"{outpath}/gfs/lead-time_{lead}hrs"

                _,_ = download_gfs(
                            EVENT_TIME, lead,
                            outpath_inner, dt=mod_opts['dt'], crop_domain=crop_region,
                            lat_max=lat_max, lat_min=lat_min,
                            lon_max=lon_max, lon_min=lon_min
                            )
                
        if mod_name == 'HRRR'.casefold():
            for lead in mod_opts['LEAD_TIMES']:
                outpath_inner = f"{outpath}/hrrr/lead-time_{lead}hrs"
                _,_ = download_hrrr(
                            EVENT_TIME, lead, 
                            outpath_inner, dt=mod_opts['dt'], crop_domain=crop_region,
                            lat_max=lat_max, lat_min=lat_min,
                            lon_max=lon_max, lon_min=lon_min
                            )
                
    
    

#----------------------------------------------------------------------------------------------------------------
# GLOBALS

# REQUIRED SETTINGS
EVENT_TIME = "2021-09-03 00:00"  # YYYY-MM-DD hh:mm (when does the EVENT we are back forecasting happen)
OUTPATH = '/home/lt0663/Documents/hlm_forecast/data/meta_checker'  # where to save out the datasets
MODELS_TO_PULL = ['hrrr', 'gfs']   # which of the supported models to download at this time

# OPTIONAL CHANGES - currently same as function defaults
CROP_REGION = True                      # do you want a subset of the full model domain when you download?
LON_MIN, LON_MAX = -75.055, -74.229     # minimum and maximum longitude (set for Raritan AORC domain rn)
LAT_MIN, LAT_MAX = 40.185, 41.024       # minimum and maximum latitude (set for Raritan AORC domain rn)

# dictionary contains the requests for each model, if you don't want a model just leave it as is...
# keys: LEAD_TIMES: hours (how far in ADVANCE of the event do we want to get the forecasts)
#       dt : hours (interval between model steps to download at each lead time i.e. frequency of input to HLM)
MODELS = {
    'gfs' : {
        'LEAD_TIMES' : [72, 48], #np.arange(72,0,-12).tolist(),  # between 0 and 240, must be a multiple of 6 for GFS (runs available at 0z, 6z, 12z, 18z)
        'dt' : 3                   # must be a multiple of 6 for GFS (minimum is 6 hours)
    },

    'hrrr' : {
        'LEAD_TIMES' : [36, 12], #np.arange(48,0,-12).tolist(),  # between 0 and 48, but 18-48 hours are only available for 0, 6, 12, 18Z runs
        'dt' : 1                      # any because HRRR is hourly (minimum is 1 hour)
    }
}

# -----------------------------------------------------------------------------------------------------------------
# Code executes here

if __name__ == '__main__':
    print("Downloading data for TigerHLM forecasting!")

    download_driver(MODELS, EVENT_TIME, OUTPATH, 
                    MODELS_TO_PULL, CROP_REGION,
                    lat_max=LAT_MAX, lat_min=LAT_MIN,
                    lon_max=LON_MAX, lon_min=LON_MIN)
    
    print(f"Success! Data can be found at {OUTPATH}")
    print(f"It is highly recommended to check output! You can do this in quick_check_fcast.ipynb")
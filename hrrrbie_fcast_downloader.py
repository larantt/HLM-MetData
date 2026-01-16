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
from datetime import datetime, timedelta
import os

import numpy as np
from herbie import Herbie, FastHerbie
import xarray as xr

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

def check_request(model, lead_time, event_time, dt):
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
        print("Validating GFS ")
        if lead_time > 384:
            # GFS goes to 384 hours out
            raise ValueError(
                f"Maximum forecast hours ({lead_time}) are invalid for "
                f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                "Ensure GFS lead_time is between 1 and 384.\n"
                f"See https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php for more"
            )
        if dt % 6 != 0:
            # GFS timestep is 6 hourly so dt must be a multiple of 6
            raise ValueError(
                f"Timestep of dt = {dt} is invalid for GFS data"
                f"GFS has a 6 hour timestep, so ensure dt is a multiple of 6\n"
                f"See https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php for more"
            )
        
    if model == 'HRRR'.casefold():
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
    valid_start = start_time + timedelta(hours=1)

    if lead_time > 48:
        raise ValueError(
            f"Maximum forecast hours ({lead_time}) are invalid for "
            f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}. "
            "Ensure lead_time is between 1 and 48."
        )

    if start_time.year < 2021 and lead_time > 36:
        raise ValueError(
            f"Maximum forecast hours ({lead_time}) are invalid for "
            f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}. "
            "Ensure lead_time is between 0 and 36 prior to 2021."
        )

    # HRRR intermediate cycles (01–05, 07–11, 13–17, 19–23 UTC)
    if start_time.hour not in (0, 6, 12, 18) and lead_time > 18:
        raise ValueError(
            f"Maximum forecast hours ({lead_time}) are invalid for "
            f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}. "
            "Ensure lead_time is between 1 and 18 for intermediate init times."
        )
    
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
        fxx=np.arange(1, lead_time+dt, dt).tolist(),  # you need to start at fxx=1 bc no precip at initialization (it accumulates in time)
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
    apcp_df = apcp_df.diff(dim='valid_time', n=dt)

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
    
    apcp_df.to_netcdf(f'{outpath}/hrrr_pr_hrly_{valid_start.strftime('%Y%m%d%H')}_{_parse_event_time(event_time).strftime('%Y%m%d%H%M')}.nc')
    t2m_df.to_netcdf(f'{outpath}/hrrr_t2m_daily_avg_{valid_start.strftime('%Y%m%d%H')}_{_parse_event_time(event_time).strftime('%Y%m%d%H%M')}.nc')

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
    valid_start = start_time + timedelta(hours=1)

    if lead_time > 384:
        raise ValueError(
            f"Maximum forecast hours ({lead_time}) are invalid for "
            f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}. "
            "Ensure lead_time is between 1 and 384."
        )
    
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
        fxx=np.arange(1, lead_time+dt, dt).tolist(),  # you need to start at fxx=1 bc no precip at initialization (it accumulates in time)
        max_threads=max_threads,
        save_dir=f"{outpath}/raw"
    )

    # have herbie search the wgrib2 keys for the precip data
    apcp_df = FH.xarray(r":APCP:")
    t2m_df = FH.xarray(r":TMP:2 m above")

    # STEP 3: deal with precipitation accumulation (timestep accum -> mm/h),
    # you will just have to assume that HLM starts one model timestep
    # after the atmospheric model initalization as at fxx=0 there is no 
    # accumulated precipitation
    apcp_df = apcp_df.diff(dim='valid_time', n=dt)

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

    # write to netcdf - filename is <PRODUCT>-<START>-<END>-<LEADTIME>.nc
    
    apcp_df.to_netcdf(f'{outpath}/gfs_pr_hrly_{valid_start.strftime('%Y%m%d%H')}_{_parse_event_time(event_time).strftime('%Y%m%d%H%M')}.nc')
    t2m_df.to_netcdf(f'{outpath}/gfs_t2m_daily_avg_{valid_start.strftime('%Y%m%d%H')}_{_parse_event_time(event_time).strftime('%Y%m%d%H%M')}.nc')

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

def download_driver():
    """
    Function downloads 
    """
    pass

###############
### GLOBALS ###
###############

# REQUIRED SETTINGS
EVENT_TIME = "2021-09-03 00:00"  # YYYY-MM-DD hh:mm (when does the EVENT we are back forecasting happen)
LEAD_TIME = 84                   # hours (how far in ADVANCE of the event do we want to get the forecast)
OUTPATH = '/home/lt0663/Documents/hlm_forecast/gfs'  # where to save out the datasets
MODELS_TO_PULL = ['gfs', 'hrrr']   # which of the supported models to download at this time

# OPTIONAL CHANGES - currently same as function defaults
CROP_REGION = True                      # do you want a subset of the full model domain when you download?
LON_MIN, LON_MAX = -75.055, -74.229     # minimum and maximum longitude (set for Raritan AORC domain rn)
LAT_MIN, LAT_MAX = 40.185, 41.024       # minimum and maximum latitude (set for Raritan AORC domain rn)

MODELS = {
    'gfs' : {
        'LEAD_TIMES' : [360, 84]  # hours (how far in ADVANCE of the event do we want to get the forecasts)
    },

    'hrrr' : {
        'LEAD_TIMES' : [48, 36, 13]
    }
}



# download the data
apcp_df, t2m_df = download_gfs(EVENT_TIME, LEAD_TIME, 
                               OUTPATH, crop_domain=CROP_REGION,
                               lat_max=LAT_MAX, lat_min=LAT_MIN,
                               lon_max=LON_MAX, lon_min=LON_MIN)


; regrid_hrrr.ncl

begin

  ; Load files
  f_in  = addfile("/home/lt0663/Documents/hlm_forecast/hrrr/hrrr_pr_hrly_202109011300_202109030000.nc", "r")
  f_tgt = addfile("/home/lt0663/Documents/hlm_forecast/AORC_pr_hrly_20210901_20210907.nc", "r")

  ; Get source data and coordinates
  pr   = f_in->pr
  lat2d = f_in->latitude
  lon2d = f_in->longitude

  ; Get target grid
  lat1d = f_tgt->latitude
  lon1d = f_tgt->longitude

  ; Set up regridding options
  opt = True
  opt@InterpMethod = "conserve"   ; or "bilinear"
  opt@SrcGridLat = lat2d
  opt@SrcGridLon = lon2d
  opt@DstGridLat = lat1d
  opt@DstGridLon = lon1d
  opt@ForceOverwrite = True
  opt@SrcGridType = "curvilinear"
  opt@DstGridType = "rectilinear"

  ; Regrid
  pr_regrid = ESMF_regrid(pr, opt)

  ; Write output
  system("rm -f hrrr_pr_regridded.nc")
  f_out = addfile("hrrr_pr_regridded.nc", "c")
  
  f_out->pr = pr_regrid
  f_out->latitude = lat1d
  f_out->longitude = lon1d
  f_out->time = f_in->time

end
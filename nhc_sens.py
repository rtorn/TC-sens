import os
import logging
import sys
import netCDF4 as nc
import xarray as xr
import numpy as np
import datetime as dt
import scipy.stats

import matplotlib
from IPython.core.pylabtools import figsize, getfigs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
import cartopy.crs as ccrs

from SensPlotRoutines import plotVecSens, plotScalarSens, computeSens, addDrop, writeSensFile, addRangeRings, set_projection, background_map

def ComputeSensitivity(datea, fhr, metname, atcf, config):
   '''
   This function is the workhorse of the code because it computes the sensitivity of a given forecast
   metric, which is computed for each ensemble member earlier in the code, to a set of forecast 
   fields at a given forecast hour.  These forecast fields were also computed and placed in separate
   netCDF files in an earlier routine.  The result of this routine is a set of sensitivity graphics
   that can be used for various purposes, and if desired netCDF files that can be ingested into
   AWIPS, or into the traveling salesman software for flight planning.  Note that this routine is 
   called several times within the main code; therefore, it could be parallelized.

   Attributes:
       datea   (string):  initialization date of the forecast (yyyymmddhh format)
       fhr        (int):  forecast hour
       metname (string):  name of forecast metric to compute sensitivity for
       atcf     (class):  ATCF class object that includes ensemble information
       config   (dict.):  dictionary that contains configuration options (read from file)
   '''

   nens = int(len(atcf.atcf_files))

   plotDict = {}
   for key in config['sens']:
      plotDict[key] = config['sens'][key]

   fhrt = '%0.3i' % fhr

   logging.warning('Sensitivity of {0} to F{1}'.format(metname,fhrt))

   if config['storm'][-1] == "l":
      bb = 'al'
   elif config['storm'][-1] == "e":
      bb = 'ep'
   elif storm[-1] == "w":
      bb = 'wp'
   bbnn = '{0}{1}'.format(bb,config['storm'][-3:-1])

   #  Compute the ensemble-mean lat/lon for plotting
   elat, elon = atcf.ens_lat_lon_time(fhr)

   m_lat = 0.0
   m_lon = 0.0
   e_cnt = 0.0
   for n in range(nens):
      if elat[n] != atcf.missing and elon[n] != atcf.missing:
         e_cnt = e_cnt + 1.0
         m_lat = m_lat + elat[n]
         m_lon = m_lon + elon[n]

   if e_cnt > 0.0:
      plotDict['tcLat'] = m_lat / e_cnt
      plotDict['tcLon'] = m_lon / e_cnt
   else:
      plotDict['tcLat'] = 0.
      plotDict['tcLon'] = 0. 

   plotDict['plotTitle'] = '{0} F{1}'.format(datea,fhrt)
   plotDict['fileTitle'] = 'TEST JHT-Torn ECMWF Sensitivity'
   plotDict['initDate']  = '{0}-{1}-{2} {3}:00:00'.format(datea[0:4],datea[4:6],datea[6:8],datea[8:10])
   plotDict['sig_value'] = scipy.stats.t.ppf(q=1.0-float(plotDict.get('significance_level','.05'))/2,df=nens)
   plotDict['ring_center_lat']=float(plotDict['tcLat'])
   plotDict['ring_center_lon']=float(plotDict['tcLon'])

   stceDict = plotDict.copy()
   stceDict['output_sens']=False
   stceDict['range_rings']='True'
   stceDict['min_lat']=float(plotDict['tcLat'])-float(plotDict.get('storm_center_radius', 10.))
   stceDict['max_lat']=float(plotDict['tcLat'])+float(plotDict.get('storm_center_radius', 10.))
   stceDict['min_lon']=float(plotDict['tcLon'])-float(plotDict.get('storm_center_radius', 10.))
   stceDict['max_lon']=float(plotDict['tcLon'])+float(plotDict.get('storm_center_radius', 10.))
   stceDict['grid_interval']=3.
   stceDict['barb_interval']=3
   stceDict["figsize"]=(8.5,11)

   if 'min_lat' in config['sens']:
      lat1 = np.minimum(float(plotDict['min_lat']), stceDict['min_lat'])
      lat2 = np.maximum(float(plotDict['max_lat']), stceDict['max_lat'])
      lon1 = np.minimum(float(plotDict['min_lon']), stceDict['min_lon'])
      lon2 = np.maximum(float(plotDict['max_lon']), stceDict['max_lon'])
   elif bb == 'al':
      plotDict['min_lat'] = 8.0
      plotDict['max_lat'] = 65.0
      plotDict['min_lon'] = -140.0
      plotDict['max_lon'] = -20.0
      lat1 = np.minimum(plotDict['min_lat'], stceDict['min_lat'])
      lat2 = np.maximum(plotDict['max_lat'], stceDict['max_lat'])
      lon1 = np.minimum(plotDict['min_lon'], stceDict['min_lon'])
      lon2 = np.maximum(plotDict['max_lon'], stceDict['max_lon'])
   elif bb == 'ep':
      lat1 = 8.0
      lat2 = 65.0
      lon1 = -180.0
      lon2 = -80.0
   else:
      lat1 = config['min_lat']
      lat2 = config['max_lat']
      lon1 = config['min_lon']
      lon2 = config['max_lon']

   datea_dt = dt.datetime.strptime(datea, '%Y%m%d%H')
   datef_dt = datea_dt + dt.timedelta(hours=fhr)
   datef    = datef_dt.strftime("%Y%m%d%H")
   if 'dropsonde_file' in plotDict:
      plotDict['dropsonde_file'] = plotDict['dropsonde_file'].format(datef)
      stceDict['dropsonde_file'] = stceDict['dropsonde_file'].format(datef)

   #  Obtain the metric information (here read from file)
   try:
      mfile = nc.Dataset('{0}/{1}_{2}.nc'.format(config['work_dir'],datea,metname))
   except IOError:
      logging.error('{0}/{1}_{2}.nc does not exist'.format(config['work_dir'],datea,metname))
      return

   metric = mfile.variables['fore_met_init'][:]
   nens   = len(metric)
   metric = metric[:] - np.mean(metric, axis=0)
   mettype = mfile.FORECAST_METRIC_SHORT_NAME

   plotDict['sensmax'] = np.std(metric) * 0.9
   plotDict['sig_value'] = scipy.stats.t.ppf(q=1-.05/2,df=nens)
   stceDict['sensmax'] = float(plotDict['sensmax'])
   stceDict['sig_value'] = float(plotDict['sig_value']) 

   #  Read major axis direction if appropriate  
   if hasattr(mfile.variables['fore_met_init'],'units'):
      plotDict['metricUnits'] = mfile.variables['fore_met_init'].units

   if hasattr(mfile,'X_DIRECTION_VECTOR') and hasattr(mfile,'Y_DIRECTION_VECTOR'):
      ivec = mfile.X_DIRECTION_VECTOR
      jvec = mfile.Y_DIRECTION_VECTOR
   else:
      ivec = 1.0
      jvec = 0.0

   if mettype == 'trackeof':
      pltlist   = {'steer': True, 'h500': True, 'pv250': True, 'pv850': False, 'e700': False, \
                   'e850': False, 'vor': False, 'q58': False, 'ivt': False} 
   elif mettype == 'pcpeof':
      pltlist   = {'steer': True, 'h500': True, 'pv250': True, 'pv850': True, 'e700': True, \
                   'e850': True, 'vor': False, 'q58': True, 'ivt': True}
   else:
      pltlist   = {'steer': True, 'h500': True, 'pv250': True, 'pv850': True, 'e700': True, \
                   'e850': True, 'vor': True, 'q58': True, 'ivt': False}


   #  Read the ensemble zonal and meridional steering wind, compute ensemble mean
   ufile = '{0}/{1}_f{2}_usteer_ens.nc'.format(config['work_dir'],datea,fhrt)
   vfile = '{0}/{1}_f{2}_vsteer_ens.nc'.format(config['work_dir'],datea,fhrt)
   if pltlist['steer'] and os.path.isfile(ufile) and os.path.isfile(vfile):

      ds = xr.open_dataset(ufile)
      if ds.latitude[0] > ds.latitude[1]:
         lattmp1 = lat1
         lattmp2 = lat2
         lat1    = lattmp2
         lat2    = lattmp1

      uens = ds.ensemble_data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2)).squeeze()
      lat = uens.latitude.values
      lon = uens.longitude.values
      usteer = np.mean(uens, axis=0)
      usteer.attrs['units'] = ds['ensemble_data'].attrs['units']
      uvar = np.var(uens, axis=0)

      ds = xr.open_dataset(vfile)
      vens = ds.ensemble_data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2)).squeeze()
      vsteer  = np.mean(vens, axis=0)
      vsteer.attrs['units'] = ds.ensemble_data.attrs['units']
      vvar = np.var(vens, axis=0)

      #  Compute sensitivity with respect to zonal steering wind
      sens, sigv = computeSens(uens, usteer, uvar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(uvar[:,:])

      outdir = '{0}/{1}/sens/usteer'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      if plotDict.get('output_sens', False) and 'intmajtrack' in metname:
         if not os.path.isdir('{0}/{1}'.format(datea,bbnn)):
            os.makedirs('{0}/{1}'.format(datea,bbnn), exist_ok=True)
         writeSensFile(lat, lon, fhr, usteer, sens, sigv, '{0}/{1}/{0}_f{2}_usteer_sens.nc'.format(datea,bbnn,fhrt), plotDict)

      plotVecSens(lat, lon, sens, usteer, vsteer, sigv, '{0}/{1}_f{2}_usteer_sens.png'.format(outdir,datea,fhrt), plotDict)

      if e_cnt > 0:
         outdir = '{0}/{1}/sens_sc/usteer'.format(config['figure_dir'],metname)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         plotVecSens(lat, lon, sens, usteer, vsteer, sigv, '{0}/{1}_f{2}_usteer_sens.png'.format(outdir,datea,fhrt), stceDict)


      #  Compute sensitivity with respect to meridional steering wind
      sens, sigv = computeSens(vens, vsteer, vvar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(vvar[:,:])

      outdir = '{0}/{1}/sens/vsteer'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      if plotDict.get('output_sens', False) and 'intmajtrack' in metname:
         writeSensFile(lat, lon, fhr, usteer, sens, sigv, '{0}/{1}/{0}_f{2}_vsteer_sens.nc'.format(datea,bbnn,fhrt), plotDict)

      plotVecSens(lat, lon, sens, usteer, vsteer, sigv, '{0}/{1}_f{2}_vsteer_sens.png'.format(outdir,datea,fhrt), plotDict)

      if e_cnt > 0:
         outdir = '{0}/{1}/sens_sc/vsteer'.format(config['figure_dir'],metname)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         plotVecSens(lat, lon, sens, usteer, vsteer, sigv, '{0}/{1}_f{2}_vsteer_sens.png'.format(outdir,datea,fhrt), stceDict)


      #  Rotate wind into major axis direction, compute sensitivity to steering wind in that direction
      wens = ivec * uens[:,:,:] + jvec * vens[:,:,:]
      emea = np.mean(wens, axis=0)
      evar = np.var(wens, axis=0)
      emea.attrs['units'] = ds.ensemble_data.attrs['units']
      sens, sigv = computeSens(wens, emea, evar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])
      masens = sens[:,:]

      outdir = '{0}/{1}/sens/masteer'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      if plotDict.get('output_sens', False) and 'intmajtrack' in metname:
         writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{0}_f{2}_masteer_sens.nc'.format(datea,bbnn,fhrt), plotDict)

      plotVecSens(lat, lon, sens, usteer, vsteer, sigv, '{0}/{1}_f{2}_masteer_sens.png'.format(outdir,datea,fhrt), plotDict)

      if e_cnt > 0:
         outdir = '{0}/{1}/sens_sc/masteer'.format(config['figure_dir'],metname)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         plotVecSens(lat, lon, sens, usteer, vsteer, sigv, '{0}/{1}_f{2}_masteer_sens.png'.format(outdir,datea,fhrt), stceDict)


   #  Read steering flow streamfunction, compute sensitivity to that field, if the file exists
   ensfile = '{0}/{1}_f{2}_ssteer_ens.nc'.format(config['work_dir'],datea,fhrt)
   if pltlist['steer'] and os.path.isfile(ensfile):

      ds = xr.open_dataset(ensfile)
      ens = ds.ensemble_data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2)).squeeze()
      lat = ens.latitude.values
      lon = ens.longitude.values
      emea  = np.mean(ens, axis=0)
      emea.attrs['units'] = ds.ensemble_data.attrs['units']
      evar = np.var(ens, axis=0)

      sens, sigv = computeSens(ens, emea, evar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

      outdir = '{0}/{1}/sens/ssteer'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      plotDict['meanCntrs'] = np.arange(-100, 104, 4)
      plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_ssteer_sens.png'.format(outdir,datea,fhrt), plotDict)

      if e_cnt > 0:
         outdir = '{0}/{1}/sens_sc/ssteer'.format(config['figure_dir'],metname)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         stceDict['meanCntrs'] = plotDict['meanCntrs']
         plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_ssteer_sens.png'.format(outdir,datea,fhrt), stceDict)


   #  Read steering flow vorticity, compute sensitivity to that field, if the file exists
   ensfile = '{0}/{1}_f{2}_csteer_ens.nc'.format(config['work_dir'],datea,fhrt)
   if pltlist['steer'] and os.path.isfile(ensfile):

      ds = xr.open_dataset(ensfile)
      ens = ds.ensemble_data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2)).squeeze()
      lat = ens.latitude.values
      lon = ens.longitude.values
      emea  = np.mean(ens, axis=0)
      emea.attrs['units'] = ds.ensemble_data.attrs['units']
      evar = np.var(ens, axis=0)

      sens, sigv = computeSens(ens, emea, evar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])
      svsens = sens[:,:]

      outdir = '{0}/{1}/sens/csteer'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      plotDict['meanCntrs'] = np.array([-5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
      plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_csteer_sens.png'.format(outdir,datea,fhrt), plotDict)

      if e_cnt > 0:
         outdir = '{0}/{1}/sens_sc/csteer'.format(config['figure_dir'],metname)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         stceDict['meanCntrs'] = plotDict['meanCntrs']
         plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_csteer_sens.png'.format(outdir,datea,fhrt), stceDict)


   #  Read 250 hPa PV, compute sensitivity to that field, if the file exists
   ensfile = '{0}/{1}_f{2}_pv250hPa_ens.nc'.format(config['work_dir'],datea,fhrt)
   if pltlist['pv250'] and os.path.isfile(ensfile): 
      
      ds = xr.open_dataset(ensfile)
      ens = ds.ensemble_data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2)).squeeze()
      lat = ens.latitude.values
      lon = ens.longitude.values
      emea  = np.mean(ens, axis=0)
      emea.attrs['units'] = ds.ensemble_data.attrs['units']
      evar = np.var(ens, axis=0)

      sens, sigv = computeSens(ens, emea, evar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])
      pvsens = sens[:,:]

      outdir = '{0}/{1}/sens/pv250hPa'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      if plotDict.get('output_sens', False) and 'intmajtrack' in metname:
         writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{0}_f{2}_pv250hPa_sens.nc'.format(datea,bbnn,fhrt), plotDict)

      plotDict['meanCntrs'] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_pv250hPa_sens.png'.format(outdir,datea,fhrt), plotDict)


   #  Read 850 hPa PV, compute sensitivity to that field, if the file exists
   ensfile = '{0}/{1}_f{2}_pv850hPa_ens.nc'.format(config['work_dir'],datea,fhrt)
   if pltlist['pv850'] and os.path.isfile(ensfile):

      ds = xr.open_dataset(ensfile)
      ens = ds.ensemble_data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2)).squeeze()
      lat = ens.latitude.values
      lon = ens.longitude.values
      emea  = np.mean(ens, axis=0)
      emea.attrs['units'] = ds.ensemble_data.attrs['units']
      evar = np.var(ens, axis=0)

      sens, sigv = computeSens(ens, emea, evar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

      outdir = '{0}/{1}/sens/pv850hPa'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      if plotDict.get('output_sens', False) and 'intmajtrack' in metname:
         writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{0}_f{2}_pv850hPa_sens.nc'.format(datea,bbnn,fhrt), plotDict)

      plotDict['meanCntrs'] = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4])
      plotDict['clabel_fmt'] = "%2.1f"
      plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_pv850hPa_sens.png'.format(outdir,datea,fhrt), plotDict)
      del plotDict['clabel_fmt']

      if e_cnt > 0:
         outdir = '{0}/{1}/sens_sc/pv850hPa'.format(config['figure_dir'],metname)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         stceDict['meanCntrs'] = plotDict['meanCntrs']
         stceDict['clabel_fmt'] = "%2.1f" 
         plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_pv850hPa_sens.png'.format(outdir,datea,fhrt), stceDict)
         del stceDict['clabel_fmt']


   #  Read 850 hPa theta-e, compute sensitivity to that field
   ensfile = '{0}/{1}_f{2}_ivt_ens.nc'.format(config['work_dir'],datea,fhrt)
   if pltlist['ivt'] and os.path.isfile(ensfile):

      ds = xr.open_dataset(ensfile)
      ens = ds.ensemble_data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2)).squeeze()
      lat = ens.latitude.values
      lon = ens.longitude.values
      emea  = np.mean(ens, axis=0)
      emea.attrs['units'] = ds.ensemble_data.attrs['units']
      evar = np.var(ens, axis=0)

      sens, sigv = computeSens(ens, emea, evar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])
      ivsens = sens[:,:]

      outdir = '{0}/{1}/sens/ivt'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      if plotDict.get('output_sens', 'False')=='True' and 'intmajtrack' in metname:
         writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{0}_f{2}_ivt_sens.nc'.format(datea,bbnn,fhrt), plotDict)

      plotDict['meanCntrs'] = np.array([200., 400., 600., 800., 1000., 1300., 1600.])
      plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_ivt_sens.png'.format(outdir,datea,fhrt), plotDict)

      if e_cnt > 0:
         outdir = '{0}/{1}/sens_sc/ivt'.format(config['figure_dir'],metname)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         stceDict['meanCntrs'] = plotDict['meanCntrs']
         plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_ivt_sens.png'.format(outdir,datea,fhrt), stceDict)


   #  Read 850 hPa theta-e, compute sensitivity to that field
   ensfile = '{0}/{1}_f{2}_e700hPa_ens.nc'.format(config['work_dir'],datea,fhrt)
   if pltlist['e700'] and os.path.isfile(ensfile):

      ds = xr.open_dataset(ensfile)
      ens = ds.ensemble_data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2)).squeeze()
      lat = ens.latitude.values
      lon = ens.longitude.values
      emea  = np.mean(ens, axis=0)
      emea.attrs['units'] = ds.ensemble_data.attrs['units']
      evar = np.var(ens, axis=0)

      sens, sigv = computeSens(ens, emea, evar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

      outdir = '{0}/{1}/sens/e700hPa'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      if plotDict.get('output_sens', 'False')=='True' and 'intmajtrack' in metname:
         writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{0}_f{2}_e700hPa_sens.nc'.format(datea,bbnn,fhrt), plotDict)

      plotDict['meanCntrs'] = np.arange(270, 390, 3)
      plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_e700hPa_sens.png'.format(outdir,datea,fhrt), plotDict)


   #  Read 850 hPa theta-e, compute sensitivity to that field
   ensfile = '{0}/{1}_f{2}_e850hPa_ens.nc'.format(config['work_dir'],datea,fhrt)
   if pltlist['e850'] and os.path.isfile(ensfile):

      ds = xr.open_dataset(ensfile)
      ens = ds.ensemble_data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2)).squeeze()
      lat = ens.latitude.values
      lon = ens.longitude.values
      emea  = np.mean(ens, axis=0)
      emea.attrs['units'] = ds.ensemble_data.attrs['units']
      evar = np.var(ens, axis=0)

      sens, sigv = computeSens(ens, emea, evar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

      outdir = '{0}/{1}/sens/e850hPa'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      if plotDict.get('output_sens', 'False')=='True' and 'intmajtrack' in metname:
         writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{0}_f{2}_e850hPa_sens.nc'.format(datea,bbnn,fhrt), plotDict)

      plotDict['meanCntrs'] = np.arange(270, 390, 3)
      plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_e850hPa_sens.png'.format(outdir,datea,fhrt), plotDict)


   #  Read 925 hPa zonal and meridional wind, compute sensitivity
   ufile = '{0}/{1}_f{2}_u925hPa_ens.nc'.format(config['work_dir'],datea,fhrt)
   vfile = '{0}/{1}_f{2}_v925hPa_ens.nc'.format(config['work_dir'],datea,fhrt)
   if ('pcp' in metname or 'precip' in metname or 'wnd' in metname) and os.path.isfile(ufile) and os.path.isfile(vfile):

      ds = xr.open_dataset(ufile)
      uens = ds.ensemble_data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2)).squeeze()
      lat = uens.latitude.values
      lon = uens.longitude.values
      umea = np.mean(uens, axis=0)
      umea.attrs['units'] = ds.ensemble_data.attrs['units']
      uvar = np.var(uens, axis=0)

      ds = xr.open_dataset(vfile)
      vens = ds.ensemble_data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2)).squeeze()
      vmea  = np.mean(vens, axis=0)
      vmea.attrs['units'] = ds.ensemble_data.attrs['units']
      vvar = np.var(vens, axis=0)

      sens, sigv = computeSens(uens, umea, uvar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(uvar[:,:])

      outdir = '{0}/{1}/sens/u925hPa'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      plotVecSens(lat, lon, sens, umea, vmea, sigv, '{0}/{1}_f{2}_u925hPa_sens.png'.format(outdir,datea,fhrt), plotDict)

      sens, sigv = computeSens(vens, vmea, vvar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(vvar[:,:])

      outdir = '{0}/{1}/sens/v925hPa'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      plotVecSens(lat, lon, sens, umea, vmea, sigv, '{0}/{1}_f{2}_v925hPa_sens.png'.format(outdir,datea,fhrt), plotDict)


   plist = [850]
   for pres in plist:

      ensfile = '{0}/{1}_f{2}_vor{3}hPa_ens.nc'.format(config['work_dir'],datea,fhrt,pres)
      if pltlist['vor'] and os.path.isfile(ensfile):

         efile = nc.Dataset(ensfile)
         lat   = efile.variables['latitude'][:]
         lon   = efile.variables['longitude'][:]
         ens   = np.squeeze(efile.variables['ensemble_data'][:])
         emea  = np.mean(ens, axis=0)
         emea.units = efile.variables['ensemble_data'].units
         evar = np.var(ens, axis=0)

         sens, sigv = computeSens(ens, emea, evar, metric)
         sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

         outdir = '{0}/{1}/sens/vor{2}hPa'.format(config['figure_dir'],metname,pres)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

#         if plotDict.get('output_sens', 'False')=='True' and 'intmajtrack' in metname:
#            writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{0}_f{2}_vor{3}hPa_sens.nc'.format(datea,bbnn,fhrt,pres), plotDict)

         plotDict['meanCntrs'] = np.array([-5.0, -4.0, -3.0, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0])
         plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_vor{3}hPa_sens.png'.format(outdir,datea,fhrt,pres), plotDict)


         if e_cnt > 0:
            outdir = '{0}/{1}/sens_sc/vor{2}hPa'.format(config['figure_dir'],metname,pres)
            if not os.path.isdir(outdir):
               os.makedirs(outdir, exist_ok=True)

            stceDict['meanCntrs'] = plotDict['meanCntrs']
            plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_vor{3}hPa_sens.png'.format(outdir,datea,fhrt,pres), stceDict)


   #  Read 500 hPa height, compute sensitivity to that field
   ensfile = '{0}/{1}_f{2}_h500hPa_ens.nc'.format(config['work_dir'],datea,fhrt)
   if pltlist['h500'] and os.path.isfile(ensfile):      

      ds = xr.open_dataset(ensfile)
      ens = ds.ensemble_data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2)).squeeze()
      lat = ens.latitude.values
      lon = ens.longitude.values
      emea  = np.mean(ens, axis=0)
      emea.attrs['units'] = ds.ensemble_data.attrs['units']
      evar = np.var(ens, axis=0)

      sens, sigv = computeSens(ens, emea, evar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

      outdir = '{0}/{1}/sens/h500hPa'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      if plotDict.get('output_sens', False) and 'intmajtrack' in metname:
         writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{0}_f{2}_h500hPa_sens.nc'.format(datea,bbnn,fhrt), plotDict)

      plotDict['meanCntrs'] = np.arange(4800, 6000, 60)
      plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_h500hPa_sens.png'.format(outdir,datea,fhrt), plotDict)


   ensfile = '{0}/{1}_f{2}_q500-850hPa_ens.nc'.format(config['work_dir'],datea,fhrt)
   if pltlist['q58'] and os.path.isfile(ensfile):

      ds = xr.open_dataset(ensfile)
      ens = ds.ensemble_data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2)).squeeze()
      lat = ens.latitude.values
      lon = ens.longitude.values
      emea  = np.mean(ens, axis=0)
      emea.attrs['units'] = ds.ensemble_data.attrs['units']
      evar = np.var(ens, axis=0)
 
      sens, sigv = computeSens(ens, emea, evar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])
      q58sens = sens[:,:]

      outdir = '{0}/{1}/sens/q500-850hPa'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      if plotDict.get('output_sens', False) and 'intmajtrack' in metname:
         writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{0}_f{2}_q500-850hPa_sens.nc'.format(datea,bbnn,fhrt), plotDict)

      plotDict['meanCntrs'] = np.arange(4, 40, 4)
      plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_q500-850hPa_sens.png'.format(outdir,datea,fhrt), plotDict)

      if e_cnt > 0:
         outdir = '{0}/{1}/sens_sc/q500-850hPa'.format(config['figure_dir'],metname)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         stceDict['meanCntrs'] = plotDict['meanCntrs']
         plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_q500-850hPa_sens.png'.format(outdir,datea,fhrt), stceDict)


   if mettype == 'trackeof' and eval(config['sens'].get('plot_summary','False')):

      outdir = '{0}/{1}/sens/summ'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      stceDict['plotTitle'] = '{0} track, init: {1}, valid: {2} (Hour: {3})'.format(config['storm'],datea,datef,fhrt)
      stceDict['plotLegend'] = ['Major Steering Wind', 'Steering Vorticity']
      plotSummarySens(lat, lon, usteer, vsteer, masens, svsens, np.array([]), '{0}/{1}_f{2}_summ_sens.png'.format(outdir,datea,fhrt), stceDict)

   if mettype == 'wndeof' and eval(config['sens'].get('plot_summary','False')):

      outdir = '{0}/{1}/sens/summ'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      stceDict['plotTitle'] = '{0} max. wind, init: {1}, valid: {2} (Hour: {3})'.format(config['storm'],datea,datef,fhrt)
      stceDict['plotLegend'] = ['Major Steering Wind', 'Steering Vorticity', '500-850 hPa qvap']
      plotSummarySens(lat, lon, usteer, vsteer, masens, svsens, q58sens, '{0}/{1}_f{2}_summ_sens.png'.format(outdir,datea,fhrt), stceDict)

   if mettype == 'pcpeof' and eval(config['sens'].get('plot_summary','False')):

      outdir = '{0}/{1}/sens/summ'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      stceDict['plotTitle'] = '{0} precip., init: {1}, valid: {2} (Hour: {3})'.format(config['storm'],datea,datef,fhrt)
      stceDict['plotLegend'] = ['Steering Vorticity', '500-850 hPa qvap', 'IVT']
      plotSummarySens(lat, lon, usteer, vsteer, svsens, q58sens, ivsens, '{0}/{1}_f{2}_summ_sens.png'.format(outdir,datea,fhrt), stceDict)


def plotSummarySens(lat, lon, usteer, vsteer, f1sens, f2sens, f3sens, fileout, plotDict):
   '''
   Function that plots the sensitivity of a forecast metric to a scalar field, along
   with the ensemble mean field in contours, and the statistical significance in 
   stippling.  The user has the option to add customized elements to the plot, including
   range rings, locations of rawinsondes/dropsondes, titles, etc.  These are all turned
   on or off using the configuration file.

   Attributes:
       lat      (float):  Vector of latitude values
       lon      (float):  Vector of longitude values
       sens     (float):  2D array of sensitivity field
       fileout (string):  Name of output figure in .png format
       plotDict (dict.):  Dictionary that contains configuration options
   '''

   minLat = float(plotDict.get('min_lat', np.amin(lat)))
   maxLat = float(plotDict.get('max_lat', np.amax(lat)))
   minLon = float(plotDict.get('min_lon', np.amin(lon)))
   maxLon = float(plotDict.get('max_lon', np.amax(lon)))

   sencnt = float(plotDict.get('summary_contour', 0.36))

   #  Create basic figure, including political boundaries and grid lines
   fig = plt.figure(figsize=plotDict.get('figsize',(11,8.5)))

   ax = background_map(plotDict.get('projection', 'PlateCarree'), minLon, maxLon, minLat, maxLat, plotDict)

#   plt1 = plt.contour(lon,lat,f1sens,[-sencnt, sencnt], linewidths=2.0, colors='g',transform=ccrs.PlateCarree())
#   pltp = plt.contourf(lon,lat,f1sens,[-sencnt, sencnt], hatches=['/', None, '/'], colors='none', \
#                        extend='both',transform=ccrs.PlateCarree())
#   for i, collection in enumerate(pltp.collections):
#      collection.set_edgecolor('g')
   plt1 = plt.contourf(lon,lat,f1sens,[-sencnt, sencnt],extend='both',alpha=0.5,transform=ccrs.PlateCarree(), \
                        cmap=matplotlib.colors.ListedColormap(("#00FF00","#FFFFFF","#00FF00")))

#   plt2 = plt.contour(lon,lat,f2sens,[-sencnt, sencnt], linewidths=2.0, colors='m',transform=ccrs.PlateCarree())
#   pltp = plt.contourf(lon,lat,f2sens,[-sencnt, sencnt], hatches=[' \ ', None, ' \ '], colors='none', \
#                       extend='both',transform=ccrs.PlateCarree())
#   for i, collection in enumerate(pltp.collections):
#      collection.set_edgecolor('m')
   plt2 = plt.contourf(lon,lat,f2sens,[-sencnt, sencnt],extend='both',alpha=0.5,transform=ccrs.PlateCarree(), \
                        cmap=matplotlib.colors.ListedColormap(("#FF00FF","#FFFFFF","#FF00FF")))

#   plt3 = plt.contour(lon,lat,f3sens,[-sencnt, sencnt], linewidths=2.0, colors='b', \
#                             zorder=10, transform=ccrs.PlateCarree())
#   pltt = plt.contourf(lon,lat,f3sens,[-sencnt, sencnt], hatches=['|', None, '|'], colors='none', \
#                       extend='both',transform=ccrs.PlateCarree())
#   for i, collection in enumerate(pltt.collections):
#      collection.set_edgecolor('b')
   if f3sens.any():
      plt3 = plt.contourf(lon,lat,f3sens,[-sencnt, sencnt],extend='both',alpha=0.5,transform=ccrs.PlateCarree(), \
                           cmap=matplotlib.colors.ListedColormap(("#0000FF","#FFFFFF","#0000FF")))

   if 'plotTitle' in plotDict:
      plt.title(plotDict['plotTitle'])

   if 'plotLegend' in plotDict:
      lamin = float(plotDict.get('min_lat', np.amin(lat)))
      lamax = float(plotDict.get('max_lat', np.amax(lat)))
      lomin = float(plotDict.get('min_lon', np.amin(lon)))
      lomax = float(plotDict.get('max_lon', np.amax(lon)))
      plt.text(lomin, lamin-(lamax-lamin)*0.06, plotDict['plotLegend'][0], color='#00FF00', fontsize=14, \
                   horizontalalignment='left', transform=ccrs.PlateCarree())
      plt.text((lomin+lomax)*0.5, lamin-(lamax-lamin)*0.06, plotDict['plotLegend'][1], color='#FF00FF', fontsize=14, \
                   horizontalalignment='center', transform=ccrs.PlateCarree())
      if len(plotDict['plotLegend']) > 2:
         plt.text(lomax, lamin-(lamax-lamin)*0.06, plotDict['plotLegend'][2], color='#0000FF', fontsize=14, \
                   horizontalalignment='right', transform=ccrs.PlateCarree())

#   if tcLat != -9999. and tcLon != -9999.:
#      plt.plot(tcLon, tcLat, 'o', color='black', markersize=10)

   addRangeRings(plotDict['ring_center_lat'], plotDict['ring_center_lon'], lat, lon, plt, plotDict)

   addDrop(plotDict.get("dropsonde_file","null"), plt, plotDict)

   plt.savefig(fileout,format='png',dpi=120,bbox_inches='tight')
   plt.close(fig)


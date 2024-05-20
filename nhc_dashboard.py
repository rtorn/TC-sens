'''
   nhc_dashboard.py - python function that can be used to read gridded sensitivity files 
                      for track, intensity, 2D wind, and precipitation metrics and plot the
                      NHC dashboard figure.  If traveling salesman dropsonde and track files
                      exist, they will also be plotted on top of the sensitivity fields for
                      each metric as well.  This program can be used to determine the extent
                      to which a proposed flight track aligns with the sensitive regions.

    usage:  python nhc_dashboard.py --init yyyymmddhh --storm XXXXXXNNB --fhour hhh 
                             --param paramfile --drops DROPFILE --turns TURNFILE

      where:

       --init is the initialization date in yyyymmddhh format
       --storm is the TC name (XXXXXX is the storm name, NN is the number, B is the basin)
       --fhour is the forecast hour to plot the sensitivity grids
       --param is the parameter file path (optional, otherwise goes to default values in default.parm)
       --drops is the path to the dropsonde locations file
       --turns is the path to the aircraft turns file

'''
import os, sys
import argparse
import configparser
import tarfile
import xarray as xr
import numpy as np
import datetime as dt

import matplotlib
from IPython.core.pylabtools import figsize, getfigs
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs

sys.path.append('../esens-util')
from SensPlotRoutines import addDrop, addTurns, addRangeRings, set_projection, background_map

def main():

   #  Read the initialization time and storm from the command line
   exp_parser = argparse.ArgumentParser()
   exp_parser.add_argument('--init',  action='store', type=str, required=True)
   exp_parser.add_argument('--storm', action='store', type=str, required=True)
   exp_parser.add_argument('--fhour', action='store', type=int, required=True)
   exp_parser.add_argument('--param', action='store', type=str)
   exp_parser.add_argument('--drops', action='store', type=str)
   exp_parser.add_argument('--turns', action='store', type=str)
   args = exp_parser.parse_args()

   #  Read paramter file and set location paramters to case-specific location
   config = configparser.ConfigParser()
   config.read(args.param)
   config['locations']['work_dir']   = '{0}/{1}.{2}'.format(config['locations']['work_dir'],args.init,args.storm)
   config['locations']['output_dir'] = '{0}/{1}.{2}'.format(config['locations']['output_dir'],args.init,args.storm)
   config['locations']['figure_dir'] = '{0}/{1}.{2}'.format(config['locations']['figure_dir'],args.init,args.storm)

   os.chdir(config['locations']['work_dir'])

   #  untar the gridded sensitivity information if it is not present in work directory
   if not os.path.isdir('{0}/{1}'.format(config['locations']['work_dir'],args.init)):
      tarout = '{0}/{1}.tar'.format(config['locations']['outgrid_dir'] + '/../awips',args.init)
      if ( os.path.isfile(tarout) and tarfile.is_tarfile(tarout) ):
         os.system('tar --skip-old-files -xf {0}'.format(tarout))

   datea_dt = dt.datetime.strptime(args.init, '%Y%m%d%H')
   datef_dt = datea_dt + dt.timedelta(hours=args.fhour)
   datef    = datef_dt.strftime("%Y%m%d%H")
   fhrt     = '%0.3i' % args.fhour

   if args.storm[-1] == "l":
      bb = 'al'
      lat1 = float(config['sens'].get('min_lat','8.0'))
      lat2 = float(config['sens'].get('max_lat','65.0'))
      lon1 = float(config['sens'].get('min_lon','-140.0'))
      lon2 = float(config['sens'].get('max_lon','-20.0'))
   elif args.storm[-1] == "e":
      bb = 'ep'
      lat1 = float(config['sens'].get('min_lat','8.0'))
      lat2 = float(config['sens'].get('max_lat','65.0'))
      lon1 = float(config['sens'].get('min_lon','-180.0'))
      lon2 = float(config['sens'].get('max_lon','-80.0'))
   elif args.storm[-1] == "w":
      bb = 'wp'
   bbnn = '{0}{1}'.format(bb,args.storm[-3:-1])

   #  Read lat/lon arrays over the desired domain
   ds = xr.open_dataset('{0}/{1}/track/{0}_f{2}_masteer_sens.nc'.format(args.init,bbnn,fhrt))

   if ds.lat[0] > ds.lat[1]:
      lattmp1 = lat1
      lattmp2 = lat2
      lat1    = lattmp2
      lat2    = lattmp1   

   lat = ds.lat.sel(lat=slice(lat1, lat2)).values.squeeze()
   lon = ds.lon.sel(lon=slice(lon1, lon2)).values.squeeze()

   plotDict = {}
   for key in config['sens']:
      plotDict[key] = config['sens'][key]

   plotDict['tcLat'] = ds.attrs['TC_latitude']
   plotDict['tcLon'] = ds.attrs['TC_longitude']
   plotDict['ring_center_lat']=float(plotDict['tcLat'])
   plotDict['ring_center_lon']=float(plotDict['tcLon'])
   plotDict['range_rings']='True'
   plotDict['min_lat']=float(plotDict['tcLat'])-float(plotDict.get('storm_center_radius', 10.))
   plotDict['max_lat']=float(plotDict['tcLat'])+float(plotDict.get('storm_center_radius', 10.))
   plotDict['min_lon']=float(plotDict['tcLon'])-float(plotDict.get('storm_center_radius', 10.))
   plotDict['max_lon']=float(plotDict['tcLon'])+float(plotDict.get('storm_center_radius', 10.))
   plotDict['grid_interval']=3.
   plotDict['subplot'] = 'True'
   plotDict['subrows'] = 2
   plotDict['subcols'] = 2

   #  Determine if dropsonde file is provided; if not, try to use parameter file value with valid date
   if args.drops:
      plotDict['dropsonde_file'] = args.drops
   elif 'dropsonde_file' in config['sens']:
      plotDict['dropsonde_file'] = config['sens']['dropsonde_file'].format(datef)

   if 'dropsonde_file' in plotDict:
      if not os.path.isfile(plotDict['dropsonde_file']):
         print("{0} dropsonde file does not exist.  Drops will not be plotted".format(plotDict['dropsonde_file']))
   else:
      print("No information on the dropsonde file.")

   #  Determine if a turns file is provided; if not, try to use parameter file value with valid date
   if args.turns:
      plotDict['turns_file'] = args.turns
   elif 'turns_file' in config['sens']:
      plotDict['turns_file'] = config['sens']['turns_file'].format(datef)

   if 'turns_file' in plotDict:
      if not os.path.isfile(plotDict['turns_file']):
         print("{0} turns file does not exist.  Turns will not be plotted".format(plotDict['turns_file']))
   else:
      print("No information on the turns file.")


   fig = plt.figure(figsize=(11,11))

   #  Read sensitivity information for the intensity metric
   masens = ds.track_sensitivity.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).squeeze()

   infile = '{0}/{1}/track/{0}_f{2}_csteer_sens.nc'.format(args.init,bbnn,fhrt)
   if os.path.isfile(infile):
      ds = xr.open_dataset(infile)
      svsens = ds.track_sensitivity.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).squeeze()
   else:
      svsens = np.array([])

   #  create track metric panel 
   plotDict['subnumber'] = 1
   plotDict['plotTitle'] = 'track'
   plotDict['plotLegend'] = ['Major Steering Wind', 'Steering Vorticity']
   plotSummarySens(lat, lon, masens, svsens, np.array([]), plotDict)


   #  Read sensitivity information for the intensity metric (if this metric exists)
   infile = '{0}/{1}/inten/{0}_f{2}_vor850hPa_sens.nc'.format(args.init,bbnn,fhrt)
   if os.path.isfile(infile):
      ds = xr.open_dataset(infile) 
      vo850sens = ds.inten_sensitivity.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).squeeze()
   else:
      vo850sens = np.array([])

   infile = '{0}/{1}/inten/{0}_f{2}_csteer_sens.nc'.format(args.init,bbnn,fhrt)
   if os.path.isfile(infile):
      ds = xr.open_dataset(infile)
      svsens = ds.inten_sensitivity.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).squeeze()
   else:
      svsens = np.array([])

   infile = '{0}/{1}/inten/{0}_f{2}_q500-850hPa_sens.nc'.format(args.init,bbnn,fhrt)
   if os.path.isfile(infile):
      ds = xr.open_dataset(infile)
      q58sens = ds.inten_sensitivity.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).squeeze()
   else:
      q58sens = np.array([])

   #  create intensity metric panel
   plotDict['subnumber'] = 2
   plotDict['plotTitle'] = 'intensity'
   plotDict['plotLegend'] = ['850 hPa vorticity', 'Steering Vorticity', '500-850 hPa qvap']
   plotSummarySens(lat, lon, vo850sens, svsens, q58sens, plotDict)


   #  Read sensitivity information for the 2D max. wind metric (if this metric exists)
   infile = '{0}/{1}/wind/{0}_f{2}_masteer_sens.nc'.format(args.init,bbnn,fhrt)
   if os.path.isfile(infile):
      ds = xr.open_dataset(infile)
      masens = ds.wind_sensitivity.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).squeeze()
   else:
      masens = np.array([])

   infile = '{0}/{1}/wind/{0}_f{2}_csteer_sens.nc'.format(args.init,bbnn,fhrt)
   if os.path.isfile(infile):
      ds = xr.open_dataset(infile)
      svsens = ds.wind_sensitivity.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).squeeze()
   else:
      svsens = np.array([])

   infile = '{0}/{1}/wind/{0}_f{2}_q500-850hPa_sens.nc'.format(args.init,bbnn,fhrt)
   if os.path.isfile(infile):
      ds = xr.open_dataset(infile)
      q58sens = ds.wind_sensitivity.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).squeeze()
   else:
      q58sens = np.array([])

   #  create max. wind metric panel
   plotDict['subnumber'] = 3
   plotDict['plotTitle'] = 'max. wind'
   plotDict['plotLegend'] = ['Major Steering Wind', 'Steering Vorticity', '500-850 hPa qvap']
   plotSummarySens(lat, lon, masens, svsens, q58sens, plotDict)


   #   Read sensitivity information for the precipitation metric (if this metric exists)
   infile = '{0}/{1}/pcp/{0}_f{2}_csteer_sens.nc'.format(args.init,bbnn,fhrt)
   if os.path.isfile(infile):
      ds = xr.open_dataset(infile)
      svsens = ds.pcp_sensitivity.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).squeeze()
   else:
      svsens = np.array([])   

   infile = '{0}/{1}/pcp/{0}_f{2}_q500-850hPa_sens.nc'.format(args.init,bbnn,fhrt)
   if os.path.isfile(infile):
      ds = xr.open_dataset(infile)
      q58sens = ds.pcp_sensitivity.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).squeeze()
   else:
      q58sens = np.array([])

   infile = '{0}/{1}/pcp/{0}_f{2}_ivt_sens.nc'.format(args.init,bbnn,fhrt)
   if os.path.isfile(infile):
      ds = xr.open_dataset(infile)
      ivsens = ds.pcp_sensitivity.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).squeeze()
   else:
      ivsens = np.array([])

   #  Create Precipitation Metric Panel
   plotDict['subnumber'] = 4
   plotDict['plotTitle'] = 'precip.'
   plotDict['plotLegend'] = ['IVT', 'Steering Vorticity', '500-850 hPa qvap']
   plotSummarySens(lat, lon, ivsens, svsens, q58sens, plotDict)

   #  Add a title and write out the file
   fig.suptitle('{0} ({1}), init: {2}, valid: {3} (Hour: {4})'.format(args.storm[0:-3].capitalize(), \
                                                    bbnn.upper(),args.init,datef,fhrt), fontsize=16, y=0.92)

   fileout = '{0}/{1}_f{2}_dashboard.png'.format(config['locations']['figure_dir'],args.init,fhrt)
   plt.savefig(fileout,format='png',dpi=120,bbox_inches='tight')
   plt.close(fig)


def plotSummarySens(lat, lon, f1sens, f2sens, f3sens, plotDict):
   '''
   Function that plots the individual panels of the NHC dashboard sensitivity plots for a 
   specific forecast metric.  Each forecast metric has a separate set of forecast fields to 
   compute the sensitivity through, passed in via f*sens arrays.  The user has the option to
   add customized elements to the plot, including range rings, locations of 
   rawinsondes/dropsondes, turns, titles, etc.  These are all turned
   on or off using the configuration file.

   Attributes:
       lat      (float):  Latitude of fields
       lon      (float):  Longitude of fields
       f1sens   (float):  Sensitivity of metric to forecast field #1
       f2sens   (float):  Sensitivity of metric to forecast field #2
       f3sens   (float):  Sensitivity of metric to forecast field #3
       plotDict (dict.):  Dictionary that contains configuration options
   '''

   minLat = float(plotDict.get('min_lat', np.amin(lat)))
   maxLat = float(plotDict.get('max_lat', np.amax(lat)))
   minLon = float(plotDict.get('min_lon', np.amin(lon)))
   maxLon = float(plotDict.get('max_lon', np.amax(lon)))
   sencnt = float(plotDict.get('summary_contour', 0.36))

   ax = background_map(plotDict.get('projection', 'PlateCarree'), minLon, maxLon, minLat, maxLat, plotDict)

   #  Plot each sensitivity area for each field that is available for this metric
   hasSens = False
   if f1sens.any():
      plt1 = plt.contourf(lon,lat,f1sens,[-sencnt, sencnt],extend='both',alpha=0.5,transform=ccrs.PlateCarree(), \
                           cmap=matplotlib.colors.ListedColormap(("#00FF00","#FFFFFF","#00FF00")))
      hasSens = True

   if f2sens.any():
      plt2 = plt.contourf(lon,lat,f2sens,[-sencnt, sencnt],extend='both',alpha=0.5,transform=ccrs.PlateCarree(), \
                           cmap=matplotlib.colors.ListedColormap(("#FF00FF","#FFFFFF","#FF00FF")))
      hasSens = True

   if f3sens.any():
      plt3 = plt.contourf(lon,lat,f3sens,[-sencnt, sencnt],extend='both',alpha=0.5,transform=ccrs.PlateCarree(), \
                           cmap=matplotlib.colors.ListedColormap(("#0000FF","#FFFFFF","#0000FF")))
      hasSens = True

   #  Add text if there are no fields with senitivity output available (i.e., metric is missing)
   if not hasSens:
      plt.text((minLon+maxLon)*0.5, minLat+(maxLat-minLat)*0.66, 'Not', color='k', fontsize=24, \
                   horizontalalignment='center', transform=ccrs.PlateCarree())
      plt.text((minLon+maxLon)*0.5, minLat+(maxLat-minLat)*0.33, 'Available', color='k', fontsize=24, \
                   horizontalalignment='center', transform=ccrs.PlateCarree())

   #  plot the title if that string is present
   if 'plotTitle' in plotDict:
      plt.title(plotDict['plotTitle'])

   #  Plot the text figure legend underneath each panel
   if 'plotLegend' in plotDict:
      plt.text(minLon, minLat-(maxLat-minLat)*0.09, plotDict['plotLegend'][0], color='#00FF00', fontsize=8, \
                   horizontalalignment='left', transform=ccrs.PlateCarree())
      plt.text((minLon+maxLon)*0.5, minLat-(maxLat-minLat)*0.09, plotDict['plotLegend'][1], color='#FF00FF', fontsize=8, \
                   horizontalalignment='center', transform=ccrs.PlateCarree())
      if len(plotDict['plotLegend']) > 2:
         plt.text(maxLon, minLat-(maxLat-minLat)*0.09, plotDict['plotLegend'][2], color='#0000FF', fontsize=8, \
                   horizontalalignment='right', transform=ccrs.PlateCarree())

   #  Add range rings if that information is present
   addRangeRings(plotDict['ring_center_lat'], plotDict['ring_center_lon'], lat, lon, plt, plotDict)

   #  Plot dropsondes and turns if file exists AND there was sensitivity information for this metric
   if hasSens:
      addDrop(plotDict.get("dropsonde_file","null"), plt, plotDict)
      addTurns(plotDict.get("turns_file","null"), plt, plotDict)


if __name__ == '__main__':

   main()

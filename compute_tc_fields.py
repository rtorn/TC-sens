import os, sys
import logging
import json
import importlib
import netCDF4 as nc
import numpy as np
import datetime as dt
import metpy.constants as mpcon
import metpy.calc as mpcalc
from metpy.units import units

import surgery
import grid_calc

def ComputeTCFields(datea, fhr, atcf, config):
    '''
    Function that computes individual 2D ensemble forecast fields for a given forecast hour.  These ensemble fields 
    will be used in the next stage of the code to compute the sensitivity.  The result will be a series of netCDF
    files, one for each forecast field, that contains all ensemble members.  The forecast fields that are computed
    are determined via the configuration options.

    Attributes:
        datea (string):  initialization date of the forecast (yyyymmddhh format)
        fhr      (int):  forecast hour
        atcf   (class):  ATCF class object that includes ensemble information
        config (dict.):  dictionary that contains configuration options (read from file)
    '''

    wnd_lev_1 = [250, 500]
    wnd_lev_2 = [350, 500]
    n_wnd_lev = len(wnd_lev_1)

    # Read steering flow parameters, or use defaults
    steerp1  = float(config['fields'].get('steer_level1', '300'))
    steerp2  = float(config['fields'].get('steer_level2', '850'))
    tcradius = float(config['fields'].get('steer_radius', '333'))

    # lat/lon domain info
    lat1 = float(config['fields'].get('min_lat','0.'))
    lat2 = float(config['fields'].get('max_lat','65.'))
    lon1 = float(config['fields'].get('min_lon','-180.'))
    lon2 = float(config['fields'].get('max_lon','-10.'))

    if not 'min_lat' in config:
       config.update({'min_lat': lat1})
       config.update({'max_lat': lat2})
       config.update({'min_lon': lon1})
       config.update({'max_lon': lon2})

    nens = int(len(atcf.atcf_files))
    datea_dt = dt.datetime.strptime(datea, '%Y%m%d%H')
    fff = str(fhr + 1000)[1:]
    datef_dt = datea_dt + dt.timedelta(hours=fhr)
    datef = datef_dt.strftime("%m%d%H%M")

    dpp = importlib.import_module(config['io_module'])

    logging.warning("Computing hour {0} ensemble fields".format(fff))

    #  Obtain the ensemble lat/lon information, replace missing values with mean
    ens_lat, ens_lon = atcf.ens_lat_lon_time(fhr)

    e_cnt = 0
    m_lat = 0.0
    m_lon = 0.0
    for n in range(nens):
       if ens_lat[n] != atcf.missing and ens_lon[n] != atcf.missing:
          e_cnt = e_cnt + 1
          m_lat = m_lat + ens_lat[n]
          m_lon = m_lon + ens_lon[n]

    if e_cnt > 0:
       m_lon = m_lon / e_cnt
       m_lat = m_lat / e_cnt

    for n in range(nens):
       if ens_lat[n] == atcf.missing or ens_lon[n] == atcf.missing:
          ens_lat[n] = m_lat
          ens_lon[n] = m_lon

    #  Read grib file information for this forecast hour
    g1 = dpp.ReadGribFiles(datea, fhr, config)

    dencode = {'ensemble_data': {'dtype': 'float32'}, 'latitude': {'dtype': 'float32'},
               'longitude': {'dtype': 'float32'}, 'ensemble': {'dtype': 'int32'}}

    #  Compute steering wind components
    uoutfile='{0}/{1}_f{2}_usteer_ens.nc'.format(config['work_dir'],datea,fff)
    voutfile='{0}/{1}_f{2}_vsteer_ens.nc'.format(config['work_dir'],datea,fff)
    stfnfile='{0}/{1}_f{2}_ssteer_ens.nc'.format(config['work_dir'],datea,fff)
    vortfile='{0}/{1}_f{2}_csteer_ens.nc'.format(config['work_dir'],datea,fff)
    if (not os.path.isfile(uoutfile) or not os.path.isfile(voutfile)) and config['fields'].get('calc_uvsteer','True') == 'True':

       logging.warning("  Computing steering wind information")

       inpDict = {'isobaricInhPa': (steerp1, steerp2)}
       inpDict = g1.set_var_bounds('zonal_wind', inpDict)

       #  Create output arrays
       outDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 
                  'description': 'zonal steering wind', 'units': 'm/s', '_FillValue': -9999.}
       outDict = g1.set_var_bounds('zonal_wind', outDict)
       uensmat = g1.create_ens_array('zonal_wind', nens, outDict)          

       outDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                  'description': 'meridional steering wind', 'units': 'm/s', '_FillValue': -9999.}
       outDict = g1.set_var_bounds('meridional_wind', outDict)
       vensmat = g1.create_ens_array('meridional_wind', nens, outDict)       

       outDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                  'description': 'steering wind streamfunction', 'units': 'm2/s', '_FillValue': -9999.}
       outDict = g1.set_var_bounds('meridional_wind', outDict)
       sensmat = g1.create_ens_array('meridional_wind', nens, outDict)

       outDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                  'description': 'steering wind vorticity', 'units': '1/s', '_FillValue': -9999.}
       outDict = g1.set_var_bounds('zonal_wind', outDict)
       vortmat = g1.create_ens_array('zonal_wind', nens, outDict)

       wencode = {'latitude': {'dtype': 'float32'}, 'longitude': {'dtype': 'float32'}}

       for n in range(nens):

          #  Read global zonal and meridional wind, write to file
          uwnd = g1.read_grib_field('zonal_wind', n, inpDict).rename('u')
          vwnd = g1.read_grib_field('meridional_wind', n, inpDict).rename('v')

#          latvec = uwnd.latitude.values
#          lonvec = uwnd.longitude.values

          if e_cnt > 0:

#             latcen = latvec[np.abs(latvec-ens_lat[n]).argmin()]
#             loncen = lonvec[np.abs(lonvec-ens_lon[n]).argmin()]

             uwnd, vwnd = surgery.remove_TC_circulation(uwnd, vwnd, (ens_lat[n], ens_lon[n]), tcradius)

          #  Integrate the winds over the layer to obtain the steering wind
          pres,lat,lon = uwnd.indexes.values()
          nlev      = len(pres)

          uint      = uwnd[0,:,:]
          uint[:,:] = 0.0
          vint      = vwnd[0,:,:]
          vint[:,:] = 0.0

          for k in range(nlev-1):

             uint[:,:] = uint[:,:] + 0.5 * (uwnd[k,:,:]+uwnd[k+1,:,:]) * abs(pres[k+1]-pres[k])
             vint[:,:] = vint[:,:] + 0.5 * (vwnd[k,:,:]+vwnd[k+1,:,:]) * abs(pres[k+1]-pres[k])

#             if pres[0] > pres[-1]:
#               uint = -np.trapz(uwnd[:,:,:], pres, axis=0) / abs(pres[-1]-pres[0])
#               vint = -np.trapz(vwnd[:,:,:], pres, axis=0) / abs(pres[-1]-pres[0])
#             else:
#               uint = np.trapz(uwnd[:,:,:], pres, axis=0) / abs(pres[-1]-pres[0])
#               vint = np.trapz(vwnd[:,:,:], pres, axis=0) / abs(pres[-1]-pres[0])

          if lat[0] > lat[-1]:
             slat1 = lat2
             slat2 = lat1
          else:
             slat1 = lat1
             slat2 = lat2

          #  Write steering flow to ensemble arrays
          usteer = np.squeeze(uint.sel(latitude=slice(slat1, slat2), longitude=slice(lon1, lon2))) / abs(pres[-1]-pres[0])
          vsteer = np.squeeze(vint.sel(latitude=slice(slat1, slat2), longitude=slice(lon1, lon2))) / abs(pres[-1]-pres[0])
          uensmat[n,:,:] = usteer[:,:]
          vensmat[n,:,:] = vsteer[:,:]

          #  Compute the streamfunction of the steering wind
          psi = surgery.wind_streamfunction(np.squeeze(uint) / abs(pres[-1]-pres[0]), np.squeeze(vint) / abs(pres[-1]-pres[0]))
          psi[:,:] = psi[:,:] - np.mean(psi)
          sensmat[n,:,:] = np.squeeze(psi.sel(latitude=slice(slat1, slat2), longitude=slice(lon1, lon2))) * 1.0e-6

          #  Compute the vorticity associated with the steering wind
          lat  = usteer.latitude.values
          lon  = usteer.longitude.values
          dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat, x_dim=-1, y_dim=-2, geod=None)
          vort = mpcalc.vorticity(usteer * units('m / sec'), vsteer * units('m / sec'), dx=dx, dy=dy)
          vortmat[n,:,:] = grid_calc.calc_circ_llgrid(vort, 500., lat, lon, False, len(lon), len(lat)) * 1.0e5

       uensmat.to_netcdf(uoutfile, encoding=dencode)
       vensmat.to_netcdf(voutfile, encoding=dencode) 
       sensmat.to_netcdf(stfnfile, encoding=dencode)
       vortmat.to_netcdf(vortfile, encoding=dencode)

    else:

       logging.warning("  Obtaining steering wind information from file")


    #  Read geopotential height from file, if ensemble file is not present
    if config['fields'].get('calc_height','True') == 'True':

       if 'height_levels' in config['fields']:
          height_list = json.loads(config['fields'].get('height_levels'))
       else:
          height_list = [500]

       for level in height_list:

          levstr = '%0.3i' % int(level)
          outfile='{0}/{1}_f{2}_h{3}hPa_ens.nc'.format(config['work_dir'],datea,fff,levstr)

          if not os.path.isfile(outfile):

             logging.warning('  Computing {0} hPa height'.format(levstr))

             vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level),
                      'description': '{0} hPa height'.format(levstr), 'units': 'm', '_FillValue': -9999.}
             vDict = g1.set_var_bounds('geopotential_height', vDict)
             ensmat = g1.create_ens_array('geopotential_height', g1.nens, vDict)

             for n in range(g1.nens):
                ensmat[n,:,:] = np.squeeze(g1.read_grib_field('geopotential_height', n, vDict))

             ensmat.to_netcdf(outfile, encoding=dencode)

          elif os.path.isfile(outfile):

             logging.warning("  Obtaining {0} hPa height data from {1}".format(levstr,outfile))


    #  Compute 250 hPa PV if the file does not exist
    outfile='{0}/{1}_f{2}_pv250_ens.nc'.format(config['work_dir'],datea,fff)
    if (not os.path.isfile(outfile) and config['fields'].get('calc_pv250hPa','True') == 'True'):

       logging.warning("  Computing 250 hPa PV")

       vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (200, 300),
                'description': '250 hPa Potential Vorticity', 'units': 'PVU', '_FillValue': -9999.}
       vDict = g1.set_var_bounds('zonal_wind', vDict)

       ensmat = g1.create_ens_array('zonal_wind', nens, vDict)

       for n in range(nens):

          #  Read all the necessary files from file, smooth fields, so sensitivities are useful
          tmpk = g1.read_grib_field('temperature', n, vDict) * units('K')

          lats = tmpk.latitude.values * units('degrees')
          lons = tmpk.longitude.values * units('degrees')
          pres = tmpk.isobaricInhPa.values * units('hPa')

          tmpk = mpcalc.smooth_n_point(tmpk, 9, 4)

          thta = mpcalc.potential_temperature(pres[:, None, None], tmpk)

          uwnd = g1.read_grib_field('zonal_wind', n, vDict) * units('m/s')
          vwnd = g1.read_grib_field('meridional_wind', n, vDict) * units('m/s')

          dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats, x_dim=-1, y_dim=-2, geod=None)

          #  Compute PV and place in ensemble array
          pvout = mpcalc.potential_vorticity_baroclinic(thta, pres[:, None, None], uwnd, vwnd,
                                           dx[None, :, :], dy[None, :, :], lats[None, :, None])

          ensmat[n,:,:] = grid_calc.calc_circ_llgrid(pvout[np.where(pres == 250 * units('hPa'))[0],:,:], \
                                                     300., lats, lons, False, len(lons), len(lats)) * 1.0e6
 
       ensmat.to_netcdf(outfile, encoding=dencode)

    elif os.path.isfile(outfile):

       logging.warning("  Obtaining 250 hPa PV data from {0}".format(outfile))


    #  Compute the equivalent potential temperature (if desired and file is missing)
    if config['fields'].get('calc_theta-e','False') == 'True':

       if 'theta-e_levels' in config['fields']:
          thetae_list = json.loads(config['fields'].get('theta-e_levels'))
       else:
          thetae_list = [700, 850]

       for level in thetae_list:

          levstr = '%0.3i' % int(level)
          outfile='{0}/{1}_f{2}_e{3}hPa_ens.nc'.format(config['work_dir'],datea,fff,levstr)

          if not os.path.isfile(outfile):

             logging.warning('  Computing {0} hPa Theta-E'.format(levstr))

             vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level),
                      'description': '{0} hPa Equivalent Potential Temperature'.format(levstr), 'units': 'K', '_FillValue': -9999.}
             vDict = g1.set_var_bounds('temperature', vDict)

             ensmat = g1.create_ens_array('temperature', g1.nens, vDict)

             for n in range(g1.nens):

                tmpk = g1.read_grib_field('temperature', n, vDict) * units.K
                pres = tmpk.isobaricInhPa.values * units.hPa

                if g1.has_specific_humidity:
                   qvap = np.squeeze(g1.read_grib_field('specific_humidity', n, vDict))
                   tdew = mpcalc.dewpoint_from_specific_humidity(pres[None, None], tmpk, qvap)
                else:
                   relh = g1.read_grib_field('relative_humidity', n, vDict)
                   relh = np.minimum(np.maximum(relh, 0.01), 100.0) * units.percent
                   tdew = mpcalc.dewpoint_from_relative_humidity(tmpk, relh)

                ensmat[n,:,:] = np.squeeze(mpcalc.equivalent_potential_temperature(pres[None, None], tmpk, tdew))

             ensmat.to_netcdf(outfile, encoding=dencode)

          elif os.path.isfile(outfile):

             logging.warning("  Obtaining {0} hPa Theta-e data from {1}".format(levstr,outfile))


    #  Compute the 500-850 hPa water vapor mixing ratio (if desired and file is missing)
    outfile='{0}/{1}_f{2}_q500-850hPa_ens.nc'.format(config['work_dir'],datea,fff)
    if (not os.path.isfile(outfile) and config['fields'].get('calc_q500-850hPa','False') == 'True'):

       logging.warning("  Computing 500-850 hPa Water Vapor")

       vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                'description': '500-850 hPa Integrated Water Vapor', 'units': 'hPa', '_FillValue': -9999.}
       vDict = g1.set_var_bounds('temperature', vDict)

       ensmat = g1.create_ens_array('temperature', nens, vDict)

       vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (500, 850),
                'description': '500-850 hPa Integrated Water Vapor', 'units': 'hPa', '_FillValue': -9999.}
       vDict = g1.set_var_bounds('temperature', vDict)

       for n in range(nens):

          tmpk = np.squeeze(g1.read_grib_field('temperature', n, vDict)) * units('K')
          pres = (tmpk.isobaricInhPa.values * units.hPa).to(units.Pa)

          if g1.has_specific_humidity:
             qvap = mpcalc.mixing_ratio_from_specific_humidity(g1.read_grib_field('specific_humidity', n, vDict))
          else:
             relh = np.minimum(np.maximum(g1.read_grib_field('relative_humidity', n, vDict), 0.01), 100.0) * units('percent')
             qvap = mpcalc.mixing_ratio_from_relative_humidity(pres[:,None,None], tmpk, relh)

          #  Integrate water vapor over the pressure levels
          ensmat[n,:,:] = np.abs(np.trapz(qvap, pres, axis=0)) / mpcon.earth_gravity

       ensmat.to_netcdf(outfile, encoding=dencode)

    elif os.path.isfile(outfile):

       logging.warning("  Obtaining 500-850 hPa water vapor data from {0}".format(outfile))


    #  Compute the IVT (if desired and file is missing)
    outfile='{0}/{1}_f{2}_ivt_ens.nc'.format(config['work_dir'],datea,fff)
    if (not os.path.isfile(outfile) and config['fields'].get('calc_ivt','True') == 'True'):

       logging.warning("  Computing IVT")

       vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                'description': 'Integrated Water Vapor Transport', 'units': 'kg s-1', '_FillValue': -9999.}
       vDict = g1.set_var_bounds('temperature', vDict)

       ensmat = g1.create_ens_array('temperature', g1.nens, vDict)

       if 'ivt' in g1.var_dict:

          for n in range(nens):
             ensmat[n,:,:] = g1.read_grib_field('ivt', n, vDict)

       else:

          vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (400, 1000),
                   'description': 'Integrated Water Vapor Transport', 'units': 'hPa', '_FillValue': -9999.}
          vDict = g1.set_var_bounds('temperature', vDict)

          for n in range(g1.nens):

             #  Obtain the wind speeds
             uwnd = g1.read_grib_field('zonal_wind', n, vDict)
             vwnd = g1.read_grib_field('meridional_wind', n, vDict)
             uwnd[:,:,:] = np.sqrt(uwnd[:,:,:]**2 + vwnd[:,:,:]**2) * units('m / sec')

             #  Compute the mixing ratio
             pres = (uwnd.isobaricInhPa.values * units.hPa).to(units.Pa)

             if g1.has_specific_humidity:
                qvap = g1.read_grib_field('specific_humidity', n, vDict) * units('dimensionless')
             else:
                tmpk = g1.read_grib_field('temperature', n, vDict) * units('K')
                relh = np.minimum(np.maximum(g1.read_grib_field('relative_humidity', n, vDict), 0.01), 100.0) * units('percent')
                qvap = mpcalc.mixing_ratio_from_relative_humidity(pres[:,None,None], tmpk, relh)
                del tmpk, relh

             #  Integrate water vapor over the pressure levels
             ensmat[n,:,:] = np.abs(np.trapz(uwnd[:,:,:]*qvap[:,:,:], pres, axis=0)) / mpcon.earth_gravity
             del uwnd,vwnd,qvap,pres

       ensmat.to_netcdf(outfile, encoding=dencode)
       del ensmat


    #  Compute wind-related forecast fields (if desired and file is missing)
    if config['fields'].get('calc_winds','False') == 'True':

       if 'wind_levels' in config['fields']:
          wind_list = json.loads(config['fields'].get('wind_levels'))
       else:
          wind_list = [850]

       for level in wind_list:

          levstr = '%0.3i' % int(level)
          ufile='{0}/{1}_f{2}_u{3}hPa_ens.nc'.format(config['work_dir'],datea,fff,levstr)
          vfile='{0}/{1}_f{2}_v{3}hPa_ens.nc'.format(config['work_dir'],datea,fff,levstr)

          if (not os.path.isfile(ufile)) or (not os.path.isfile(vfile)):

             logging.warning('  Computing {0} hPa wind information'.format(levstr))

             uDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level),
                      'description': '{0} hPa zonal wind'.format(levstr), 'units': 'm/s', '_FillValue': -9999.}
             uDict = g1.set_var_bounds('zonal_wind', uDict)

             uensmat = g1.create_ens_array('zonal_wind', g1.nens, uDict)

             vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level),
                      'description': '{0} hPa meridional wind'.format(levstr), 'units': 'm/s', '_FillValue': -9999.}
             vDict = g1.set_var_bounds('meridional_wind', vDict)

             vensmat = g1.create_ens_array('meridional_wind', g1.nens, vDict)

             for n in range(g1.nens):

                uwnd = g1.read_grib_field('zonal_wind', n, uDict).squeeze()
                vwnd = g1.read_grib_field('meridional_wind', n, vDict).squeeze()

                uensmat[n,:,:] = uwnd[:,:]
                vensmat[n,:,:] = vwnd[:,:]

             uensmat.to_netcdf(ufile, encoding=dencode)
             vensmat.to_netcdf(vfile, encoding=dencode)

          elif os.path.isfile(outfile):

             logging.warning("  Obtaining {0} hPa wind information from file".format(levstr))


    #  Compute vorticity forecast fields (if desired and file is missing)
    if config['fields'].get('calc_vorticity','False') == 'True':

       if 'vorticity_levels' in config['fields']:
          vor_list = json.loads(config['fields'].get('vorticity_levels'))
       else:
          vor_list = [850]

       vortrad = float(config['fields'].get('vorticity_radius','100'))

       for level in vor_list:

          levstr = '%0.3i' % int(level)
          outfile='{0}/{1}_f{2}_vor{3}hPa_ens.nc'.format(config['work_dir'],datea,fff,levstr)

          if not os.path.isfile(outfile):

             logging.warning('  Computing {0} hPa vorticity information'.format(levstr))

             vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level),
                      'description': '{0} hPa vorticity'.format(levstr), 'units': '1/s', '_FillValue': -9999.}
             vDict = g1.set_var_bounds('zonal_wind', vDict)

             ensmat = g1.create_ens_array('zonal_wind', g1.nens, vDict)

             for n in range(g1.nens):

                uwnd = g1.read_grib_field('zonal_wind', n, vDict).squeeze() * units('m/s')
                vwnd = g1.read_grib_field('meridional_wind', n, vDict).squeeze() * units('m/s')

                lat  = ensmat.latitude.values
                lon  = ensmat.longitude.values
                dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat, x_dim=-1, y_dim=-2, geod=None)
                div = mpcalc.divergence(uwnd, vwnd, dx=dx, dy=dy)
                ensmat[n,:,:] = grid_calc.calc_circ_llgrid(div, vortrad, lat, lon, eval(config['fields'].get('global','False')), \
                                                           len(lon), len(lat)) * 1.0e5

             ensmat.to_netcdf(outfile, encoding=dencode)
             del ensmat

          elif os.path.isfile(outfile):

             logging.warning("  Obtaining {0} hPa vorticity information from file".format(levstr))


if __name__ == "__main__":
    ComputeTrackFields('2019082900', 12)

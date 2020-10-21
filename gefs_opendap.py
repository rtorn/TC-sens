import os
import time
import shutil
import gzip
import sys
import netCDF4 as nc
import urllib
import cfgrib
import datetime as dt
import glob
import pandas as pd
import numpy as np
import xarray as xr

def stage_grib_files(datea, config):

    '''
    This is a generic class for copying or linking grib file data from a specific location
    to a directory where calculations are performed.  No matter where these calculations are
    carried out, this routine must exist.

    This particular instance employs GEFS OpeNDAP data that is on NCEP servers.  As a 
    consequence, this routine is a placeholder and does not need to do anything.

    Attributes:
        datea (string):  The initialization time of the forecast (yyyymmddhh)
        config  (dict):  The dictionary with configuration information
    '''

    #  Make the work directory if it does not exist
    if not os.path.isdir(config['work_dir']):
       try:
          os.makedirs(config['work_dir'])
       except OSError as e:
          raise e


def stage_atcf_files(datea, bbnnyyyy, config):
    '''
    This is a generic class for copying or linking ATCF file data from a specific location
    to a directory where calculations are performed.  No matter where these calculations are
    carried out, this routine must exist.

    The result is a set of ATCF files in the work directory of the format atcf_NN.dat,
    where NN is the ensemble member number.

    This particular instance is for GEFS data that is on the NHC FTP server. In this
    case, all ATCF data for a particular storm is in one file, so the code unzips and
    reads the storm file, then uses sed to get the lines attributed to each
    ensemble member and places that data in a seperate file.

    Attributes:
        datea (string):  The initialization time of the forecast (yyyymmddhh)
        config  (dict):  The dictionary with configuration information
    '''

    src  = config['atcf_dir'] + "/a" + bbnnyyyy + ".dat.gz"
    nens = int(config['num_ens'])

    #  Unzip the file from the NHC server, write the file to the work directory
    gzfile = gzip.GzipFile(fileobj=urllib.request.urlopen(src))
    uzfile = open(config['work_dir'] + '/a' + bbnnyyyy + '.dat', 'wb')
    uzfile.write(gzfile.read())        
    gzfile.close()
    uzfile.close()

    #  Wait for the ensemble ATCF information to be placed in the file
#    while ( len(os.popen("sed -ne /" + self.init + "/p " + self.dest + "/a" + bbnnyyyy + ".dat | sed -ne /EE/p").read()) == 0 ):
#       time.sleep(20.7)

    #  Wait for the file to be finished being copied
#    while ( (time.time() - os.path.getmtime(self.src)) < 60 ):
#       time.sleep(10)

    for n in range(nens + 1):

       if ( n > 0 ):
          modid = 'AP'
       else:
          modid = 'AC'

       nn = '%0.2i' % n
       file_name = config['work_dir'] + "/atcf_" + nn + ".dat"

       #  If the specific member's ATCF file does not exist, copy from the source file with sed.
       if not os.path.isfile(file_name):

          fo = open(file_name,"w")
          fo.write(os.popen("sed -ne /" + datea + "/p " + config['work_dir'] + "/a" + bbnnyyyy + ".dat | sed -ne /" + modid + nn + "/p").read())
          fo.close()

 
#  Class to read information from ensemble grib files
class ReadGribFiles:
    def __init__(self, datea, fhr, config):
        self.datea_str = datea
        self.src_path = config['model_dir']
        self.datea = dt.datetime.strptime(datea, '%Y%m%d%H')
        self.datea_s = self.datea.strftime("%m%d%H%M")
        self.datef = self.datea + dt.timedelta(hours=fhr)
        self.hhh     = '%0.3i' % fhr
        self.fday    = int(round(fhr / 24.0))

        self.ds_dict = self.__read_grib_dict()

        self.ds_dict.coords['lon'] = (self.ds_dict.coords['lon'] + 180) % 360 - 180

        self.var_dict = {'zonal_wind': 'ugrdprs', 'meridional_wind': 'vgrdprs', 'geopotential_height': 'hgtprs', 'temperature': 'tmpprs', 
                         'relative_humidity': 'rhprs', 'sea_level_pressure': 'prmslmsl'}


    ###  Function that creates a dictionary for each ensemble gribfile
    def __read_grib_dict(self):

        file_name = self.__create_file_name(0)
        ds_dict = xr.open_dataset(file_name)

        return ds_dict

    ###  Function to read an ensemble array of fields based on input information in vdict
    def create_ens_array(self, varname, nens, vdict):

       #  Determine latitude bounds.  If values are in vdict, use, otherwise, use entire array
       if 'latitude' in vdict:

          if float(self.ds_dict[self.var_dict[varname]].lat.data[0]) > float(self.ds_dict[self.var_dict[varname]].lat.data[-1]):
             slat1 = int(vdict['latitude'][1])
             slat2 = int(vdict['latitude'][0])
          else:
             slat1 = int(vdict['latitude'][0])
             slat2 = int(vdict['latitude'][1])

       else:

          latvec = list(self.ds_dict[self.var_dict[varname]].lat.data)
          slat1  = latvec[0]
          slat2  = latvec[-1]

       #  Determine longitude bounds.  If values are in vdict, use, otherwise, use entire array
       if 'longitude' in vdict:

          lon1 = int(vdict['longitude'][0])
          lon2 = int(vdict['longitude'][1])

       else:

          lonvec = list(self.ds_dict[self.var_dict[varname]].lon.data)
          lon1   = lonvec[0]
          lon2   = lonvec[-1]

       #  Create attributes based on what is in the file
       attrlist = {}
       if 'description' in vdict:
         attrlist['description'] = vdict['description']
       if 'units' in vdict:
         attrlist['units'] = vdict['units']
       if '_FillValue' in vdict:
         attrlist['_FillValue'] = vdict['_FillValue']

       #  Create a dummy array that can be used to copy data into
       lonvec = list(self.ds_dict[self.var_dict[varname]].sel(lon=slice(lon1, lon2)).lon.data)
       latvec = list(self.ds_dict[self.var_dict[varname]].sel(lat=slice(slat1, slat2)).lat.data)
       ensarr = xr.DataArray(name='ensemble_data', data=np.zeros([nens, len(latvec), len(lonvec)]),
                             dims=['ensemble', 'latitude', 'longitude'], attrs=attrlist, 
                             coords={'ensemble': [i for i in range(nens)], 'latitude': latvec, 'longitude': lonvec}) 

       return(ensarr)

    #  Function to read a single ensemble member's forecast field
    def read_grib_field(self, varname, member, vdict):

       vname = self.var_dict[varname]

       if 'latitude' in vdict:

          if float(self.ds_dict[vname].lat.data[0]) > float(self.ds_dict[vname].lat.data[-1]):
             slat1 = int(vdict['latitude'][1])
             slat2 = int(vdict['latitude'][0])
          else:
             slat1 = int(vdict['latitude'][0])
             slat2 = int(vdict['latitude'][1])

       else:

          latvec = list(self.ds_dict[vname].lat.data)
          slat1  = latvec[0]
          slat2  = latvec[-1]

       if 'longitude' in vdict:

          lon1 = int(vdict['longitude'][0])
          lon2 = int(vdict['longitude'][1])

       else:

          lonvec = list(self.ds_dict[vname].lon.data)
          lon1   = lonvec[0]
          lon2   = lonvec[-1]

       #  Read a single pressure level of data, if this is a variable that has pressure levels
       if 'isobaricInhPa' in vdict:

          if float(self.ds_dict[vname].lev.data[0]) > float(self.ds_dict[vname].lev.data[-1]):
            slev1 = int(vdict['isobaricInhPa'][1])
            slev2 = int(vdict['isobaricInhPa'][0])
          else:
            slev1 = int(vdict['isobaricInhPa'][0])
            slev2 = int(vdict['isobaricInhPa'][1])

          vout  = self.ds_dict[vname].sel(lat=slice(slat1, slat2), lon=slice(lon1, lon2), lev=slice(slev1, slev2), ens=slice(member+1, member+1), \
                                time=slice(self.datef, self.datef)).squeeze().rename({'lon': 'longitude','lat': 'latitude','lev': 'isobaricInhPa'})

          atest = self.ds_dict[vname].sel(lat=slice(slat1, slat2), lon=slice(lon1, lon2), ens=slice(member+1, member+1), time=slice(self.datef, self.datef)).squeeze()

       #  Read the only level if it is a single level variable
       else:

          vout  = self.ds_dict[vname].sel(lat=slice(slat1, slat2), lon=slice(lon1, lon2), ens=slice(member+1, member+1), \
                                time=slice(self.datef, self.datef)).squeeze().rename({'lon': 'longitude','lat': 'latitude'})

       return(vout)

    def __create_file_name(self,member):

        if ( member > 0 ):
            modid = 'p'
        else:
            modid = 'c'
        nn = '%0.2i' % member
#        return self.src_path + '/gefs' + self.datea_str[0:8] + '/ge' + modid + nn + '_' + self.datea_str[8:10] + 'z_pgrb2a'
        return self.src_path + '/gefs' + self.datea_str[0:8] + '/gefs_pgrb2ap5_all_' + self.datea_str[8:10] + 'z'

    @staticmethod
    def __check_file_exists(filename):
        isfile = False
        try:
            if os.path.isfile(filename):
                isfile = True
        except Exception as err:
            print(err)

        return isfile

if __name__ == '__main__':

    src1 = "/Users/parthpatwari/RA_Atmospheric_Science/Old_Code/atcf_data"
    grib_src = "/Users/parthpatwari/RA_Atmospheric_Science/GRIB_files"
    dest1 = "/Users/parthpatwari/RA_Atmospheric_Science/New_Code/atcf_data"
    atcf_src = "/Users/parthpatwari/RA_Atmospheric_Science/Old_Code/atcf_data"
    # c1 = CopyFiles(src, dest)
    # if c1.checkandcreatedir():
    #     c1.copy_filestowork()
    g1 = ReadGribFiles(grib_src, '2019082900', 180)
    print(g1.grib_dict)
    a1 = Readatcfdata(atcf_src)
    print(a1.atcf_array)

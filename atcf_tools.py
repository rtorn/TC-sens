import os
import sys
import datetime as dt
import glob
import pandas as pd
import numpy as np


def lat_str_to_float(latstr):
    '''
    Function that converts an ATCF-format latitude value into a -90 to 90 
    degree latitude float.

    Attributes:
        latstr (string):  ATCF-format latitude string
    '''

    if latstr[-1] == 'N':
       return float(latstr[0:-1]) * 0.1
    else:
       return -float(latstr[0:-1]) * 0.1


def lon_str_to_float(lonstr):
    '''
    Function that converts an ATCF-format longitude value into a -180 to 180 
    degree longitude float.

    Attributes:
        lonstr (string):  ATCF-format longitude string
    '''

    if lonstr[-1] == 'W':
       return -float(lonstr[0:-1]) * 0.1
    else:
       return float(lonstr[0:-1]) * 0.1



class ReadATCFData:
    '''
    Class that reads ATCF-format data from an ensemble of files (assumes that each file contains one
    ensemble member at one initialization time.  The class itself contains a series of dictionaries that
    contain the TC track and intensity information.  The data is saved both with pandas and using a 
    dictionary.  The data can be accessed through a series of routines.

    This class is currently under development, especially the best track portions

    Attributes:
        infiles (string): list of ATCF ensemble member files

    '''

    def __init__(self, infiles, **kwargs):

        self.cols = ['basin', 'tcnum', 'datea', 'mm', 'ftype', 'fhr', 'lat', 'lon', 'wnd', 'mslp', 'stype',\
                     'rval', 'ord', 'rad1', 'rad2', 'rad3', 'rad4', 'pouter', 'router', 'rmw', 'gusts', 'eye', \
                     'subregion', 'maxseas', 'initials', 'dir', 'speed', 'stormname', 'depth', 'seas', \
                     'seascode', 'seas1', 'seas2', 'seas3', 'seas4', 'user1', 'user2', 'user3', 'user4', 'user5']

        self.ens_atcf = {}
        self.missing = -9999.
        atcf_files = sorted(glob.glob(infiles))
        self.num_atcf_files = len(atcf_files)
        self.ens_name = []
        datea = "YYYYMMDDHH"
        fctid = "XXXX"

        if self.num_atcf_files > 1:

           for f in range(self.num_atcf_files):

              with open(atcf_files[f], 'r') as fo:
                 data = np.array(fo.readlines())
                 total_rows = data.shape[0]
                 if total_rows == 0:
                    n_lines = 0
                 else:
                    n_lines = total_rows
                    n_cols = len(data[0])
              fo.close()

              mname = 'mem_{0}'.format('%0.3i' % f)

              if n_lines > 0:
                 filedb = pd.read_csv(filepath_or_buffer=atcf_files[f], header=None)
                 filedb.columns = self.cols[0:len(filedb.columns)]
                 self.ens_atcf.update({mname: filedb})
              else:
                 filedb = pd.DataFrame(columns = self.cols[0:10])
                 self.ens_atcf.update({mname: filedb})

              self.ens_name.append(mname)

        else:

           with open(infiles, 'r') as fo:
              data = np.array(fo.readlines())
              total_rows = data.shape[0]
              if total_rows == 0:
                 n_lines = 0
              else:
                 n_lines = total_rows
                 n_cols = np.min([str(data[0]).count(','), 20])
           fo.close()

           if n_lines > 0:
              filedb = pd.read_csv(filepath_or_buffer=infiles, header=None, index_col=False, usecols=range(n_cols))
              filedb.columns = self.cols[0:len(filedb.columns)]

              for f in range(len(kwargs['fcstid'])):

                 reddb = filedb.loc[filedb['ftype'].str.strip() == kwargs['fcstid'][f]]
                 if 'init' in kwargs:
                    reddb = reddb.loc[reddb['datea'] == int(kwargs['init'])]

                 self.ens_atcf.update({kwargs['fcstid'][f]: reddb})
                 self.ens_name.append(kwargs['fcstid'][f])

           else: 
              filedb = pd.DataFrame(columns = self.cols[0:10])
              filedb.columns = self.cols[0:len(filedb.columns)]

              for f in range(len(kwargs['fcstid'])):
                 self.ens_atcf.update({kwargs['fcstid'][f]: filedb})
                 self.ens_name.append(kwargs['fcstid'][f])
 
#        self.atcf_array.get(f).update({'initialization_time': datea})


    def ens_lat_lon_time(self, fhr):
        '''
        Function that returns all ensemble member's latitude and longitude for a given 
        forecast hour.  The result is two vectors, one with the latitude and one with
        the ensemble TC longitude.
      
        Attributes:
            fhr (int):  forecast hour
        '''

        ens_lat = np.ones(len(self.ens_name)) * self.missing
        ens_lon = np.ones(len(self.ens_name)) * self.missing
        for n in range(len(self.ens_name)):
           dfalt = self.ens_atcf.get(self.ens_name[n]).loc[self.ens_atcf.get(self.ens_name[n])['fhr'] == float(fhr)].reset_index()         

           if not dfalt.empty:
              ens_lat[n] = lat_str_to_float(dfalt['lat'][0])
              ens_lon[n] = lon_str_to_float(dfalt['lon'][0])

        return ens_lat, ens_lon


    def ens_intensity_time(self, fhr):
        '''
        Function that returns all ensemble member's minimum sea-level pressure and
        maximum wind speed for a given forecast hour.  The result is two vector arrays, 
        one with the ensemble SLP, and the other with the ensemble maximum wind.
      
        Attributes:
            fhr (int):  forecast hour
        '''

        ens_slp = np.ones(len(self.ens_name)) * self.missing
        ens_wnd = np.ones(len(self.ens_name)) * self.missing
        for n in range(len(self.ens_name)):
           dfalt = self.ens_atcf.get(self.ens_name[n]).loc[self.ens_atcf.get(self.ens_name[n])['fhr'] == float(fhr)].reset_index()

           if not dfalt.empty:
              ens_slp[n] = float(dfalt['mslp'].values[0])
              ens_wnd[n] = float(dfalt['wnd'].values[0])

        return ens_slp, ens_wnd


    def fcst_init_times(self, fcstid):

        dfalt = self.ens_atcf.get(fcstid).loc[self.ens_atcf.get(fcstid)['fhr'] == float(0)].reset_index()

        init_times = sorted([*set(dfalt['datea'].values.astype(str))])
       
        return init_times 


class ReadBestData:
    '''
    Class that reads ATCF-format data from an ensemble of files (assumes that each file contains one
    ensemble member at one initialization time.  The class itself contains a series of dictionaries that
    contain the TC track and intensity information.  The data is saved both with pandas and using a 
    dictionary.  The data can be accessed through a series of routines.

    This class is currently under development, especially the best track portions

    Attributes:
        infiles (string): list of ATCF ensemble member files

    '''

    def __init__(self, bestfile, **kwargs):
        '''
        Function that reads the best track data from the specified best track file 
        and saves the information into a pandas database that can be accessed through the
        appropriate routines.
      
        Attributes:
            bestfile (string):  best track file to read
        '''

        self.cols = ['basin', 'tcnum', 'datea', 'mm', 'ftype', 'fhr', 'lat', 'lon', 'wnd', 'mslp', 'stype',\
                     'rval', 'ord', 'rad1', 'rad2', 'rad3', 'rad4', 'pouter', 'router', 'rmw', 'gusts', 'eye', \
                     'subregion', 'maxseas', 'initials', 'dir', 'speed', 'stormname', 'depth', 'seas', \
                     'seascode', 'seas1', 'seas2', 'seas3', 'seas4', 'user1', 'user2', 'user3', 'user4', 'user5']
        self.missing = -9999.

        try:
          self.bestdf = pd.read_csv(filepath_or_buffer=bestfile, header=None, usecols=range(11))
          self.bestdf.columns = self.cols[0:len(self.bestdf.columns)]
          self.has_best = True
        except:
          print("{0} not found".format(bestfile))
          self.has_best = False

    def best_fix_times(self, tconly):

        dlist = self.bestdf['datea'].unique()
        dateout = []

        if tconly:
           for date in dlist:
              if self.best_is_NHC_tc(date):
                 dateout.append(str(date))
        else:
           dateout = dlist

        return(dateout)


    def best_vitals_time(self, datea):
        '''
        Function that returns the TC position, maximum wind and minimum SLP for a single 
        time based on the information in the pandas database.  If there is no data for that
        time, returns missing values.

        Attributes:
            datea (string):  date to obtain best track information for
        '''

        try:
 
           #  Grab subset of database for this time 
           dfalt = self.bestdf.loc[self.bestdf['datea'] == float(datea)].reset_index()

           #  Compute the latitude/longitude
           lat = lat_str_to_float(dfalt['lat'][0])
           lon = lon_str_to_float(dfalt['lon'][0])

           return lat, lon, float(dfalt['wnd'][0]), float(dfalt['mslp'][0])

        except:

           return self.missing, self.missing, self.missing, self.missing

    def best_lat_lon_time(self, datea):
        '''
        Function that returns the TC position for a single time based on the information in
        the pandas database.  If there is no data for that time, returns missing values.
      
        Attributes:
            datea (string):  date to obtain best track information for
        '''

        try:

           #  Grab subset of database for this time 
           dfalt = self.bestdf.loc[self.bestdf['datea'] == float(datea)].reset_index()

           #  Compute the latitude/longitude
           lat = lat_str_to_float(dfalt['lat'][0])
           lon = lon_str_to_float(dfalt['lon'][0])

           return lat, lon

        except:

           return self.missing, self.missing


    def best_intensity_time(self, datea):
        '''
        Function that returns the TC min. SLP and max. wind for a single time based on the 
        information in the pandas database.  If there is no data for that time, returns missing values.     
 
        Attributes:
            datea (string):  date to obtain best track information for
        '''

        try:

           #  Grab subset of database for this time 
           dfalt = self.bestdf.loc[self.bestdf['datea'] == float(datea)].reset_index()

           return float(dfalt['wnd'][0]), float(dfalt['mslp'][0])

        except:

           return self.missing, self.missing

    def best_is_NHC_tc(self, datea):

       try:

          stype = self.bestdf.loc[self.bestdf['datea'] == int(datea)]['stype'].values[0].strip()
          return ( stype == 'TD' or stype == 'SD' or stype == 'TS' or
                   stype == 'SS' or stype == 'HU' or stype == 'TY' )

       except:

          return False


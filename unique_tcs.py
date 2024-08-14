import os
import argparse
import pandas as pd
import numpy as np

def unique_tcs_from_vitals(vitalfile, yyyymmddhh):
  '''
  This function reads a tcvitals format file, finds all of the unique TCs that are present for
  a given time, and returns the list of TC names.  This could be called from another python 
  program, or from the command line using main below.

  Attributes:
      vitalfile (string):   name of the tcvital plot to read
      yyyymmddhh (string):  date string of requested analysis time (yyyymmddhh format)
  '''

  #  Read tcvitals file using pandas
  df = pd.read_csv(filepath_or_buffer=vitalfile, header=None, sep = '\s+', engine='python', \
                   names=['ID', 'name', 'yyyymmdd', 'hhmm', 'latitude', 'longitude'], usecols=[1, 2, 3, 4, 5, 6])

  #  Scan dataframe for lines that match analysis time and remove duplicate storms
  df = df.loc[(df['yyyymmdd'] == int(yyyymmddhh[0:8])) & (df['hhmm'] == (int(yyyymmddhh[8:10])*100))].reset_index()
  df.drop_duplicates(subset=['ID'], inplace=True)
  if df.empty:
    return([])

  #  Construct the list of TC names and return the list
  stormlist = []
  for i in range(len(df.index)):
    if int(df['ID'][i][0:2]) < 50:
      stormlist.append('{0}{1}'.format(df['name'][i].lower(),df['ID'][i].lower()))

  return(stormlist)


if __name__ == '__main__':
  '''
     unique_tcs.py - python program that reads a specified tcvitals format file and returns the 
                     list of unique TC names for the given time using command line arguments.

      usage:  unique_tcs.py --yyyymmddhh yyyymmddhh --filename FILE

      where:

        --yyyymmddhh is the date in yyyymmddhh format
        --filename is the tcvitals file to read; otherwise, reads current file on NHC server
  '''

  exp_parser = argparse.ArgumentParser()
  exp_parser.add_argument('--yyyymmddhh',  action='store', type=str, required=True)
  exp_parser.add_argument('--filename',    action='store', type=str)
  args  = exp_parser.parse_args()   

  if args.filename is not None:
    vitalfile = args.filename
  else:
    vitalfile = 'https://ftp.nhc.noaa.gov/atcf/com/tcvitals'

  #  Scan the tcvitals file.  Write the list to a file called tclist
  tclist = unique_tcs_from_vitals(vitalfile, args.yyyymmddhh)

  if len(tclist) > 0:
    pd.DataFrame(data={'storm': tclist}).to_csv('tclist', header=False, index=False)


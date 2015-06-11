'''
    __author__:         Willie Liao
    __description__: Get the trips in train with the closest starting location.
                     Use the weighted average of train trips to estimate trip duration.
                     I could not get R to read lines fast enough so here's the Python version.
                     It's been several years since I've used Python, 
                         so please fork and make it more efficient!
    __edit__:        Multiply lonlat1[1] and lonlat2[1] by pi/180
'''                     

import json
import zipfile
import numpy as np
import pandas as pd
from datetime import datetime
# from __future__ import division

### Control the number of trips read for training 
### Control the number of closest trips used to calculate trip duration
N_read = 2500000
N_trips = 80

### Get Haversine distance
def get_dist(lonlat1, lonlat2):
  lon_diff = np.abs(lonlat1[0]-lonlat2[0])*np.pi/360.0
  lat_diff = np.abs(lonlat1[1]-lonlat2[1])*np.pi/360.0
  a = np.sin(lat_diff)**2 + np.cos(lonlat1[1]*np.pi/180.0) * np.cos(lonlat2[1]*np.pi/180.0) * np.sin(lon_diff)**2  
  d = 2*6371*np.arctan2(np.sqrt(a), np.sqrt(1-a))
  return(d)

# read test

match_fieldA = 'TIMESTAMP'
print("Start!")
test = pd.read_csv(open('test.csv'), usecols=['TRIP_ID', 'POLYLINE', match_fieldA])
test['POLYLINE'] = test['POLYLINE'].apply(json.loads)
test['snapshots'] = test['POLYLINE'].apply(len)
test['lonlat1'] = test['POLYLINE'].apply(lambda x: x[0])
test['lonlat2'] = test['POLYLINE'].apply(lambda x: x[1] if len(x)>1 else x[0])
test.drop('POLYLINE', axis=1, inplace=True)

# read train

print("Loading train...")
train = pd.read_csv(open('train.csv'), usecols=['POLYLINE', match_fieldA])
print("Loaded train")
print("Filtering short trips...")
train['POLYLINE'] = train['POLYLINE'].apply(json.loads)
train['snapshots'] = train['POLYLINE'].apply(len)
train = train[train.snapshots>25]
print("Finished filtering short training trips")
print("Getting lonlats...")
train['lonlat1'] = train['POLYLINE'].apply(lambda x: x[0])
train['lonlat2'] = train['POLYLINE'].apply(lambda x: x[1] if len(x)>1 else x[0])
print("Finished getting lonlats")
train.drop('POLYLINE', axis=1, inplace=True)

# train['TIMESTAMP'] = train['TIMESTAMP'].astype(int).map(datetime.fromtimestamp).map(pd.Timestamp.hour)
# test['TIMESTAMP'] = test['TIMESTAMP'].astype(int).map(datetime.fromtimestamp).map(pd.Timestamp.hour)



test['TRAVEL_TIME1'] = 0
test['TRAVEL_TIME2'] = 0
test['TRAVEL_TIME'] = 0

print("To process:",len(test['lonlat1']))


for wch in [1,2]:
    which_step='lonlat%s'%wch
    for row, ll in enumerate(zip(test[which_step],test[match_fieldA])):
        print(row)
        typ = ll[1]

        ll=ll[0]
        d = train[which_step]

        N = min(N_trips,len(d)-1)
        d = d.apply(lambda x: get_dist(x, ll))
        i = np.argpartition(np.array(d), N)[0:N]
        w = np.maximum(d.iloc[i], 0.01)
        s = train.iloc[i]['snapshots']
        j = np.argpartition(np.array(s), int(N*.98))[0:int(N*.98)]
        test.loc[row, 'TRAVEL_TIME%s'%wch] = 15*np.maximum(test.loc[row, 'snapshots'], np.average(s.iloc[j], weights=1/w.iloc[j]**2))


test['TRAVEL_TIME'] = (test['TRAVEL_TIME1']+test['TRAVEL_TIME2'])/2.0
test['TRAVEL_TIME'] = test['TRAVEL_TIME'].astype(int)
test[['TRIP_ID', 'TRAVEL_TIME']].to_csv('submission_last.csv', index=False)

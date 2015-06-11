import json
import zipfile
import numpy as np
import pandas as pd
import sklearn, sklearn.linear_model, sklearn.cluster, sklearn.ensemble

pd.set_option('display.max_columns', None)

### Control the number of trips read for training 
### Control the number of closest trips used to calculate trip duration
N_read = 10000
N_trips = 100

### Get Haversine distance
def get_dist(lonlat1, lonlat2):
  lon_diff = np.abs(lonlat1[0]-lonlat2[0])*np.pi/360.0
  lat_diff = np.abs(lonlat1[1]-lonlat2[1])*np.pi/360.0
  a = np.sin(lat_diff)**2 + np.cos(lonlat1[1]) * np.cos(lonlat2[1]) * np.sin(lon_diff)**2  
  d = 2*6371*np.arctan2(np.sqrt(a), np.sqrt(1-a))
  return(d)

# read test
test = pd.read_csv(open('test.csv'), usecols=['TRIP_ID','CALL_TYPE', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE'])
test['POLYLINE'] = test['POLYLINE'].apply(json.loads)
test['snapshots'] = test['POLYLINE'].apply(len)
test['lonlat'] = test['POLYLINE'].apply(lambda x: x[0])
test.drop('POLYLINE', axis=1, inplace=True)

# read train
train = pd.read_csv(open('train.csv'), usecols=['TRIP_ID','CALL_TYPE', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE'], nrows=N_read)
train['POLYLINE'] = train['POLYLINE'].apply(json.loads)
train['snapshots'] = train['POLYLINE'].apply(len)
train = train[train.snapshots>30]
train['lonlat'] = train['POLYLINE'].apply(lambda x: x[0])
train.drop('POLYLINE', axis=1, inplace=True)


test.loc[:, 'TRAVEL_TIME'] = 0
train.loc[:, 'TRAVEL_TIME'] = 0

for df in [train,test]:
    for feat in ['CALL_TYPE', 'DAY_TYPE']:
        for letter in ['A','B','C']:
            df[feat+"_IS_"+letter]= (df[feat]==letter).astype(int)

print("k")
m = sklearn.cluster.KMeans(n_clusters=30)
l = list(test['lonlat'])
print(l)

test['lonlat_k'] = m.fit_predict(l)
train['lonlat_k'] = m.predict(list(train['lonlat']))
print(test.head())

cv = train[-int(len(train)/4):]
train = train[:-int(len(train)/4)]


for row, ll in enumerate(test['lonlat']):	
    ### Weighted mean of trip duration
    ### Bound below by 10 meters since we use 1/distance^2 as weight
    ### Only consider trips that have lasted as long as the truncated test trip
    train.loc[:, 'd'] = train['lonlat'].apply(lambda x: get_dist(x, ll))
    ds = train.loc[train.snapshots>=test.loc[row, 'snapshots'], ['d', 'snapshots']].sort('d')[0:N_trips]


    t = np.average(ds.snapshots, weights=1/np.maximum(ds.d, 0.01)**2) if ds.shape[0]>0 else test.loc[row, 'snapshots']*1.1
    test.loc[row, 'TRAVEL_TIME_NEAREST_EUCLID'] = int(15*t)

    t = np.average(ds.snapshots, weights=1/np.maximum(ds.d, 0.01)) if ds.shape[0]>0 else test.loc[row, 'snapshots']*1.1
    test.loc[row, 'TRAVEL_TIME_NEAREST_1NORM'] = int(15*t)


for row, ll in enumerate(cv['lonlat']):
    print(row)
    row = cv.iloc[row].name

    ### Weighted mean of trip duration
    ### Bound below by 10 meters since we use 1/distance^2 as weight
    ### Only consider trips that have lasted as long as the truncated test trip
    train.loc[:, 'd'] = train['lonlat'].apply(lambda x: get_dist(x, ll))
    ds = train.loc[train.snapshots>=cv.loc[row, 'snapshots'], ['d', 'snapshots']].sort('d')[0:N_trips]


    t = np.average(ds.snapshots, weights=1/np.maximum(ds.d, 0.01)**2) if ds.shape[0]>0 else cv.loc[row, 'snapshots']*1.1
    cv.loc[row, 'TRAVEL_TIME_NEAREST_EUCLID'] = int(15*t)

    t = np.average(ds.snapshots, weights=1/np.maximum(ds.d, 0.01)) if ds.shape[0]>0 else cv.loc[row, 'snapshots']*1.1
    cv.loc[row, 'TRAVEL_TIME_NEAREST_1NORM'] = int(15*t)


model = sklearn.ensemble.RandomForestRegressor()

model.fit(cv[['lonlat_k','TRAVEL_TIME_NEAREST_EUCLID','CALL_TYPE_IS_A' , 'CALL_TYPE_IS_B'  ,'CALL_TYPE_IS_C' , 'DAY_TYPE_IS_A','DAY_TYPE_IS_B' , 'DAY_TYPE_IS_C']], cv['snapshots']*15)
test['TRAVEL_TIME'] = model.predict(test[['lonlat_k','TRAVEL_TIME_NEAREST_EUCLID','CALL_TYPE_IS_A' , 'CALL_TYPE_IS_B'  ,'CALL_TYPE_IS_C' , 'DAY_TYPE_IS_A','DAY_TYPE_IS_B' , 'DAY_TYPE_IS_C']])
# test['TRAVEL_TIME'] = test['TRAVEL_TIME_NEAREST_EUCLID']
print(test.head())

test[['TRIP_ID', 'TRAVEL_TIME']].to_csv('submission.csv', index=False)
import pandas as pd
import numpy as np

def mk_2d_var(dat, index, area):
    sample=dat.copy()
    array=sample.to_array().transpose('lat', 'lon', 'variable').values
    lats = index[:,1]
    lons = index[:,2]

    tmp = np.tile(sample.lat.data, [len(lats),1])
    laidx = np.argmin(abs(tmp.T - lats), axis=0)
    tmp = np.tile(sample.lon.data, [len(lons),1])
    loidx = np.argmin(abs(tmp.T - lons), axis=0)

    feature = list(sample)
    latmat = np.zeros((len(index), area*2+1, area*2+1))
    lonmat = np.zeros((len(index), area*2+1, area*2+1))

    for a in range(area*2+1):
        latmat[:,-a-1,:] = np.array(laidx)[:,np.newaxis]-a+area
        lonmat[:,:,-a-1] = np.array(loidx)[:,np.newaxis]-a+area
    allmat = np.array([latmat.flatten().astype(int), lonmat.flatten().astype(int)]).T

    merge = []
    for f, feat in enumerate(feature):
        tmp = array[:,:,f][allmat[:,0], allmat[:,1]].reshape(latmat.shape)
        merge.append(tmp)

    conc = np.array(merge).transpose(tuple(range(1, array.ndim+1))+(0,))
    return conc

def mk_prof_var(dat, index):
    sample = dat.copy()
    array=sample.to_array().transpose('time', 'lat', 'lon', 'variable', 'depth').values
    times = index[:,0]
    lats = index[:,1]
    lons = index[:,2]

    alltime = sample.time.data.astype('datetime64[s]').astype('int64')
    itv = 50000
    tmpmat = []
    for c in range(0, len(times), itv):
        tmp = times[c:c+itv]
        tmp1 = np.tile(alltime, [len(tmp),1])
        tmp2 = abs(tmp1.T - times[c:c+itv])
        tidx = np.argmin(tmp2, axis=0)
        tmpmat.append(tidx)
    tidx = np.concatenate(tmpmat, axis=0)

    tmp = np.tile(sample.lat.data, [len(lats),1])
    laidx = np.argmin(abs(tmp.T - lats), axis=0)
    tmp = np.tile(sample.lon.data, [len(lons),1])
    loidx = np.argmin(abs(tmp.T - lons), axis=0)

    allmat = np.array([tidx, laidx, loidx]).T

    feature = list(sample)
    merge = []
    for f, feat in enumerate(feature):
        tmp = array[:,:,:,f,:][allmat[:,0], allmat[:,1], allmat[:,2]]
        merge.append(tmp)

    conc = np.array(merge).transpose(tuple(range(1, 3))+(0,))
    conc = conc.reshape(conc.shape[0], -1)
    return conc

import pandas as pd
import xarray as xr
import numpy as np
import glob, os, sys
import warnings
import datetime as dt
from tensorflow.keras.layers import Input
pdir=os.getcwd()
sys.path.append(pdir)
from S99_build_model import *
warnings.filterwarnings('ignore')

ddir = '00_Data_n_wgts/'
sdir = '01_3D_temp_results/'
odates = dt.datetime.today()
batch = 4000
dep = [10,20,30,50,75,100,125,150,200,250,300,400,500]

tntd = pd.read_csv(ddir+'01_Fixed_data/D04_tntd.dat', index_col=0)
inps = np.load(ddir+'02_Daily_data/Inps_%s.npz'%odates.strftime('%Y-%m-%d'))
lon, lat = inps['lon'], inps['lat']
nidx = pd.MultiIndex.from_arrays([lon,lat])

inp_2d, inp_dt, inp_woa = inps['inp_2d'], inps['dmat'], inps['woa']

mask = xr.open_dataset(ddir+'01_Fixed_data/D05_mask.nc').mask
wgts = glob.glob(ddir+'99_Weights/*.hdf5')
ofnm = sdir+'%s_results.nc'%(odates.strftime('%Y%m%d'))


isf = os.path.isfile(ofnm)
iss = False
if isf:
    size = os.path.getsize(ofnm)/1024/1024
    if (size < 0.7):
        iss = True
if (not isf) or iss:
    try:
        m_2d = Input(batch_shape=(None, inp_2d.shape[1], inp_2d.shape[2], inp_2d.shape[3]), name='2D')
        m_dt = Input(batch_shape=(None, inp_dt.shape[1]), name='days')
        m_woa = Input(batch_shape=(None, inp_woa.shape[1]), name='WOA')
        mlds = [m_2d, m_dt, m_woa]
        m=build_model(*mlds, 1)
        esbs = []
        for wgt in wgts:
            print(wgt)
            m.load_weights(wgt)
            y_hat = m.predict([inp_2d, inp_dt, inp_woa], batch_size=batch)
            y_hat += tntd['td'].values[np.newaxis,:]
            y_hat /= tntd['tn'].values[np.newaxis,:]
            y_df = pd.DataFrame(y_hat, index=nidx, columns=dep)
            esbs.append(y_df)
        esbs = pd.concat(esbs)
        esb_mean = esbs.groupby(esbs.index).mean()
        esb_mean.index = nidx
        est = esb_mean.stack().to_xarray()
        est = est.rename({'level_0':'lon', 'level_1':'lat', 'level_2':'depth'})
        est *=  mask
        # est = est.assign_coords({'time':odate})
        est.name = 'SeaTemp'

        est.to_netcdf(ofnm)
    except:
        None

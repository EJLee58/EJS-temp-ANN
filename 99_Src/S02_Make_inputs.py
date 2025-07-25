#%%
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import glob, sys, os, warnings
warnings.filterwarnings('ignore')
pdir=os.getcwd()
sys.path.append(pdir)
import F01_make_dset as model

ddir = '00_Data_n_wgts/'
ofnm = ddir+'02_Daily_data/Inps_%s.npz'
xs = np.arange(127, 141.01, 1/12)
ys = np.arange(34, 45.01, 1/12)
area = 6

#%% Data load
pos = np.load(ddir+'01_Fixed_data/D03_xys.npz')
xys = pos['xys']
lon, lat = (((xys.T)*12).round())/12

gepco_pre = xr.open_dataset(ddir+'01_Fixed_data/D01_topo_dset.nc').interp(lat=ys, lon=xs)
woa_pre = xr.open_dataset(ddir+'01_Fixed_data/D02_WOA_all_itp.nc')
for var in woa_pre:
    tmp = woa_pre[var]
    tmp -= np.mean(tmp.values) 
    tmp /= np.std(tmp.values) 

stats = xr.open_mfdataset(ddir+'01_Fixed_data/*stat.nc')
sfls = [glob.glob(ddir+'02_Daily_data/%s*.nc'%x)[-1] for x in ['OSTIA','AVISO']]
itp_mat = []
for s, sfnm in enumerate(sfls):
    tmp = xr.open_dataset(sfnm).rename({'longitude':'lon', 'latitude':'lat'}).load()
    if s==0:
        tmp = tmp.rename({'analysed_sst':'sst'})
    tmp = tmp.drop_vars('time').squeeze()
    itps = []
    for var in list(tmp):
        tmp0 = tmp[var]
        tmp1 = tmp0.to_dataframe()[var].unstack()
        xitp = tmp1.interpolate(axis=1)
        yitp = tmp1.interpolate(axis=0)
        itps.append(xr.DataArray(np.nanmean([xitp, yitp], axis=0), coords=tmp0.coords, name=var))
    itps = xr.merge(itps)
    dset_itp = itps.interp(lat=ys, lon=xs)
    dset_itp = dset_itp.astype('float32')
    dset_itp -= stats.sel(index='mean')[list(tmp)]
    dset_itp /= stats.sel(index='std')[list(tmp)]
    itp_mat.append(dset_itp)
itp_mat = xr.merge(itp_mat)
inp_2d_pre = xr.merge([gepco_pre, itp_mat]).load()

#%% Make dataset 
odate = dt.datetime.today().strftime('%Y-%m-%d')
ttimes = pd.DatetimeIndex(np.tile(np.array(odate), len(lon)))

indexes = np.array([ttimes, lat, lon]).T
inp_2d = model.mk_2d_var(inp_2d_pre, indexes, area)

woa = model.mk_prof_var(woa_pre, indexes)
woa = woa.astype('float32')

# Dmat
days = pd.date_range('1990-01-01', '2040-12-31', freq='1d')
dnu = 365*3
dmat = pd.DataFrame(np.zeros([len(days), dnu]), index=days, columns=days[:dnu]).T
doy = days.dayofyear
for d1, d2 in zip(dmat.columns, doy):
    dmat.loc[:,d1].iloc[d2-1+365] = 100
dmat = dmat.rolling('63d', center=True).mean().resample('1M').mean()
dmat = dmat.groupby(dmat.index.month).mean()
dmat = dmat.T*10
dmat = dmat.loc[ttimes]
dmat /= 1.7994169839936274

np.savez(ofnm%odate, inp_2d=inp_2d, dmat=dmat, woa=woa, lon=lon, lat=lat)
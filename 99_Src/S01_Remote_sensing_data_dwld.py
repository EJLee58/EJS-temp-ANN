import copernicusmarine
import datetime as dt
sdir = '00_Data_n_wgts/02_Daily_data/'


tday = dt.datetime.today()
dset = {'OSTIA':'METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2',
        'AVISO':'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D'}
variables = {'OSTIA':["analysed_sst"],
             'AVISO':["adt", "sla", "ugosa", "vgosa"]}

def data_dwld(dname, tday_str):
    sfnm = sdir+'%s_%s.nc'%(dname, tday_str)
    vnames = variables[dname]
    copernicusmarine.subset(
    dataset_id=dset[dname],
    variables=vnames,
    minimum_longitude=126.5,
    maximum_longitude=141.5,
    minimum_latitude=33.5,
    maximum_latitude=45.5,
    start_datetime="%sT00:00:00"%tday_str,
    end_datetime="%sT00:00:00"%tday_str,
    output_filename=sfnm)

for dname in dset.keys():
    print(dname)
    try:
        tday_str = tday.strftime('%Y-%m-%d')
        data_dwld(dname, tday_str)
        print('Success to download %s data'%dname)
    except Exception as e:
        print(e)
        i = 1
        while i<5:
            tday_str = (tday-dt.timedelta(days=i)).strftime('%Y-%m-%d')
            try:
                data_dwld(dname, tday_str)
                print('Success to download %s data'%dname)
                break
            except Exception as e:
                print(e)
                i += 1
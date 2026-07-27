
# Download the 20CRv3 tas and psl monthly data to obs_raw subfolder

mkdir -p /data/databases/dmdc-variants/mmlea_v2/obs_raw

wget -P /data/databases/dmdc-variants/mmlea_v2/obs_raw https://downloads.psl.noaa.gov//Datasets/20thC_ReanV3/Monthlies/2mSI-MO/air.2m.mon.mean.nc

wget -P /data/databases/dmdc-variants/mmlea_v2/obs_raw https://downloads.psl.noaa.gov/Datasets/20thC_ReanV3/Monthlies/miscSI-MO/prmsl.mon.mean.nc
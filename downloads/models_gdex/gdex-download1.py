#!/usr/bin/env python
""" 
Python script to download selected files from gdex.ucar.edu.
After you save the file, don't forget to make it executable
i.e. - "chmod 755 <name_of_script>"
"""
import sys, os, time
from urllib.request import build_opener

opener = build_opener()

filelist = [
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1101.006i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1231.009i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1251.001i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1231.008i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1281.002i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1301.007i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1251.005i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1121.007i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1301.001i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1281.003i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1231.005i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1251.006i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1231.002i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1301.010i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1081.005i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1141.008i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1061.004i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1231.007i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1231.006i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1001.001i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1281.001i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1281.008i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1251.004i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1231.010i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1251.003i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1041.003i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1281.009i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1301.004i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1251.002i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1281.010i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1231.003i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1021.002i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1301.009i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1281.005i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1301.003i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1251.007i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1251.008i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1301.005i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1301.002i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1251.010i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1231.004i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1161.009i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1281.007i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1231.001i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1281.006i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1301.008i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1281.004i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1301.006i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1181.010i1p1f1_g025_185001-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/cesm2_lens/Amon/psl/psl_Amon_CESM2_cmip6_historical_ssp370_1251.009i1p1f1_g025_185001-210012.nc',
]

DOWNLOAD_TIMEOUT_SECONDS = 60
CHUNK_SIZE = 1024 * 1024
MIN_VALID_FILE_SIZE_BYTES = 2 * 1024 * 1024

def download_with_timeout(file_url, output_file):
    start = time.monotonic()
    with opener.open(file_url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as infile, open(output_file, "wb") as outfile:
        while True:
            if time.monotonic() - start > DOWNLOAD_TIMEOUT_SECONDS:
                raise TimeoutError(f"download exceeded {DOWNLOAD_TIMEOUT_SECONDS}s")
            chunk = infile.read(CHUNK_SIZE)
            if not chunk:
                break
            outfile.write(chunk)

for i in range(10):
  for file in filelist:
      ofile = os.path.basename(file)
      if os.path.exists(ofile):
        if os.path.getsize(ofile) >= MIN_VALID_FILE_SIZE_BYTES:
          sys.stdout.write("skipping " + ofile + " (already downloaded)\n")
          continue
        os.remove(ofile)
      sys.stdout.write("downloading " + ofile + " ... ")
      sys.stdout.flush()
      try:
        download_with_timeout(file, ofile)
        sys.stdout.write("done\n")
      except TimeoutError:
        sys.stdout.write(f"timed out (>{DOWNLOAD_TIMEOUT_SECONDS}s)\n")
        if os.path.exists(ofile):
          os.remove(ofile)
      except Exception as e:
        sys.stdout.write(f"failed ({type(e).__name__})\n")
        if os.path.exists(ofile):
          os.remove(ofile)

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
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r13i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r6i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r13i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r19i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r21i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r8i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r18i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r7i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r22i1p1f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r20i1p1f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r25i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r2i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r14i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r12i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r9i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r16i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r19i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r21i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r17i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r13i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r33i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r36i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r40i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r3i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r31i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r23i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r16i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r6i1p1f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r16i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r5i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r5i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r15i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r15i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r10i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r8i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r14i1p1f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r15i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r4i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r21i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r22i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r21i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r25i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r17i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r22i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r2i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r14i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r16i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r32i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r6i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r23i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r22i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r15i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r9i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r19i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r18i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r4i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r39i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r6i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r25i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r6i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r13i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r20i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r13i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r1i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r23i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r23i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r24i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r5i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r25i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r35i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r6i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r24i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r7i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r10i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r8i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r16i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r14i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r3i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r9i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r24i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r26i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r20i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r4i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r2i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r8i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r17i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r21i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r11i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r11i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r21i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r20i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r24i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r19i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r15i1p1f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r19i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r1i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r10i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r18i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r9i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r12i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r11i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r8i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r10i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r4i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r2i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r23i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r23i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r13i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r8i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r7i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r28i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r11i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r22i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r18i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r11i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r20i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r15i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r9i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r34i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r10i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r25i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r19i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r37i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r3i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r11i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r12i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r2i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r1i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r1i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r12i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r17i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r20i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r4i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r1i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r30i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r16i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r1i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r10i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r2i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r3i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r4i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r18i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r25i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r27i1p2f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r3i1p1f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r7i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r24i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r17i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r17i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r14i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r38i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r5i1p2f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r12i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r3i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r14i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r22i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r5i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r29i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r9i1p2f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp370_r24i1p1f1_g025_201501-210012.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r7i1p1f1_g025_185001-201412.nc',
  'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_historical_r7i1p2f1_g025_185001-201412.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r5i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r12i1p1f1_g025_201501-210012.nc',
  # 'https://osdf-director.osg-htc.org/ncar/gdex/d651039/canesm5_lens/Amon/tas/tas_Amon_CanESM5_ssp585_r18i1p1f1_g025_201501-210012.nc'
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

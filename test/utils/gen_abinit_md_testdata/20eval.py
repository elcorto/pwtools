#!/usr/bin/python

import os.path
import shutil
from pwtools import sql, common

db = sql.SQLiteDB('calc/calc.db', table='calc')
idx = db.get_list1d("select idx from calc")
ionmov = db.get_list1d("select ionmov from calc")
optcell = db.get_list1d("select optcell from calc")

if not os.path.exists('abi'):
    os.makedirs('abi')
for ii in idx:
    fnbase = "abi_ionmov%i_optcell%i" %(ionmov[ii], optcell[ii])
    print fnbase
    for ext in ['in', 'out']:
        shutil.copy("calc/%i/abi.%s" %(ii, ext), "abi/%s.%s" %(fnbase, ext))

common.system("gzip abi/*.out")

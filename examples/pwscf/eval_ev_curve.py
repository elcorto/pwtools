#!/usr/bin/env python

import os
pj = os.path.join
import numpy as np

from pwtools import parse, common

if __name__ == '__main__':
    
    calc_dir = 'calc'

    # calc/0/ .. 
    idxs = range(0, 14)
    
    lst = []
    for idx in idxs:
        dir = pj(calc_dir, str(idx))
        p = parse.PwOutputFile(pj(dir, 'pw.out'), pj(dir, 'pw.in'))
        p.set_attr_lst(['volume', 'etot'])
        p.parse()
        lst.append([p.volume[-1], p.etot[-1]])
    
    arr = np.array(lst)
    np.savetxt('ev.txt', arr)
    
    min_en_idx = np.argmin(arr[:,1])
    eosfit_inp = 'ev.txt\n%f' %arr[min_en_idx,0]
    common.file_write('eosfit.in', eosfit_inp)
    common.system('eosfit.x < eosfit.in')

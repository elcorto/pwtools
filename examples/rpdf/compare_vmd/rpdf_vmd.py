# Use some real world MD data (only 10 steps :) and calculate some RPDFs with
# crys.rpdf() and VMD.

import os
import numpy as np
from matplotlib import pyplot as plt
from pwtools import parse, crys, constants, common

pj = os.path.join

if __name__ == '__main__':
    
    tmpdir = '/tmp/rpdf_vmd_test/'
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    cdct = {}
    
    # real data
    # ---------
    
    common.system("gunzip pw.out.gz")
    pwout = parse.PwOutputFile('pw.out')
    pwin = parse.PwInputFile('pw.in')
    pwout.parse()
    pwin.parse()
    common.system("gzip pw.out")
    alat_ang = float(pwin.namelists['system']['celldm(1)']) * constants.a0_to_A
    # cart Angstrom -> crystal
    cell = np.identity(3)*alat_ang
    coords = pwout.coords / alat_ang # 3d
    
    # O_Ca
    symbols = np.array(pwin.symbols)
    msk1 = symbols=='O'
    msk2 = symbols=='Ca'
    tslice = np.s_[-1]
    coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    cdct['O:Ca:-1:-1'] = coords_lst
    tslice = np.s_[0:]
    coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    cdct['O:Ca:0:-1'] = coords_lst
    
    # Ca_O
    symbols = np.array(pwin.symbols)
    msk1 = symbols=='Ca'
    msk2 = symbols=='O'
    tslice = np.s_[-1]
    coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    cdct['Ca:O:-1:-1'] = coords_lst
    tslice = np.s_[0:]
    coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    cdct['Ca:O:0:-1'] = coords_lst
    
    # O_H
    symbols = np.array(pwin.symbols)
    msk1 = symbols=='O'
    msk2 = symbols=='H'
    tslice = np.s_[-1]
    coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    cdct['O:H:-1:-1'] = coords_lst
    tslice = np.s_[0:]
    coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    cdct['O:H:0:-1'] = coords_lst
    
    # all_all
    tslice = np.s_[-1]
    coords_lst = [coords[...,tslice], coords[...,tslice]]
    cdct['all:all:-1:-1'] = coords_lst
    tslice = np.s_[0:]
    coords_lst = [coords[...,tslice], coords[...,tslice]]
    cdct['all:all:0:-1'] = coords_lst

    dr = 0.1
    for key, val in cdct.iteritems():
        rad, hist, num_int, rmax_auto = \
            crys.rpdf(coords=val, 
                      cell=cell, 
                      dr=dr, 
                      full_output=True,
                      tslice=slice(None))
        s1,s2,first,last = key.split(':')
        first = int(first)
        last = int(last)
        if s1 != 'all':
            s1 = "name %s" %s1
        if s2 != 'all':
            s2 = "name %s" %s2
        # Could also time-slice `coords` here and always use first=0, last=-1,
        # like we use rpdf().
        rad_vmd, hist_vmd, num_int_vmd, rmax_auto_vmd = \
            crys.vmd_measure_gofr(coords=coords, 
                                  cell=cell,
                                  symbols=symbols,
                                  dr=dr, 
                                  rmax='auto', 
                                  selstr1=s1,
                                  selstr2=s2,
                                  first=first,
                                  last=last,
                                  keepfiles=True,
                                  tmpdir=tmpdir,
                                  full_output=True)
        print("rmax_auto: %f" %rmax_auto)
        print("rmax_auto_vmd: %f" %rmax_auto_vmd)
        plt.figure()
        plt.plot(rad, hist, 'b')
        plt.plot(rad_vmd, hist_vmd, 'r')
        plt.plot(rad, num_int, 'b')
        plt.plot(rad_vmd, num_int_vmd, 'r')
        plt.title(key)
    
    plt.show()

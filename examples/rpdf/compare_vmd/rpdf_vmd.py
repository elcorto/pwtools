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
    
    axsf_fn = pj(tmpdir, 'pw.out.axsf')
    common.system("gunzip pw.out.gz")
    pp = parse.PwOutputFile('pw.out', 'pw.in')
    pp.parse()
    common.system('pwo2xsf.sh -a pw.out > %s' %axsf_fn)
    common.system("gzip pw.out")
    alat_ang = float(pp.infile.namelists['system']['celldm(1)']) * constants.a0_to_A
    # cart Angstrom -> crystal
    cp = np.identity(3)*alat_ang
    coords = pp.coords / alat_ang # 3d
    
    # O_Ca
    sy = np.array(pp.infile.symbols)
    msk1 = sy=='O'
    msk2 = sy=='Ca'
    tslice = np.s_[-1]
    coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    cdct['O:Ca:-1:-1'] = coords_lst
    tslice = np.s_[1:]
    coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    cdct['O:Ca:1:-1'] = coords_lst
    
    # Ca_O
    sy = np.array(pp.infile.symbols)
    msk1 = sy=='Ca'
    msk2 = sy=='O'
    tslice = np.s_[-1]
    coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    cdct['Ca:O:-1:-1'] = coords_lst
    tslice = np.s_[1:]
    coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    cdct['Ca:O:1:-1'] = coords_lst
    
    # O_H
    sy = np.array(pp.infile.symbols)
    msk1 = sy=='O'
    msk2 = sy=='H'
    tslice = np.s_[-1]
    coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    cdct['O:H:-1:-1'] = coords_lst
    tslice = np.s_[1:]
    coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    cdct['O:H:1:-1'] = coords_lst
    
    # all_all
    tslice = np.s_[-1]
    coords_lst = [coords[...,tslice], coords[...,tslice]]
    cdct['all:all:-1:-1'] = coords_lst
    tslice = np.s_[1:]
    coords_lst = [coords[...,tslice], coords[...,tslice]]
    cdct['all:all:1:-1'] = coords_lst

    # Random data
    # -----------

    ##natoms_O = 20
    ##natoms_H = 5
    ##symbols = ['O']*natoms_O + ['H']*natoms_H
    ##coords = np.random.rand(natoms_H + natoms_O, 3, 500)
    ##cp = np.identity(3)*10
    ##axsf_fn = 'foo.axsf'
    ##crys.write_axsf(axsf_fn, coords, cp, symbols)

    ### all all
    ##tslice = np.s_[-1]
    ##coords_lst = [coords[...,tslice], coords[...,tslice]]
    ##cdct['all:all:-1:-1'] = coords_lst
    ##tslice = np.s_[200:]
    ##coords_lst = [coords[...,tslice], coords[...,tslice]]
    ##cdct['all:all:200:-1'] = coords_lst

    ### O H
    ##sy = np.array(symbols)
    ##msk1 = sy=='O'
    ##msk2 = sy=='H'
    ##tslice = np.s_[-1]
    ##coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    ##cdct['O:H:-1:-1'] = coords_lst
    ##tslice = np.s_[200:]
    ##coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    ##cdct['O:H:200:-1'] = coords_lst
    ##
    ### H O
    ##sy = np.array(symbols)
    ##msk1 = sy=='H'
    ##msk2 = sy=='O'
    ##tslice = np.s_[-1]
    ##coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    ##cdct['H:O:-1:-1'] = coords_lst
    ##tslice = np.s_[200:]
    ##coords_lst = [coords[msk1,...,tslice], coords[msk2,...,tslice]]
    ##cdct['H:O:200:-1'] = coords_lst

    dr = 0.1
    for key, val in cdct.iteritems():
        rad, hist, num_int, rmax_auto = \
            crys.rpdf(coords=val, 
                      cp=cp, 
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
        if first != -1:
            first += 1
        gofr = crys.vmd_measure_gofr(axsf_fn, 
                                 dr=dr, 
                                 rmax=rmax_auto, 
                                 selstr1=s1,
                                 selstr2=s2,
                                 first=first,
                                 last=last,
                                 keepfiles=True,
                                 tmpdir=tmpdir)
        print("rmax_auto: %f" %rmax_auto)
        plt.figure()
        plt.plot(rad, hist, 'b')
        plt.plot(gofr[:,0], gofr[:,1], 'r')
        plt.plot(rad, num_int, 'b')
        plt.plot(gofr[:,0], gofr[:,2], 'r')
        plt.title(key)
    
    plt.show()

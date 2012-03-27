#!/usr/bin/env python

# Use some real world MD data (only 10 steps :) and calculate some RPDFs with
# crys.rpdf() and VMD, only up to rmax_auto.

import os
import numpy as np
from matplotlib import pyplot as plt
from pwtools import parse, crys, constants, common, io

pj = os.path.join

if __name__ == '__main__':
    
    tmpdir = '/tmp/rpdf_vmd_test/'
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    dct = {}
    
    common.system("gunzip pw.out.gz")
    traj = io.read_pw_md('pw.out')
    common.system("gzip pw.out")
    symbols = np.array(traj.symbols)
    
    # O_Ca
    msk1 = symbols=='O'
    msk2 = symbols=='Ca'
    amask = [msk1, msk2]
    tmask = np.s_[-1]
    dct['O:Ca:-1:-1'] = {'amask': amask, 'tmask': tmask}
    tmask = np.s_[0:]
    dct['O:Ca:0:-1'] = {'amask': amask, 'tmask': tmask}
    
    # Ca_O
    msk1 = symbols=='Ca'
    msk2 = symbols=='O'
    amask = [msk1, msk2]
    tmask = np.s_[-1]
    dct['Ca:O:-1:-1'] = {'amask': amask, 'tmask': tmask}
    tmask = np.s_[0:]
    dct['Ca:O:0:-1'] = {'amask': amask, 'tmask': tmask}
    
    # O_H
    msk1 = symbols=='O'
    msk2 = symbols=='H'
    amask = [msk1, msk2]
    tmask = np.s_[-1]
    dct['O:H:-1:-1'] = {'amask': amask, 'tmask': tmask}
    tmask = np.s_[0:]
    dct['O:H:0:-1'] = {'amask': amask, 'tmask': tmask}
    
    # all_all
    amask = None
    tmask = np.s_[-1]
    dct['all:all:-1:-1'] = {'amask': amask, 'tmask': tmask}
    tmask = np.s_[0:]
    dct['all:all:0:-1'] = {'amask': amask, 'tmask': tmask}

    dr = 0.1
    for key, val in dct.iteritems():
        print "--- %s ---" %key
        print "pwtools ..."
        out_pwt = \
            crys.rpdf(traj, 
                      dr=dr, 
                      tmask=val['tmask'],
                      amask=val['amask'],
                      )
        s1,s2,first,last = key.split(':')
        first = int(first)
        last = int(last)
        if s1 != 'all':
            s1 = "name %s" %s1
        if s2 != 'all':
            s2 = "name %s" %s2
        print "vmd ..."
        out_vmd = \
            crys.vmd_measure_gofr(traj, 
                                  dr=dr, 
                                  rmax='auto', 
                                  sel=[s1,s2],
                                  first=first,
                                  last=last,
                                  keepfiles=True,
                                  tmpdir=tmpdir,
                                  )
        print("rmax_auto: %f" %crys.rmax_smith(traj.cell[0,...]))
        plt.figure()
        plt.plot(out_pwt[:,0], out_pwt[:,1], 'b', label='pwtools')
        plt.plot(out_vmd[:,0], out_vmd[:,1], 'r', label='vmd')
        plt.plot(out_pwt[:,0], out_pwt[:,2], 'b')
        plt.plot(out_vmd[:,0], out_vmd[:,2], 'r')
        plt.title(key)
        plt.legend()
    
    plt.show()

#!/usr/bin/env python

# Compare two MD trajectories. Suppose you have two runs:
#   run1/pw.{in,out} 
#   run2/pw.{in,out} 

import os.path.join as pj
import matplotlib.pyplot as plt
from pwtools import parse, common, crys

# {run: dir}
runs = {'1': 'run1', '2': 'run2'}
pp = {}

if __name__ == '__main__':
    
    # parse MD data
    # simply
    #   p1 = parse.PwOutputFile(filename='run1/pw.out', infile='run1/pw.in') 
    #   p2 = parse.PwOutputFile(filename='run2/pw.out', infile='run2/pw.in') 
    #   p1.parse()
    #   p2.parse()
    # or ...  
    for run, dir in runs.iteritems():
        pp[run] = parse.PwOutputFile(filename=pj(dir, 'pw.out'), 
                                     infile=pj(dir, 'pw.in'))
        pp[run].parse()

    # plot etot, total_force
    # simply
    #   figure()
    #   plot(p1.etot, label='etot: run 1')
    #   plot(p2.etot, label='etot: run 2')
    #   legend()
    #   figure()
    #   plot(p1.total_force, label='total_force: run 1')
    #   plot(p2.total_force, label='total_force: run 2')
    #   legend()
    # or ...   
    for what in ['etot', 'total_force']:
        plt.figure()
        for run, p in pp.iteritems():
            lab = "%s: run %s" %(what, run)
            plt.plot(getattr(p, what), label=lab)
        plt.legend()
    
    # plot difference of atomic coords vs. MD steps
    dcoords = pp['1'].coords - pp['2'].coords
    # coords is a 3d array, time axis=2
    rms1 = np.sqrt((dcoords**2.0).sum(axis=0).sum(axis=0))
    # or easier ...
    rms2 = crys.rms3d(dcoords, axis=-1, nitems=1)
    plt.figure()
    plt.plot(rms1)
    plt.plot(rms2)
    plt.legend()

    plt.show()

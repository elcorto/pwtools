#!/usr/bin/env python

# Compare two MD trajectories. Suppose you have two runs:
#   run1/pw.{in,out} 
#   run2/pw.{in,out} 

import matplotlib.pyplot as plt
from pwtools import parse, common

if __name__ == '__main__':
    
    # parse MD data
    p1 = parse.PwOutputFile(fielname='run1/pw.out', infile='run1/pw.in'))
    p2 = parse.PwOutputFile(fielname='run2/pw.out', infile='run2/pw.in'))
    p1.parse()
    p2.parse()

    # plot etot
    plt.figure()
    plt.plot(p1.etot, label='run1')
    plt.plot(p2.etot, label='run2')
    plt.legend()

    # plot difference of atomic coords vs. MD steps
    plt.figure()
    # coords is a 3d array, time axis=1
    diff = (p1.coords - p2.coords).sum(axis=0).sum(axis=1)
    plt.plot(diff)
    plt.legend()

    plt.show()

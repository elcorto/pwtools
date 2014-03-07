#!/usr/bin/env python

"""
Example for smoothing a signal with a Lorentz kernel. 

The problem with Lorentz is that the function has very long tails (never really
goes to zero at both ends) compared to a Gaussian with the same spread
parameter "std".

* If your to-be-smoothed data is properly zero at both ends (i.e. discrete
  frequency values which need to be convoluted with a spreading funtion to plot
  a spectrum
* you want very very precise results

then use direct convolution (scipy.signal.convolve)

* without edge-effect correction
* a Lorentz function of at least 2 x data length   

instead of pwtools.signal.smooth() with egde correction, since the current
implementation is limited in the kernel length to npoints-1.
"""

import numpy as np
from pwtools import mpl
from pwtools.signal import scale, lorentz, smooth
from scipy.signal import convolve
plt = mpl.plt

npoints = 200
std = 1.0

for nrand_fac in [0.2, 1.0]:
    # random data to be smoothed, with much (nrand_fac=0.2) or no
    # (nrand_fac=1.0) zeros at both ends
    plt.figure()
    y = np.zeros(npoints)
    x = np.arange(len(y))
    nrand = int(npoints*nrand_fac)
    # even nrand
    if nrand % 2 == 1:
        nrand += 1
    y[npoints/2-nrand/2:npoints/2+nrand/2] = np.random.rand(nrand)
    
    # Sum of Lorentz functions at data points. This is the same as convolution
    # with a Lorentz function withOUT end point correction, valid if data `y`
    # is properly zero at both ends, else edge effects are visible: smoothed
    # data always goes to zero at both ends, even if original data doesn't. We
    # need to use a very wide kernel with at least 2*npoints b/c of long
    # Lorentz tails.
    a2 = np.zeros_like(y)
    for xi,yi in enumerate(y):
        a2 += yi * std / ((x-xi)**2.0 + std**2.0)
    a2 = scale(a2)
    plt.plot(a2, label='sum')
    # convolution with wide kernel
    klen = 2*npoints
    klen = klen+1 if klen % 2 == 0 else klen # odd kernel
    kern = lorentz(klen, std=std)
    plt.plot(scale(convolve(y, kern/float(kern.sum()), 'same')),
             label='conv, klen=%i' %klen)

    # Convolution with Lorentz function with end-point correction. With
    # signal.smooth() we can only use npoints-1 as max kernel length due to the
    # way the end-point correction is construted (need to check if this can be
    # done better!). 
    for klen in [npoints-1, npoints/12]:
        klen = klen+1 if klen % 2 == 0 else klen # odd kernel
        kern = lorentz(klen, std=std)
        a1 = scale(smooth(y, kern))
        plt.plot(a1, label='conv+egde, klen=%i' %klen)
    plt.title("npoints=%i" %npoints)
    plt.legend()
plt.show()

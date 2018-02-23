#!/usr/bin/env python3

"""
Example for smoothing a signal with a Lorentz kernel. 

We show how to use (a) scipy.signal.convolve, (b) direct sum of Lorentz
functions (convolution by hand) and (c) pwtools.signal.smooth. We also test
various kernel lengths (klen below) and we show the severe edge effects with
normal convolution.

Tails
-----

The problem with Lorentz is that the function has very long tails (never really
goes to zero at both ends) compared to a Gaussian with the same spread
parameter "std". Therefore very wide kernels are needed, as in 100*std or
better, where we get away with 6*std for gaussians.

Edge effects
------------

If your to-be-smoothed data is properly zero at both ends, then you may skip
convolution with edge-effect correction (pwtools.signal.smooth) and use direct
convolution (scipy.signal.convolve), but there is really no reason to do so,
except that you don't have pwtools :)
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
    # (nrand_fac=1.0) zeros at both ends, edge effects are visible for no zeros
    # at the ends
    plt.figure()
    y = np.zeros(npoints)
    x = np.arange(len(y))
    nrand = int(npoints*nrand_fac)
    # even nrand
    if nrand % 2 == 1:
        nrand += 1
    y[npoints//2-nrand//2:npoints//2+nrand//2] = np.random.rand(nrand) + 2.0
    
    # Sum of Lorentz functions at data points. This is the same as convolution
    # with a Lorentz function withOUT end point correction, valid if data `y`
    # is properly zero at both ends, else edge effects are visible: smoothed
    # data always goes to zero at both ends, even if original data doesn't. We
    # need to use a very wide kernel with at least 100*std b/c of long
    # Lorentz tails. Better 200*std to be safe.
    sig = np.zeros_like(y)
    for xi,yi in enumerate(y):
        sig += yi * std / ((x-xi)**2.0 + std**2.0)
    sig = scale(sig)
    plt.plot(sig, label='sum')
    # convolution with wide kernel
    klen = 200*std
    klen = klen+1 if klen % 2 == 0 else klen # odd kernel
    kern = lorentz(klen, std=std)
    plt.plot(scale(convolve(y, kern/float(kern.sum()), 'same')),
             label='conv, klen=%i' %klen)

    # Convolution with Lorentz function with end-point correction.  
    for klen in [10*std, 100*std, 200*std]:
        klen = klen+1 if klen % 2 == 0 else klen # odd kernel
        kern = lorentz(klen, std=std)
        plt.plot(scale(smooth(y, kern)), label='conv+egde, klen=%i' %klen)
    plt.title("npoints=%i" %npoints)
    plt.legend()
plt.show()

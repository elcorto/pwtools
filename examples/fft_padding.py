#!/usr/bin/env python3

# Sometimes, a signal needs to be zero-padded before an FFT (e.g. when one
# calculates the correlation with FFT). That introduces "leakage" and sinc-like
# ripple pattern due to the cut-off between signal and zero-padding.

# This example shows how to use a Welch window before zero-padding or smoothing
# the padded signal. The smoothing is done by convolution with a gaussian
# kernel.
#
# The result is that Welch windowing is better than smoothing. Also, smoothing
# needs an adjustable parameter - the kernel's std dev. If that is choosen too
# big, then the smoothing will filter out high frequencies.


from math import pi
import numpy as np
from matplotlib import pyplot as plt
from pwtools import signal
from scipy.signal import convolve, gaussian, correlate
from scipy.fftpack import fft

nn = 200
nadd = 5*nn
t = np.linspace(0.123,0.567,nn)
x = np.sin(2*pi*10*t) + np.cos(2*pi*3*t) + np.sin(2*pi*30*t)
dt = t[1]-t[0]

pad_x = signal.pad_zeros(x, nadd=nadd)
pad_welch_x = signal.pad_zeros(x*signal.welch(nn), nadd=nadd)
kern = gaussian(M=20,std=2) # width M must be 6..10 x std
smooth_pad_x = convolve(signal.pad_zeros(x,nadd=nadd),kern,'same')/10.0
##mirr_x = signal.mirror(x)
##welch_mirr_x = signal.mirror(x)*signal.welch(2*nn-1)
##pad_welch_mirr_x = signal.pad_zeros(signal.mirror(x)*signal.welch(2*nn-1),
##                                    nadd=2*nn-1)

plt.figure()
plt.plot(pad_x, label='pad_x (padded signal)')
plt.plot(pad_welch_x, label='pad_welch_x')
plt.plot(smooth_pad_x,label='smooth_pad_x')
plt.xlabel('time [s]')
plt.xlim(0,300)
plt.legend()

plt.figure()
f,d = signal.ezfft(x, dt)
plt.plot(f,abs(d), label='x')
f,d = signal.ezfft(pad_x, dt)
plt.plot(f,abs(d), label='pad_x')
f,d = signal.ezfft(pad_welch_x, dt)
plt.plot(f,abs(d), label='pad_welch_x')
f,d = signal.ezfft(smooth_pad_x, dt)
plt.plot(f,abs(d), label='smooth_pad_x')
##f,d = signal.ezfft(mirr_x, dt)
##plt.plot(f,abs(d), label='mirr_x')
##f,d = signal.ezfft(welch_mirr_x, dt)
##plt.plot(f,abs(d), label='welch_mirr_x')
##f,d = signal.ezfft(pad_welch_mirr_x, dt)
##plt.plot(f,abs(d), label='pad_welch_mirr_x')

plt.xlabel('freq [Hz]')
plt.legend()
plt.xlim(0,50)
plt.show()

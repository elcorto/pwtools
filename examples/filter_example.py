# Example for using a digital filter.
#
# Aliasing
# --------
# Suppose you have a signal with 50 + 80 or 120 Hz and a Nyquist freq of 100.
# The 80 Hz part can be filtered out by using a lowpass with e.g. cutoff=70 or
# a bandpass with cutoff=[10,70] or so.
#
# Because the Nyquist frequency is 100 Hz, the 120 Hz signal is aliased back
# (folded back at 100 Hz) to 80 Hz by sampling the signal and shows then up as
# a peak in FFT. This can be also taken care of by using a filter in time
# domain on the signal before FFT, but make sure that the filter cutoff
# frequencies are such that the aliased peak is excluded (i.e. smaller then 80
# Hz). As such, the aliased 50+120 signal behaves exactly like a 50+80 signal.
#
# Note that in general, you don't know to which frequency aliases have been put
# and just using a bandpass around you desired frequency band won't help. The
# only solution in this case is to avoid aliasing in the first place :)


import numpy as np
from pwtools import mpl
from scipy.signal import hanning
from scipy.fftpack import fft
from pwtools.signal import fftsample, FIRFilter, pad_zeros
pi = np.pi
plt = mpl.plt

plots = mpl.prepare_plots(['freq', 'filt_pad', 'filt_nopad'])
nyq = 100 # Hz
df = 1.0  # Hz
dt, nstep = fftsample(nyq, df, mode='f')
t = np.linspace(0, 1, int(nstep))
filt1 = FIRFilter(cutoff=[10,50], nyq=nyq, mode='bandpass', ripple=60,
                  width=10)
filt2 = FIRFilter(cutoff=[10,50], nyq=nyq, mode='bandpass', ntaps=100,
                  window='hamming')
plots['freq'].ax.plot(filt1.w, abs(filt1.h), label='filt1')
plots['freq'].ax.plot(filt2.w, abs(filt2.h), label='filt2')
plots['freq'].ax.legend()

for pad in [True,False]:
    x = np.sin(2*pi*20*t) + np.sin(2*pi*80*t)
    if pad:
        x = pad_zeros(x, nadd=len(x))
        pl = plots['filt_pad']
    else:
        pl = plots['filt_nopad']
    f = np.fft.fftfreq(len(x), dt)
    sl = slice(0, len(x)/2, None)
    win = hanning(len(x))
    pl.ax.plot(f[sl], np.abs(fft(x)[sl]), label='fft(x)')
    pl.ax.plot(f[sl], np.abs(fft(filt1(x))[sl]),     label='fft(filt1(x))')
    pl.ax.plot(f[sl], np.abs(fft(filt1(win*x))[sl]), label='fft(filt1(hanning*x))')
    pl.ax.plot(f[sl], np.abs(fft(filt2(win*x))[sl]), label='fft(filt2(hanning*x))')
    pl.ax.set_title('zero pad = %s' %pad)
    pl.ax.legend()

plt.show()

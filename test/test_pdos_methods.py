# -*- coding: utf-8 -*- 

"""
methods
-------

Test 3 different methods to calculate the phonon density as the power spectrum
(PSD) of atomic velocities. We generate random 1d and 3d data containing sums
of sin()s.

1) FFT of the VACF (pydos.vacf_pdos())
2) direct FFT of the velocities (pydos.direct_pdos())
3) Axel Kohlmeyer's fourier.x tool from the CPMD contrib sources. We
   can use this as a reference to see if we do The Right Thing.

(1) and (2) must produce exactly the same result within numerical noise b/c
they are mathematically equivalent and are implemented to produce the same
frequency resolution.
(3) must match in principle w.r.t. peak positions and relative peak heights. It
can't match exactly b/c it has lower frequency resolution (b/c they do not
zero-padd before fft-ing the velocities).

frequency axis
--------------

In the literature, the PDOS is defined as depending on omega (angular
frequency). 

But here, all 3 methods give a frequency axis in f (ordinary cyclic frequency)
AND NOT IN ANGULAR FREQUENCY (omega)!!!! This is b/c (1) and (2) use scipy.fft
and numpy.fftfreq() and these tools mostly use the conventions used in
Numerical Recipes, where we deal with f, not omega:

    f_i = i / (dt * N) , i = -N/2 .. N/2, N = len(data)

We have:
    (1) : f [Hz]
    (2) : f [Hz]
    (3) : f [1/cm] (see below)

You may want to multiply by 2*pi to get an omega-axis.

theory
------

For 3d arrays with real velocity data, (1) is much slower b/c ATM the VACF is
calculated directly via loops. Possible optimization: calc the autocorrelation
via FFT (see lib/corr.py and the Wiener-Khinchin theorem). But this is useless
b/c the theorem tells us that in fact method (1) is just a more complicated way
of doing (2). BTW, (3) is the same as (2) -- direct FFT of velocities.

Let corr(v,v) == VACF == the velocity autocorrelation. We skip all technical
details like zero-padding, mirroring, normalization. See also NR, chap. 12
"Fast Fourier Transform".
  
  Correlation theorem:
    fft(corr(v,w)) = fft(v)*fft(w).conj()
 
  Wiener-Khinchin theorem:
    fft(corr(v,v)) = fft(v)*fft(v).conj() = |fft(v)|^2
    
    =>
    PSD = fft(corr(v,v))         # (1)
        = |fft(v)|^2             # (2), (3)

Note that we skip the factor of 2 from the definition of the "real" power
spectrum in NR (PSD = 2*|fft(v)|**2) b/c `v` is a real function of time and we
normalize to unity integral area in the frequency domain anyway.

When using (in the 1d and 3d case) arr = <velocity>, then the PSD peak
heights increase from low to high frequencies. With arr = coords, they have
all approx. the same height. That's b/c with arr = <velocity>, we are
integrating (fft-ing) the *derivative* of a time function. Then, with
integration by parts:

F[v'(t)](f) =             Int(t) v' * exp(i*2*pi*f*t) dt 
            = ...
            = -i*2*pi*f * Int(t) v  * exp(i*2*pi*f*t) dt
            = -i*2*pi*f * F[v(t)]
=>
|F[v'(t)](f)|^2 ~ f^2 * |F[v(t)](f)|^2

So the power spectrum should go like f^2. One could also say that the
derivation of `v` damps low frequencies and pushes higher ones. See also
http://mathworld.wolfram.com/FourierTransform.html
Note: Not sure if this applies one to one to FFT (i.e. discrete FT), but the
idea should be the same.

fourier.x
---------

You must set the path to fourier.x below. If it is not found, it won't be used
and the tests will run w/o it. If it is used, some input and output files in a
dir `fourier_dir` (set below) are produced.

The tool reads from stdin (interactive use or we pipe an input file to it). The
input is queried in main.f90. These quantities are required in this order:

  - input data file
  - output data file
  - dt in Hartree (see below)
  - temperature (seems not important for pure power spectrum w/o any prefactor, 
        i.e. the 2nd column of the output data file)
  - max frequency in 1/cm
  - frequency increment (always use 1 to do no averaging)

Unfortunately, the code is sparesly commented and hard to understand.
Nevertheless, we verified that the frequency axis is indeed f, not omega. In
main.f90, there is a part:

  PRINT *,"give time difference in input file (in atomic units):"
  PRINT *,"[CAUTION: this is the MD time step times the sampling frequency.]"
  READ (*,*) data_t
  wave_fac = 219474.0_dbl/data_t

whete `data_t` is our input dt in Hartree atomic units (dt [s] / constants.th).
Now WTF is this number 219474.0 !!?? By experiment(!), we found that it is
    1 / constants.th / (constants.c0*100) / (2*pi)

Some variables in main.f90 and the corresponding value here (main.f90 : here)
    n : nstep - 1
        - main.f90: n comes from check_file()
        - here: nstep-1, -1 b/c we calculate a "velocity" and
          len(diff(arange(10))) == 9, this is also the number of time values
          (num. of lines) in the input data file
    nn : (nstep-1)/2
    nstep :
        nstep is the input "frequency increment"
    nover :
        Somewhere in main.f90, we have "nover = nstep". Duh.

Looking thru the code, we find that all 2*pi terms eventually cancel at the
end. The output frequency axis is "t*wave_fac" in write_file() and t == wtrans
in main.f90. We have 

    f = i / ((n+1) * 100*c0 * dt)

which is just (almost) the definition of the f-axis for DFT, converted to 1/cm. 
The n+1 might be wrong leading to very slightly shifted frequencies, though.
"""

import os
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
##from scipy import signal

from pwtools import fft as pwfft 
from pwtools import pydos, corr, constants, common

rand = np.random.rand
pj = os.path.join

def cut_norm(full_y, dt, area=1.0):
    """Cut out and FFT spectrum from scpiy.fftpack.fft() (or numpy.fft.fft())
    result and normalize the integral `int(f) y(f) df = area`.
    
    full_y : 1d array
        Result of fft(...)
    dt : time step
    area : integral area
    """
    full_faxis = np.fft.fftfreq(full_y.shape[0], dt)
    split_idx = full_faxis.shape[0]/2
    y_out = full_y[:split_idx]
    faxis = full_faxis[:split_idx]
    return faxis, common.norm_int(y_out, faxis, area=area)


###############################################################################
# common settings for 1d and 3d case
###############################################################################

fourier_exe = common.fullpath('~/soft/lib/fourier/fourier.x')
fourier_dir = '/tmp/fourier_test'
use_fourier = os.path.exists(fourier_exe)
if use_fourier:
    print "will use fourier.x"
    if not os.path.exists(fourier_dir):
        os.mkdir(fourier_dir)
    else:        
        print("warning: dir exists: %s" %fourier_dir)
else:
    print "will NOT use fourier.x"

# For the calculation of nstep and dt, we increase `fmax` to
# fmax*fmax_extend_fac to lower dt. That way, the signal still contains
# frequencies up to `fmax`, but the FFT frequency axis does not end at exactly
# `fmax` but extends to higher values (we increase the Nyquist frequency).
df = 0.001
fmax = 1
fmax_extend_fac = 1.5 

# number of frequencies contained in the signal
nfreq = 10

# time axis that assures that the fft catches all freqs up to and beyond fmax
# to avoid FFT aliasing
dt, nstep = pwfft.fftsample(fmax*fmax_extend_fac, df)
nstep = int(nstep)
taxis = np.linspace(0, dt*nstep, nstep, endpoint=False)

print "dt [s]:", dt
print "dt [Hartree]:", dt/constants.th
print "nstep:", nstep
print "fmax [Hz]:", fmax
print "nfreq:", nfreq


###############################################################################
# 1d arrays
###############################################################################

# First, we treat a simple 1d signal to test the math.

# coords: 1d signal composed of `nfreq` sin's containing frequencies up to
# `fmax`, the power spectrum of `coords` must reproduce peaks at these
# frequencies exactly
print "coords ..."
freqs = rand(nfreq)*fmax
freqs.sort()
print "frequencies:", freqs
coords = np.sin(2*pi*freqs[:,None]*taxis).sum(axis=0)

# below, `arr` is used fo all following calcs
##arr = coords
arr = np.diff(coords) # "velocity"

# Our methods 1 and 2 must result in exactly the same after
# norm_int()'ing them (within numerical noise). 
# In both, we use a Welch window as in fourier.x .

# 1) Zero-pad arr at the end and fft directly. For padding, use nadd=N-1 to get
# exactly the same signal length as y2 b/c in case of y2: mirror(rand(N)).shape
# = (2N-1,), i.e. the point at t=0 apprears only once. Padding increases the
# frequency resolution (almost factor 2) to exactly the same df as we get for
# method 2 b/c of the mirroring of the signal there.
print "|fft(arr)|^2 ..."
fft_arr = pwfft.pad_zeros(arr*pwfft.welch(arr.shape[0]), nadd=arr.shape[0]-1)
y1 = np.abs(fft(fft_arr))**2
print "y1.shape", y1.shape 

# 2) fft the autocorrelation of `arr`
print "|fft(acorr(arr))| ..."
fft_arr = pydos.mirror(corr.acorr(arr*pwfft.welch(arr.shape[0]), method=5))
y2 = np.abs(fft(fft_arr))
print "y2.shape", y2.shape 

# 3) fourier.x
#
# For the 1d case, we write the time trace in a format suggested in the CPMD
# manual and the fourier.x README file: awk ’ { print $1, 0.0, 0.0, 0.0, 0.0,
# 0.0, $2; } ’ ENERGIES > ekinc.dat where ENERGIES is a CPMD output file. From
# that, only column 1 (time step) and some energy value from column 2 is used.
if use_fourier:
    print "fourier.x ..."
    fourier_in_data = np.zeros((arr.shape[0],7))
    fourier_in_data[:,0] = np.arange(arr.shape[0])
    fourier_in_data[:,6] = arr
    fourier_in_data_fn = pj(fourier_dir, 'fourier_in_data_1d.txt')
    fourier_out_data_fn = pj(fourier_dir, 'fourier_out_data_1d.txt')
    fourier_in_fn = pj(fourier_dir, 'fourier_1d.in')
    fourier_in_txt = '%s\n%s\n%.16e\n%f\n%.16e\n%i' %(fourier_in_data_fn,
                                                      fourier_out_data_fn,
                                                      dt/constants.th,
                                                      300,
                                                      fmax*fmax_extend_fac/(constants.c0*100),
                                                      1)
    common.file_write(fourier_in_fn, fourier_in_txt)
    np.savetxt(fourier_in_data_fn, fourier_in_data)
    # that prints some jabber
    common.system(fourier_exe + ' < ' + fourier_in_fn)
    fourier_out_data = np.loadtxt(fourier_out_data_fn)
    f3 = fourier_out_data[:,0]*(constants.c0*100) # 1/cm -> Hz
    y3n = common.norm_int(fourier_out_data[:,1], f3)

f1, y1n = cut_norm(y1, dt)
f2, y2n = cut_norm(y2, dt)

figs = []
axs = []

print "plotting ..."
figs.append(plt.figure())
axs.append(figs[-1].add_subplot(111))
axs[-1].set_title('1d arr')
axs[-1].plot(f1, y1n, label='1d |fft(arr)|^2, direct')
axs[-1].plot(f2, y2n, label='1d |fft(acorr(arr))|, vacf')
if use_fourier:
    axs[-1].plot(f3, y3n, label='1d fourier.x')
axs[-1].legend()


###############################################################################
# 3d arrays
###############################################################################

# Now, 3d arrays with "real" atomic velocities, test the pydos methods.

# Use most settings (nfreq, ...) from above. Create random array of x,y,z time
# traces for `natoms` atoms. Each x,y,z trajectory is a sum of sin's (`coords`
# in the 1d case).
natoms = 5
coords = np.empty((natoms, nstep, 3))
# `nfreq` frequencies for each x,y,z component of each atom
freqs = rand(natoms, nfreq, 3)*fmax
for i in range(coords.shape[0]):
    for k in range(coords.shape[2]):
        # vector w/ frequencies: freqs[i,:,k] <=> f_j, j=0, ..., nfreq-1
        # sum_j sin(2*pi*f_j*t)
        coords[i,:,k] = np.sin(2*pi*freqs[i,:,k][:,None]*taxis).sum(axis=0)

##arr = coords
arr = pydos.velocity(coords, copy=True)
massvec = rand(natoms)

# no mass weighting
M = None
f4, y4n = pydos.vacf_pdos(arr, dt=dt, m=M, mirr=True)
f5, y5n = pydos.direct_pdos(arr, dt=dt, m=M)

# with mass weighting
M = massvec
f6, y6nm = pydos.vacf_pdos(arr, dt=dt, m=M, mirr=True)
f7, y7nm = pydos.direct_pdos(arr, dt=dt, m=M)

if use_fourier:
    # For each atom, write an array (time.shape[0], 3) with coords at all time
    # steps, run fourier.x on that, sum up the power spectra. No mass
    # weighting.
    fourier_in_data = np.zeros((arr.shape[1],7))
    fourier_in_data[:,0] = np.arange(arr.shape[1])
    print "running fourier.x for all atoms ..."
    for iatom in range(arr.shape[0]):
        fourier_in_data[:,4:] = arr[iatom,...]
        fourier_in_data_fn = pj(fourier_dir, 'fourier_in_data_3d_atom%i.txt'
                                %iatom)
        fourier_out_data_fn = pj(fourier_dir, 'fourier_out_data_3d_atom%i.txt'
                                 %iatom)
        fourier_in_fn = pj(fourier_dir, 'fourier_3d_atom%i.in' %iatom)
        fourier_in_txt = '%s\n%s\n%.16e\n%f\n%.16e\n%i' %(fourier_in_data_fn,
                                                          fourier_out_data_fn,
                                                          dt/constants.th,
                                                          300,
                                                          fmax*fmax_extend_fac/(constants.c0*100),
                                                          1)
        common.file_write(fourier_in_fn, fourier_in_txt)
        np.savetxt(fourier_in_data_fn, fourier_in_data)
        common.system(fourier_exe + ' < ' + fourier_in_fn + ' > /dev/null')
        fourier_loaded_data = np.loadtxt(fourier_out_data_fn)
        # frequency axis fourier_out_data[:,0] is the same for all atoms, sum up
        # only power spectra
        if iatom == 0:
            fourier_out_data = fourier_loaded_data
        else:        
            fourier_out_data[:,1:] += fourier_loaded_data[:,1:]
        f8 = fourier_out_data[:,0]*(constants.c0*100)
        y8n = common.norm_int(fourier_out_data[:,1], f8)

figs.append(plt.figure())
axs.append(figs[-1].add_subplot(111))
axs[-1].set_title('3d arr, no mass')
axs[-1].plot(f4, y4n, label='3d vacf')
axs[-1].plot(f5, y5n, label='3d direct')
if use_fourier:
    axs[-1].plot(f8, y8n, label='3d fourier.x')
axs[-1].legend()

figs.append(plt.figure())
axs.append(figs[-1].add_subplot(111))
axs[-1].set_title('3d arr, with mass')
axs[-1].plot(f6, y6nm, label='3d vacf')
axs[-1].plot(f7, y7nm, label='3d direct')
axs[-1].legend()
 
plt.show()

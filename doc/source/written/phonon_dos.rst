Velocity autocorrelation function and phonon DOS
================================================

Correlation and power spectrum
------------------------------
See also :func:`pwtools.signal.acorr`, ``method=7``.

Correlation via fft. After ifft, the imaginary part is (in theory) =
0, in practise < 1e-16, so we are safe to return the real part only.

The cross-correlation theorem for the two-sided correlation::

  corr(a,b) = ifft(fft(a)*fft(b).conj())

If a == b (like here), then this reduces to the special case of the 
Wiener-Khinchin theorem (autocorrelation of `a`)::
  
  corr(a,a) = ifft(abs(fft(a))**2)

Both theorems assume *periodic* data, i.e. `a` and `b` repeat after `nstep`
points. To deal with non-periodic data, we use zero-padding at the end of `a`.
The result of ``ifft(...)`` contains the correlations for positive and negative
lags. Since the autocorrelation function is symmetric around lag=0, we return
0 ... +lag.

Here are these equalities with discrete data. Note that due to the
way in which fft/ifft packs the data in the returned array, we need
to do some slicing + mirror tricks to get it right. In each example,
the arrays c1,c2,c3 and p1,p2 are the same.

Two-sided correlation for -lag...0...+lag::
    
    >>> from pwtools.signal import pad_zeros, welch, mirror, acorr
    >>> from scipy.signal import correlate
    >>> from scipy.fftpack import fft,ifft
    >>> pad=lambda x: pad_zeros(x, nadd=len(x)-1)
    >>> n=50; v=rand(n); w=welch(n)

    >>> c1=mirror(ifft(abs(fft(pad(v)))**2.0)[:n].real)
    >>> c2=correlate(v,v,'full')
    >>> c3=mirror(acorr(v,norm=False))

and the power spectra as ``fft(corr(v,v))``, now one-sided::
    
    >>> p1=(abs(fft(pad(v)))**2.0)[:n]
    >>> p2=(abs(fft(mirror(acorr(v,norm=False)))))[:n]

also with a Welch window::    
    
    >>> p1=(abs(fft(pad(v*w)))**2.0)[:n]
    >>> p2=(abs(fft(mirror(acorr(v*w,norm=False)))))[:n]

The zero-padding must be done always! It is done inside
:func:`scipy.signal.correlate()`.  

Note that the two-sided correlation calculated like this is ``2*nstep-1`` long.

Padding and smoothing
---------------------

There is another code [tfreq]_ out there (appart from ``fourier.x`` from CPMD)
which calculates the phonon DOS from MD data. But what he does is padding the
`correlation` function, i.e. something like ``fft(pad(acorr(v)))``, which seems
odd b/c the padding must be done on `v` as outlined above. Also, he uses
smoothing (convolution with a gaussian, i.e. ``fft(smooth(pad(acorr(v))))``)
after padding, which is less effective than using a Welch (or any other) window
functin. But I haven't tested the code, so ...

For smoothing the spectrum using our implementation, either use more padding in
the case ``p1=(abs(fft(pad(v)))**2.0)[:n]`` or smooth the `spectrum` afterwards
by using :func:`pwtools.signal.smooth_convolve`.


Calculation of the phonon DOS from MD data in pwtools
-----------------------------------------------------

There are two ways of computing the phonon density of states (PDOS) from 
an MD trajectory (V is is array of atomic velocities, see pydos.velocity(). 

(1) vacf way: FFT of the velocity autocorrelation function (vacf):
    V -> VACF -> FFT(VACF) = PDOS, see pydos.vacf_pdos()
(2) direct way: ``|FFT(V)**2|`` = PDOS, see pydos.direct_pdos(), this is much
    faster and mathematically exactly the same, see examples/pdos_methods.py
    and test/test_pdos.py .

Both methods are implemented but actually only method (2) is worth using.
Method (1) still exists for historical reasons and as reference.

* In method (1), if you mirror the VACF at t=0 before the FFT, then you get
  double frequency resolution. 

* By default, direct_pdos() uses zero padding to get the same frequency
  resolution as you would get with mirroring the signal in vacf_pdos().

* Both methods use Welch windowing by default to reduce "leakage" from
  neighboring peaks. See also examples/pdos_methods.py 

* Both methods must produce exactly the same results (up to numerical noise).

* The frequency axis of the PDOS is in Hz. It is "f", NOT the angular frequency 
  2*pi*f. See also examples/pdos_methods.py .



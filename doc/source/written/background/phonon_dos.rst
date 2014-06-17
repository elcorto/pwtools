Velocity autocorrelation function and phonon DOS
================================================

Correlation and power spectrum
------------------------------
Here are some (textbook) notes about correlation, which you should read in
order to understand how the phonon DOS (= vibrational density of states =
power spectrum of the atomic velocities) is calculated in pwtools (see
:mod:`~pwtools.pydos`).

The cross-correlation theorem for the two-sided correlation::

  corr(a,b) = ifft(fft(a)*fft(b).conj())

If a == b, then this reduces to the special case of the Wiener-Khinchin theorem
(autocorrelation of `a`)::
  
  corr(a,a) = ifft(abs(fft(a))**2)

where the power spectrum of `a` is simply ``PSD = fft(corr(a,a)) == abs(fft(a))**2``.

Both theorems assume *periodic* data, i.e. `a` and `b` repeat after `nstep`
points. To deal with non-periodic data, we use zero-padding with ``nstep-1``
points at the end of `a`. Therefore, the correlated signal is ``2*nstep-1``
points long and contains the correlations for positive and negative lags. Since
the autocorrelation function is symmetric around lag=0, we return 0 ... +lag
in :func:`pwtools.signal.acorr`. To compare that with
``scipy.signal.correlate(a,a,'full')``, we need to mirror the result at lag=0
again.

Here are these equalities with discrete data. Note that due to the
way in which fft/ifft packs the data in the returned array, we need
to do some slicing + mirror tricks to get it right. In each example,
the arrays c1,c2,c3 and p1,p2 are the same and for ``corr(v,v)`` we use
``acorr(v)`` here.

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

The fft-based correlation is implemented, along with other methods, in
:func:`pwtools.signal.acorr`.

Padding and smoothing
---------------------

There is another code [tfreq]_ out there (appart from ``fourier.x`` from CPMD)
which calculates the phonon DOS from MD data. But what he does is padding the
`correlation` function, i.e. something like ``fft(pad(acorr(v)))``, which seems
odd b/c the padding must be done on `v` as outlined above. Also, he uses
smoothing (convolution with a gaussian, i.e. ``fft(smooth(pad(acorr(v))))``)
after padding, which is less effective than using a Welch (or any other) window
function. But I haven't tested the code, so ...

For smoothing the spectrum using our implementation, either use more padding
`of the time series` in the case ``p1=(abs(fft(pad(v)))**2.0)[:n]`` or smooth
the `spectrum` afterwards by using :func:`pwtools.signal.smooth`.


Calculation of the phonon DOS from MD data in pwtools
-----------------------------------------------------

There are two ways of computing the phonon density of states (PDOS) from an MD
trajectory (V is the 3d array of atomic velocities with shape (nstep,natoms,3),
i.e. ``Trajectory.velocity``, see :func:`~pwtools.crys.velocity_traj`. 

(1) vacf way: FFT of the velocity autocorrelation function (vacf):
    V -> VACF -> FFT(VACF) = PDOS, see :func:`~pwtools.pydos.vacf_pdos`
(2) direct way: ``|FFT(V)**2|`` = PDOS, see :func:`~pwtools.pydos.direct_pdos`,
    this is much faster and mathematically exactly the same, see
    ``examples/examples/phonon_dos`` and ``test/test_pdos.py`` .

Both methods are implemented but actually only method (2) is worth using.
Method (1) still exists for historical reasons and as reference.

The actual implementation is in :func:`~pwtools.pydos.pdos` and the above two
functions are convenience wrappers.

* In method (1), if you mirror the VACF at t=0 before the FFT, then you get
  double frequency resolution. 

* By default, direct_pdos() uses zero padding to get the same frequency
  resolution as you would get with mirroring the signal in vacf_pdos().
  Also, padding is necessary b/c of the arguments outlined above for the 1d
  case.

* Both methods use Welch windowing by default to reduce "leakage" from
  neighboring peaks.

* Both methods must produce exactly the same results (up to numerical noise).

* The frequency axis of the PDOS is in Hz. It is "f", NOT the angular frequency 
  2*pi*f. See also examples/pdos_methods.py .

* The difference to the 1d case: 
    * mass weighting: this affects only the relative peak `heights` in the
      PDOS, not the peak positions
    * averaging over `natoms` to get a 1d array (time series) 

Velocity autocorrelation function and phonon DOS
================================================

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

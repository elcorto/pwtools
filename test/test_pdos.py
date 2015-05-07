import numpy as np
import os, tempfile
from pwtools import parse, common, constants, common
from pwtools import pydos as pd
from pwtools.signal import pad_zeros, welch, mirror, acorr
from scipy.signal import correlate
from scipy.fftpack import fft,ifft
from pwtools.test.tools import aae
from pwtools.test.testenv import testdir
from pwtools.test import tools
rand = np.random.rand

def test_pdos():
    filename = tools.unpack_compressed('files/pw.md.out.gz', prefix=__file__)
    pp = parse.PwMDOutputFile(filename=filename)
    traj = pp.get_traj()

    # timestep dt
    # -----------
    # Only needed in pd.*_pdos(), not in pd.velocity(). Here is why:
    #
    # vacf_pdos, direct_pdos:
    # If we compute the *normalized* VCAF, then dt is a factor: 
    #       <v_x(0) v_x(t)> = 1/dt^2 <dx(0) dx(t)> 
    # which cancels in the normalization. dt is not needed in the velocity
    # calculation, hence not 
    #   V=velocity(coords, dt=dt) 
    # only
    #   V=velocity(coords).
       
    V = traj.velocity # Ang / fs
    mass = traj.mass # amu
    dt = traj.timestep # fs
    timeaxis = traj.timeaxis
    assert np.allclose(150.0, dt * constants.fs / constants.tryd) # dt=150 Rydberg time units
    fd, dd = pd.direct_pdos(V, m=mass, dt=dt, npad=1, tonext=False)
    fv, dv = pd.vacf_pdos(V, m=mass, dt=dt, mirr=True)

    np.testing.assert_array_almost_equal(fd, fv, err_msg="freq not equal")
    np.testing.assert_array_almost_equal(dd, dv, err_msg="dos not equal")
    
    assert np.allclose(fd, np.loadtxt('files/ref_test_pdos/fd.txt.gz'))
    assert np.allclose(fv, np.loadtxt('files/ref_test_pdos/fv.txt.gz'))
    assert np.allclose(dd, np.loadtxt('files/ref_test_pdos/dd.txt.gz'))
    assert np.allclose(dv, np.loadtxt('files/ref_test_pdos/dv.txt.gz'))

    df = fd[1] - fd[0]
    print "Nyquist freq: %e" %(0.5/dt)
    print "df: %e:" %df
    print "timestep: %f fs = %f tryd" %(dt, dt * constants.fs / constants.tryd)
    print "timestep pw.out: %f tryd" %(pp.timestep)
    
    # API
    fd, dd, ffd, fdd, si = pd.direct_pdos(V, m=mass, dt=dt, full_out=True)
    fv, dv, ffv, fdv, si, vacf, fft_vacf = pd.vacf_pdos(V, m=mass, dt=dt,
                                                        mirr=True, full_out=True)

    # Test padding for speed.
    fd, dd, ffd, fdd, si = pd.direct_pdos(V, m=mass, dt=dt, npad=1, tonext=True, \
                           full_out=True)
    assert len(fd) == len(dd)
    assert len(ffd) == len(fdd)
    # If `tonext` is used, full fft array lengths must be a power of two.
    assert len(ffd) >= 2*V.shape[timeaxis] - 1
    assert np.log2(len(ffd)) % 1.0 == 0.0


def test_pdos_1d():
    pad=lambda x: pad_zeros(x, nadd=len(x)-1)
    n=500; w=welch(n)
    # 1 second signal
    t=np.linspace(0,1,n); dt=t[1]-t[0]
    # sum of sin()s with random freq and phase shift, 10 frequencies from
    # f=0...100 Hz
    v=np.array([np.sin(2*np.pi*f*t + rand()*2*np.pi) for f in rand(10)*100]).sum(0)
    f=np.fft.fftfreq(2*n-1, dt)[:n]

    c1=mirror(ifft(abs(fft(pad(v)))**2.0)[:n].real)
    c2=correlate(v,v,'full')
    c3=mirror(acorr(v,norm=False))
    assert np.allclose(c1, c2)
    assert np.allclose(c1, c3)

    p1=(abs(fft(pad(v)))**2.0)[:n]
    p2=(abs(fft(mirror(acorr(v,norm=False)))))[:n]
    assert np.allclose(p1, p2)

    p1=(abs(fft(pad(v*w)))**2.0)[:n]
    p2=(abs(fft(mirror(acorr(v*w,norm=False)))))[:n]
    assert np.allclose(p1, p2)


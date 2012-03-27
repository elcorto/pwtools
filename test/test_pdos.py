import numpy as np
from pwtools import parse, common, constants
from pwtools import common
from pwtools import pydos as pd
from pwtools.crys import coord_trans
from pwtools.test.tools import aae

def test():
    filename = 'files/pw.md.out'
    common.system('gunzip %s.gz' %filename)
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
    aae(150.0, dt * constants.fs / constants.tryd) # dt=150 Rydberg time units
    fd, dd = pd.direct_pdos(V, m=mass, dt=dt, axis=timeaxis)
    fv, dv = pd.vacf_pdos(V, m=mass, dt=dt, mirr=True, axis=timeaxis)

    np.testing.assert_array_almost_equal(fd, fv, err_msg="freq not equal")
    np.testing.assert_array_almost_equal(dd, dv, err_msg="dos not equal")

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
    fd, dd, ffd, fdd, si = pd.direct_pdos(V, m=mass, dt=dt, pad_tonext=True, \
                           full_out=True)
    assert len(fd) == len(dd)
    assert len(ffd) == len(fdd)
    # If `pad_tonext` is used, full fft array lengths must be a power of two.
    assert len(ffd) >= 2*V.shape[timeaxis] - 1
    assert np.log2(len(ffd)) % 1.0 == 0.0

    common.system('gzip %s' %filename)

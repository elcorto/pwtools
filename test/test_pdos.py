import numpy as np
from pwtools import parse, common, constants
from pwtools import common
from pwtools import pydos as pd
from pwtools.crys import coord_trans

def test():
    filename = 'files/pw.md.out'
    infile = 'files/pw.md.in'
    common.system('gunzip %s.gz' %filename)
    pwout = parse.PwOutputFile(filename)
    pwin = parse.PwInputFile(infile)
    pwin.parse()
    pwout.parse()

    # Transform coords if needed. See .../pwtools/README .
    ibrav = int(pwin.namelists['system']['ibrav'])
    c_sys = pwin.atpos['unit'].lower().strip()
    if c_sys == 'crystal':
        if ibrav == 0:
            if pwin.cell is None:
                raise StandardError("error: no cell parameters in infile, "
                                    "set manually here")
            else:        
                coords = coord_trans(pwout.coords, old=pwin.cell,
                                     new=np.identity(3), align='rows') 
        else:
            raise StandardError("error: ibrav != 0, cannot get cell "
                "parameters from infile set manually here")
    else:
        coords = pwout.coords
    
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
    V = pd.velocity(coords)
    m = pwin.massvec
    dt = float(pwin.namelists['control']['dt'])*constants.tryd
    fd, dd = pd.direct_pdos(V, m=m, dt=dt)
    fv, dv = pd.vacf_pdos(V, m=m, dt=dt, mirr=True)

    np.testing.assert_array_almost_equal(fd, fv, err_msg="freq not equal")
    np.testing.assert_array_almost_equal(dd, dv, err_msg="dos not equal")

    df = fd[1] - fd[0]
    print "Nyquist freq [Hz]: %e" %(0.5/dt)
    print "df [Hz] %e:" %df

    ##from matplotlib import pyplot as plt
    ##plt.plot(fd, dd, label='direct')
    ##plt.plot(fv, dv, label='vacf')
    ##plt.legend()
    ##plt.show()

    common.system('gzip %s' %filename)

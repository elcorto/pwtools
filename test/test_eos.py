# test_eos.py
#
# Test if calling eos.x from the Elk code works. Compare to reference data.

import numpy as np
from pwtools.eos import ElkEOSFit
from pwtools import common
from testenv import testdir

def test():
    # This must be on your $PATH.
    exe = 'eos.x'
    app = common.backtick("which %s" %exe)
    if app == '':
        print("warning: cannot find '%s' on PATH, skipping test" %exe)
    else:
        # EV data
        data = np.loadtxt("files/ev/evdata.txt")
        volume = data[:,0]
        energy = data[:,1]
        # ref data, created with
        #     eos = ElkEOSFit(energy=energy,
        #                     volume=volume,
        #                     natoms=1,
        #                     etype=1,
        #                     npoints=300)
        # natoms=1 -> no normalitation in ref. data *.OUT                    
        ref_ev = np.loadtxt("files/ev/EVPAI.OUT.gz")
        ref_pv = np.loadtxt("files/ev/PVPAI.OUT.gz")
        assert ref_ev.shape[0] == ref_pv.shape[0], ("reference data lengths "
            "inconsistent")
        ref = {}        
        ref['ev_v'] = ref_ev[:,0]        
        ref['ev_e'] = ref_ev[:,1] * 2.0 # Ha -> Ry
        ref['pv_v'] = ref_pv[:,0]
        ref['pv_p'] = ref_pv[:,1]        
        
        # natoms = 1, no normalization
        eos = ElkEOSFit(energy=energy,
                        volume=volume,
                        natoms=1,
                        etype=1,
                        npoints=300,
                        dir=testdir)
        eos.fit()
        now = {}
        now['ev_v'], now['ev_e'] = eos.get_ev()
        now['pv_v'], now['pv_p'] = eos.get_pv()
        
        # internal consistence
        for key in now.iterkeys():
            print key
            assert (now[key] == getattr(eos, key)).all()
        
        # compare to reference
        for key, val in ref.iteritems():
            print key
            np.testing.assert_array_almost_equal(now[key], ref[key])
        
        # other getters
        print eos.get_min()
        x,y = eos.get_bv()

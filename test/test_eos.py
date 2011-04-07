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
        ref_min = np.loadtxt("files/ev/min.txt")
        assert ref_ev.shape[0] == ref_pv.shape[0], ("reference data lengths "
            "inconsistent")
        ref = {}        
        ref['ev_v'] = ref_ev[:,0]        
        ref['ev_e'] = ref_ev[:,1] * 2.0 # Ha -> Ry
        ref['pv_v'] = ref_pv[:,0]
        ref['pv_p'] = ref_pv[:,1]
        ref['v0'], ref['e0'], ref['p0'], ref['b0'] = ref_min
        
        eos_store = {}
        type_arr = type(np.array([1.0,2.0]))
        for method in ['ev', 'pv']:
            print "method: %s" %method
            # natoms = 1, no normalization
            eos = ElkEOSFit(energy=energy,
                            volume=volume,
                            natoms=1,
                            etype=1,
                            npoints=300,
                            dir=testdir,
                            method=method)
            eos.fit()
            now = {}
            now['ev_v'], now['ev_e'] = eos.get_ev()
            now['pv_v'], now['pv_p'] = eos.get_pv()
            now['v0'], now['e0'], now['p0'], now['b0'] = eos.get_min()
            
            # compare to reference
            for key, val in ref.iteritems():
                print key
                if type(val) == type_arr:
                    np.testing.assert_array_almost_equal(now[key], ref[key])
                else:
                    np.testing.assert_almost_equal(now[key], ref[key],
                                                   decimal=3)
            eos_store[method] = eos
        
        # Test other attrs between methods 'ev' and 'pv' among each other for
        # which we do not have external ref data.
        print "bv_v"
        np.testing.assert_array_almost_equal(eos_store['ev'].bv_v, 
                                             eos_store['pv'].bv_v)
        print "bv_b"
        np.testing.assert_array_almost_equal(eos_store['ev'].bv_b, 
                                             eos_store['pv'].bv_b,
                                             decimal=2)
        

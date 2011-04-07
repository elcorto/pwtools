# test_eos.py
#
# Test if calling eos.x from the Elk code works. Compare to reference data.
#
# ref data created with
#     eos = ElkEOSFit(energy=energy,
#                     volume=volume,
#                     natoms=1,
#                     etype=1,
#                     npoints=300)
# natoms=1 -> no normalitation in ref. data *.OUT 

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
        # EV input data
        data = np.loadtxt("files/ev/evdata.txt")
        volume = data[:,0]
        energy = data[:,1]
        # reference fitted data points
        ref_ev = np.loadtxt("files/ev/EVPAI.OUT.gz")
        ref_ev[:,1] *= 2.0 # Ha -> Ry
        ref_pv = np.loadtxt("files/ev/PVPAI.OUT.gz")
        ref_min = np.loadtxt("files/ev/min.txt")
        assert ref_ev.shape[0] == ref_pv.shape[0], ("reference data lengths "
            "inconsistent")
        ref = {}        
        ref['ev'] = ref_ev        
        ref['pv'] = ref_pv
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
            now['ev'] = eos.ev
            now['pv'] = eos.pv
            now.update(eos.get_min())
            
            # compare to reference
            for key, val in ref.iteritems():
                print key
                if type(val) == type_arr:
                    np.testing.assert_array_almost_equal(now[key], ref[key])
                else:
                    np.testing.assert_almost_equal(now[key], ref[key],
                                                   decimal=3)
            eos_store[method] = eos
            
            # internal check: are the splines correct?
            for name in ['ev', 'pv', 'bv']:
                # (N,2) arrays self.{ev,pv,bv}
                data = getattr(eos, name)
                vv = data[:,0]
                yy = data[:,1]
                # self.spl_{ev,pv,bv}
                spl = getattr(eos, 'spl_' + name)
                np.testing.assert_array_almost_equal(yy, spl(vv))


        # Other attrs for which we do not have external ref data. Test only
        # among the two methods 'ev' and 'pv'.
        print "bv"
        np.testing.assert_array_almost_equal(eos_store['ev'].bv, 
                                             eos_store['pv'].bv,
                                             decimal=2)
        

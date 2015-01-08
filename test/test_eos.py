# test_eos.py
#
# Test if calling eos.x from the Elk code works. Compare to reference data.
# Also test new EosFit class.
#
# ref data created with
#     eos = ElkEOSFit(energy=energy,
#                     volume=volume,
#                     natoms=1,
#                     etype=1,
#                     npoints=300)
# natoms=1 -> no normalitation in ref. data *.OUT 
#
# Note: Ref data generated w/ old units Ry, Bohr, we convert to eV, Ang here

import numpy as np
from pwtools.eos import ElkEOSFit, EosFit
from pwtools import common, num
from pwtools.constants import Ry, Ha, Bohr, Ang, eV, eV_by_Ang3_to_GPa
from pwtools.test import tools
from testenv import testdir

Bohr3_to_Ang3 = (Bohr**3 / Ang**3)

def test_eos():
    # load reference fitted with ElkEOSFit, data points [Bohr^3, Ha] -> [Ang^3, eV]
    # EV input data [Bohr^3, Ry] -> [Ang^3, eV]
    data = np.loadtxt("files/ev/evdata.txt")
    volume = data[:,0] * Bohr3_to_Ang3
    energy = data[:,1] * (Ry / eV)
    ref_ev = np.loadtxt("files/ev/EVPAI.OUT.gz")
    ref_ev[:,0] *= Bohr3_to_Ang3
    ref_ev[:,1] *= (Ha / eV)
    ref_pv = np.loadtxt("files/ev/PVPAI.OUT.gz")
    ref_pv[:,0] *= Bohr3_to_Ang3
    ref_min = np.loadtxt("files/ev/min.txt")
    ref_min[0] *= Bohr3_to_Ang3 # v0
    ref_min[1] *= (Ry / eV)     # e0
    assert ref_ev.shape[0] == ref_pv.shape[0], ("reference data lengths "
        "inconsistent")
    ref = {}        
    ref['ev'] = ref_ev        
    ref['pv'] = ref_pv
    ref['v0'], ref['e0'], ref['p0'], ref['b0'] = ref_min

    # test new EosFit class, default func=Vinet()
    eos = EosFit(volume=volume,
                 energy=energy)
    assert np.allclose(eos.params['v0'], eos.spl.get_min())
    assert np.allclose(eos.params['v0'], eos.get_min())
    assert np.allclose(eos.params['e0'], eos(eos.params['v0']))
    assert np.allclose(eos.params['b0']*eV_by_Ang3_to_GPa, eos.bulkmod(eos.params['v0']))
    now = {}
    now['v0'] = eos.params['v0']
    now['e0'] = eos.params['e0']
    now['b0'] = eos.params['b0'] * eV_by_Ang3_to_GPa
    now['p0'] = eos.pressure(eos.params['v0'])
    for key,val in now.iteritems():
        msg = "EosFit: key=%s, ref=%e, val=%e" %(key, ref[key], val)
        assert np.allclose(val, ref[key], atol=1e-7), msg

    # Test legacy ElkEOSFit / ExternEOS. 
    # 'exe' must be on your $PATH.
    exe = 'eos.x'
    app = common.backtick("which %s" %exe).strip()
    if app == '':
        tools.skip("cannot find '%s' on PATH, skipping test" %exe)
    else:
        eos_store = {}
        type_arr = type(np.array([1.0,2.0]))
        for bv_method in ['ev', 'pv']:
            print "bv_method: %s" %bv_method
            # natoms = 1, no normalization
            eos = ElkEOSFit(energy=energy,
                            volume=volume,
                            natoms=1,
                            etype=1,
                            npoints=300,
                            dir=testdir,
                            bv_method=bv_method)
            eos.fit()
            now = {}
            now['ev'] = eos.ev
            now['pv'] = eos.pv
            now.update(eos.get_min())
            
            # compare to reference
            for key, val in ref.iteritems():
                print "ElkEOSFit: testing:", key
                if type(val) == type_arr:
                    np.testing.assert_array_almost_equal(now[key], ref[key])
                else:
                    np.testing.assert_almost_equal(now[key], ref[key],
                                                   decimal=3)
            eos_store[bv_method] = eos
            
            # internal check: are the splines correct?
            for name in ['ev', 'pv', 'bv']:
                # API
                getter = getattr(eos, 'get_spl_' + name)
                assert getattr(eos, 'spl_' + name) == getter()
                # (N,2) arrays self.{ev,pv,bv}
                data = getattr(eos, name)
                vv = data[:,0]
                yy = data[:,1]
                # self.spl_{ev,pv,bv}
                spl = getattr(eos, 'spl_' + name)
                np.testing.assert_array_almost_equal(yy, spl(vv))

        # Other attrs for which we do not have external ref data. Test only
        # among the two bv_methods 'ev' and 'pv'.
        print "bv"
        np.testing.assert_array_almost_equal(eos_store['ev'].bv, 
                                             eos_store['pv'].bv,
                                             decimal=2)

def test_eos_fit_deriv():
    data = np.loadtxt("files/ev/evdata.txt")
    volume = data[:,0] * Bohr3_to_Ang3
    energy = data[:,1] * (Ry / eV)
    
    eos = EosFit(volume=volume,
                 energy=energy,
                 splpoints=500)
    
    # EosFit is a num.Fit1D subclass, so it has self.x and self.y
    assert (volume == eos.x).all()
    assert (energy == eos.y).all()
    
    # this reference to self.spl causes self.spl to be defined (lazyprop)               
    xx = eos.spl.x             
    yy = eos.spl.y             
    assert len(xx) == len(yy) == 500
    
    # spline thru fitted data, must be exactly like eos.spl
    spl = num.Spline(xx, yy, k=5, s=None)
    
    assert np.allclose(eos(xx), yy)             # call _vinet, consistency of fitted data
    assert np.allclose(eos(xx), spl(xx))        # consistency of splines
    assert np.allclose(eos(xx), eos.spl(xx))    # consistency of splines
    assert np.allclose(eos(xx, der=1), spl(xx, der=1)) # call _vinet_deriv1
    assert np.allclose(eos(xx, der=2), spl(xx, der=2)) # call _vinet_deriv2
    assert np.allclose(eos(xx, der=3), spl(xx, der=3)) # call eos.spl(xx, der=3)


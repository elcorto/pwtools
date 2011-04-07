# test_qha.py
#
# Test thermo.HarmonicThermo against results from F_QHA.f90 from QE 4.2 .

import numpy as np
from pwtools.thermo import HarmonicThermo
from pwtools import common
from pwtools.constants import Ry_to_J

def assrt_aae(*args, **kwargs):
    np.testing.assert_array_almost_equal(*args, **kwargs)

def test():
    def pack(fns):
        for fn in fns:
            common.system('gzip %s' %fn)

    def unpack(fns):
        for fn in fns:
            common.system('gunzip %s' %fn)

    class Store(object):
        def __init__(self, arr1, arr2):
            self.arr1 = arr1
            self.arr2 = arr2

    fqha_fn = 'files/fqha.out'
    pdos_fn = 'files/si.phdos'
    files = [fqha_fn, pdos_fn]
    unpack([x + '.gz' for x in files])

    fqha = np.loadtxt(fqha_fn)
    pdos = np.loadtxt(pdos_fn)
    temp = fqha[:,0] 

    #--------------------------------------------------------------------
    # Verify against ref data
    #--------------------------------------------------------------------
    
    # No nan warnings, b/c we do not take all data into account. However
    # assert_array_almost_equal() will fail with decimal=3.
    # 
    ##ha = HarmonicThermo(pdos[3:,0], pdos[3:,1], temp, fixzero=True, fixnan=True,
    ##                    checknan=True)

    
    # all data, mmaybe get some nans, but assert_array_almost_equal() will pass
    # with decimal=3 .
    ha = HarmonicThermo(pdos[:,0], pdos[:,1], temp, fixzero=True, fixnan=True,
                        checknan=True)

    dct = {'evib': Store(arr1=ha.evib(), arr2=fqha[:,1]),
           'fvib': Store(arr1=ha.fvib(), arr2=fqha[:,2]),
           'cv':   Store(arr1=ha.cv(),   arr2=fqha[:,3]),
           'svib': Store(arr1=ha.svib(), arr2=fqha[:,4]),
           }
    for key, store in dct.iteritems():
        assrt_aae(store.arr1, store.arr2, decimal=2)
    ##from matplotlib import pyplot as plt        
    ##    plt.figure()
    ##    plt.plot(temp, store.arr1, label='%s: ha'%key)
    ##    plt.plot(temp, store.arr2, label='%s: fqha'%key)
    ##    plt.plot(temp, store.arr1 - store.arr2, label='%s: diff'%key)
    ##    plt.legend()
    ##plt.show()
    
    #--------------------------------------------------------------------
    # Consistency
    #--------------------------------------------------------------------
    
    # Fvib = Evib -T*Svib
    assrt_aae(dct['fvib'].arr1*Ry_to_J, 
              (dct['evib'].arr1 - temp*dct['svib'].arr1)*Ry_to_J)
    
    #--------------------------------------------------------------------
    # API tests
    #--------------------------------------------------------------------

    # use temp arg in methods
    ha = HarmonicThermo(pdos[:,0], pdos[:,1], fixzero=True, fixnan=True,
                        checknan=True)
    x=ha.evib(temp)                        
    x=ha.fvib(temp)
    x=ha.cv(temp)
    x=ha.svib(temp)
    pack(files)
    
    # test fix* 
    freq = np.linspace(1, 10, 100)
    dos = freq**2.0
    temp = np.linspace(10, 100, 100)
    # fixzero
    freq[0] = 0.0
    ha = HarmonicThermo(freq=freq, 
                        dos=dos, 
                        fixzero=False, 
                        checknan=True, 
                        fixnan=False, 
                        fixneg=False)
    assert ha.f[0] == 0.0 
    ha = HarmonicThermo(freq=freq, 
                        dos=dos, 
                        fixzero=True, 
                        checknan=True, 
                        fixnan=False, 
                        fixneg=False)
    assert ha.f[0] > 0.0                        
    # fixneg 
    freq[0] = -100.0
    ha = HarmonicThermo(freq=freq, 
                        dos=dos, 
                        fixzero=False, 
                        checknan=True, 
                        fixnan=False, 
                        fixneg=False)
    assert ha.f[0] == -100                
    ha = HarmonicThermo(freq=freq, 
                        dos=dos, 
                        fixzero=False, 
                        checknan=True, 
                        fixnan=False, 
                        fixneg=True)
    assert ha.f[0] > 0.0                
                

# test_qha.py
#
# Test thermo.HarmonicThermo against results from F_QHA.f90 from QE 4.2 .

def test():
    import numpy as np
    from pwtools.thermo import HarmonicThermo
    from pwtools import common
    
    # Verify result tests

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
    T = fqha[:,0] 

    #
    # No nan warnings, b/c we do not take all data into account. However
    # assert_array_almost_equal() will fail with decimal=3.
    # 
    ##ha = HarmonicThermo(pdos[3:,0], pdos[3:,1], T, fixzero=True, fixnan=True,
    ##                    checknan=True)

    #
    # all data, mmaybe get some nans, but assert_array_almost_equal() will pass
    # with decimal=3 .
    #
    ha = HarmonicThermo(pdos[:,0], pdos[:,1], T, fixzero=True, fixnan=True,
                        checknan=True)

    dct = {'evib': Store(arr1=ha.evib(), arr2=fqha[:,1]),
           'fvib': Store(arr1=ha.fvib(), arr2=fqha[:,2]),
           'cv':   Store(arr1=ha.cv(),   arr2=fqha[:,3]),
           'svib': Store(arr1=ha.svib(), arr2=fqha[:,4]),
           }
    for key, store in dct.iteritems():
        np.testing.assert_array_almost_equal(store.arr1, store.arr2, decimal=2)
    ##from matplotlib import pyplot as plt        
    ##    plt.figure()
    ##    plt.plot(T, store.arr1, label='%s: ha'%key)
    ##    plt.plot(T, store.arr2, label='%s: fqha'%key)
    ##    plt.plot(T, store.arr1 - store.arr2, label='%s: diff'%key)
    ##    plt.legend()
    ##plt.show()
    

    # API tests

    # use temp arg in methods
    ha = HarmonicThermo(pdos[:,0], pdos[:,1], fixzero=True, fixnan=True,
                        checknan=True)
    x=ha.evib(T)                        
    x=ha.fvib(T)
    x=ha.cv(T)
    x=ha.svib(T)
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
                

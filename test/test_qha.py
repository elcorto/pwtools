# test_qha.py
#
# Test thermo.HarmonicThermo against results from F_QHA.f90 from QE 4.2 .

import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from scipy.integrate import simpson as simps
from pwtools.thermo import HarmonicThermo
from pwtools import common
from pwtools.constants import Ry_to_J, eV, Ry, kb
from pwtools.test.tools import unpack_compressed

def msg(txt):
    bar = '-'*79
    print(bar)
    print(txt)
    print(bar)

class Store:
    def __init__(self, arr1, arr2):
        self.arr1 = arr1
        self.arr2 = arr2


def test_qha():
    fqha_fn = unpack_compressed('files/fqha.out.gz')
    pdos_fn = unpack_compressed('files/si.phdos.gz')
    fqha = np.loadtxt(fqha_fn)
    pdos = np.loadtxt(pdos_fn)
    temp = fqha[:,0]

    msg('Verify against ref data')

    ha = HarmonicThermo(pdos[:,0], pdos[:,1], temp, skipfreq=True)

    # refenence Evib + Fvib [Ry], need to convert. Cv and Svib [kb].
    dct = {'evib': Store(arr1=ha.evib(), arr2=fqha[:,1]*Ry/eV),
           'fvib': Store(arr1=ha.fvib(), arr2=fqha[:,2]*Ry/eV),
           'cv':   Store(arr1=ha.cv(),   arr2=fqha[:,3]),
           'svib': Store(arr1=ha.svib(), arr2=fqha[:,4]),
           }
    for key, store in dct.items():
        aaae(store.arr1, store.arr2, decimal=2)

    msg('Consistency')

    # Fvib = Evib -T*Svib
    aaae(dct['fvib'].arr1,
        (dct['evib'].arr1 - temp*dct['svib'].arr1*kb/eV))

    msg('API tests')

    # use temp arg in methods
    ha = HarmonicThermo(pdos[:,0], pdos[:,1], skipfreq=True)
    x=ha.evib(temp)
    x=ha.fvib(temp)
    x=ha.cv(temp)
    x=ha.svib(temp)

    HarmonicThermo(pdos[:,0], pdos[:,1], T=temp)
    with pytest.deprecated_call():
        HarmonicThermo(pdos[:,0], pdos[:,1], temp=temp)

    msg('skip and fix')
    freq = np.linspace(1, 10, 100)
    dos = freq**2.0
    temp = np.linspace(10, 100, 100)
    freq[0] = 0.0
    ha = HarmonicThermo(freq=freq,
                        dos=dos,
                        skipfreq=False)
    assert ha.f[0] == 0.0
    ha = HarmonicThermo(freq=freq,
                        dos=dos,
                        skipfreq=True)
    assert ha.f[0] > 0.0
    freq[0] = -100.0
    ha = HarmonicThermo(freq=freq,
                        dos=dos,
                        skipfreq=False)
    assert ha.f[0] == -100
    ha = HarmonicThermo(freq=freq,
                        dos=dos,
                        skipfreq=True)
    assert ha.f[0] > 0.0

    msg('API: dosarea + integrator')
    area = np.random.rand()*10
    ha = HarmonicThermo(pdos[:,0], pdos[:,1], skipfreq=True, dosarea=area,
                        integrator=simps)
    assert np.allclose(simps(ha.dos, x=ha.f), area)

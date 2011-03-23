# thermo.py
#
# (Quasi)harmonic approximation.

import numpy as np
from scipy.integrate import simps
from pwtools.constants import kb, h, R, pi, c0, Ry_to_J
from pwtools.verbose import verbose


def coth(x):
    return 1.0/np.tanh(x)


class HarmonicThermo(object):
    """Calculate vibrational internal energy (Evib), free energy (Fvib),
    entropy (Svib) and isochoric heat capacity (Cv) in the harmonic
    approximation from a phonon density of states. 
    """    
    def __init__(self, freq, dos, temp, hplanck=h, kb=kb, fixzero=True,
                 checknan=True, fixnan=False):
        """                 
        args:
        -----
        freq : 1d array
            frequency f (NOT 2*pi*f) in cm^-1
        dos : 1d array
            phonon dos such that int(freq) dos = 3*natom
        temp : 1d array
            temperature range [K]
        hplanck, kb : Planck [J*s] / Boltzmann [J/K] constants
        fixzero : bool
            Handle zero frequencies and other values by setting them to smth
            like 1.5*eps. This prevents NaNs in subsequent calculations. This
            is a safe operation and should not perturb the results.
        checknan : bool
            Warn about found NaNs. To actually fix them, set fixnan=True.
        fixnan : bool
            Currently, set all NaNs to 0.0 if checknan=True. This is
            a HACK b/c we must assume that these numbers should be 0.0.
            Use if YKWYAD.
        """
        # notes:
        # ------
        # - This is actually a re-implementation of F_QHA.f90 found in Quantum
        #   Espresso as of v4.2.
        # - All relations can be found in M.T. Dove, Introduction to Lattice
        #   Dynamics, ch. 5 .
        # - The frequency axis "f" in cm^-1 is what QE's matdyn.x returns
        #   when it calculates the phonon DOS (input: dos=.true.).
        # - For high T, Cv in units of R, the universal gas constant, should
        #   approach 3*N where N = natom = atoms in the unit cell. This is the
        #   Dulong-Petit limit (usually 3*N*R, here 3*N).
        #
        # Theory (example Cv):
        # --------------------
        #
        # Let Z = hbar*w/(2*kb*T), D(w) = phonon dos.
        # Cv(T) = kb * Z**2 * Int(w) [w**2 * D(w) / sinh(z))**2]
        #
        # Cv is in J/K. To get Cv in R[J/(mol*K)], one would have to do 
        #   Cv[J/K] * Navo[1/mol] / R = Cv[J/K] / kb[J/K]
        # since kb = R/Navo.  
        #
        # We save the division by "kb" by dropping the "kb" prefactor:
        #   Cv(T) = Z**2 * Int(w) [w**2 * D(w) / sinh(z))**2]
        #          ^^^^
        # random note:
        #
        # in F_QHA.f90:
        #   a3 = 1.0/8065.5/8.617e-5 # = hbar*c0*100*2*pi / kb
        #
        # Must convert freq to w=2*pi*freq for hbar*w. Or, use 
        # h*freq.
        #
        # cm^-1 -> 1/s       : * c0*100
        # 1/s   -> cm^-1     : / (c0*100)
        # s     -> cm        : * c0*100
        # => hbar or h: J*s -> J*cm : *c0*100
        self.f = freq
        self.dos = dos
        self.T = temp
        self.h = hplanck * c0 * 100
        self.kb = kb
        self.fixnan = fixnan
        self.checknan = checknan

        self.eps = np.finfo(float).eps

        if fixzero:
            self.f = self._fixzero(self.f, copy=True)
    
    def _fixzero(self, arr, copy=True):
        arr2 = arr.copy() if copy else arr
        arr2[arr2 < 1.5*self.eps] = 1.5*self.eps
        return arr2

    def _integrate(self, y, f):
        # y: 2d array (len(T), len(dos)), integrate along axis=1
        if self.checknan:
            mask = np.isnan(y)
            if mask.any():
                print("HarmonicThermo._integrate: warning: NaNs found!")
                if self.fixnan:
                    print("HarmonicThermo._integrate: warning: fixing NaNs!")
                    y[mask] = 0.0
        return np.array([simps(y[i,:], f) for i in range(y.shape[0])])
    
    def vibrational_internal_energy(self):
        h, f, T, kb, dos = self.h, self.f, self.T, self.kb, self.dos
        arg = h * f / (kb*T[:,None])
        # 1/[ exp(x) -1] = NaN for x=0, that's why we use _fixzero(): For
        # instance
        #   1/(exp(1e-17) - 1) = NaN
        #   1/(exp(3e-16) - 1) = 4503599627370496.0
        arg = self._fixzero(arg, copy=False)
        y = dos * f * (0.5 + 1.0 / (np.exp(arg) - 1.0))
        eint = self._integrate(y, f) 
        return eint * (h/ Ry_to_J) # [Ry]
        
    def isochoric_heat_capacity(self):
        h, f, T, kb, dos = self.h, self.f, self.T, self.kb, self.dos
        arg = h * f / (kb*T[:,None])
        y = dos * f**2 / np.sinh(arg / 2.0)**2.0
        fac = (h / (kb*2*T))**2
        cv = self._integrate(y, f) * fac
        return cv # [R]
    
    def vibrational_free_energy(self):
        h, f, T, kb, dos = self.h, self.f, self.T, self.kb, self.dos
        arg = h * f / (kb*T[:,None])
        y = dos * np.log(2.0*np.sinh(arg/2.0))
        ret = self._integrate(y,f) * T
        return ret * (kb / Ry_to_J) # [Ry]
    
    def vibrational_entropy(self):
        h, f, T, kb, dos = self.h, self.f, self.T, self.kb, self.dos
        arg = h * f / (kb*T[:,None])
        y = dos * (0.5/T[:,None] * (h / kb) * f * coth(arg/2.0) - 
                   np.log(2.0*np.sinh(arg/2.0)))
        ret = self._integrate(y,f)
        return ret # [kb]
    
    def evib(self):
        return self.vibrational_internal_energy()

    def cv(self):
        return self.isochoric_heat_capacity()
    
    def fvib(self):
        return self.vibrational_free_energy()
    
    def svib(self):
        return self.vibrational_entropy()


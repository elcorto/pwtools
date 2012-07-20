# thermo.py
#
# (Quasi)harmonic approximation.

import numpy as np
from scipy.integrate import simps
from pwtools.constants import kb, hplanck, R, pi, c0, Ry_to_J, eV
from pwtools.verbose import verbose
from pwtools import crys


def coth(x):
    return 1.0/np.tanh(x)


class HarmonicThermo(object):
    """Calculate vibrational internal energy (Evib [eV]), free energy (Fvib
    [eV]), entropy (Svib [R,kb]) and isochoric heat capacity (Cv [R,kb]) in the
    harmonic approximation from a phonon density of states. 
    """    
    def __init__(self, freq, dos, temp=None, hplanck=hplanck, kb=kb, fixzero=True,
                 checknan=True, fixnan=False, fixneg=False, warn=True):
        """                 
        Parameters
        ----------
        freq : 1d array
            frequency f (NOT 2*pi*f) [cm^-1]
        dos : 1d array
            phonon dos such that int(freq) dos = 3*natom
        temp : 1d array, optional
            temperature range [K], if not given in the constructor then use
            `temp` in the calculation methods
        hplanck, kb : Planck [J*s] / Boltzmann [J/K] constants
        fixzero : bool
            Handle zero frequencies and other values by setting them to smth
            like 1.5*eps. This prevents NaNs in subsequent calculations. This
            is a safe operation and should not perturb the results.
        checknan : bool
            Warn about found NaNs. To actually fix them, set fixnan=True.
        fixnan : bool
            Use if YKWYAD, test before using! Currently, set all NaNs to 0.0 if
            checknan=True. This is a HACK b/c we must assume that these numbers
            should be 0.0.
        fixneg : bool
            Use if YKWYAD, test before using! Same as fixnan, but for negative
            numbers. Sometimes, a few frequencies (ususally the 1st entry only)
            are close to zero and negative. Set them to a very small positive
            value. The rms of the fixed values is printed. The default is False
            b/c it may hide large negative frequencies (i.e. unstable
            structure), which is a perfectly valid result (but you shouldn't do
            thermodynamics with that :)
        warn : bool
            Turn warnings on and off. Use only if you know you can ignore them.
        """
        # Notes
        # -----
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
        # since kb = R/Navo, with 
        #   R = gas constant = 8.314 J/(mol*K)
        #   Navo = Avogadro's number = 6e23
        # 
        # So, Cv [R] == Cv [kb]. The same holds for the entropy Svib.
        #
        # We save the division by "kb" by dropping the "kb" prefactor:
        #   Cv(T) = Z**2 * Int(w) [w**2 * D(w) / sinh(z))**2]
        #          ^^^^
        # random note:
        #
        # in F_QHA.f90:
        #   a3 = 1.0/8065.5/8.617e-5 # = hbar*c0*100*2*pi / kb
        #   Did you know that?
        #
        # All formulas (cv, fvib etc) are written for angular frequency
        # w=2*pi*freq. Either we use hbar*w or h*freq. We do the latter.
        # We also convert s -> cm and keep freq in [cm^-1].
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
        self.fixneg = fixneg
        self.fixzero = fixzero
        self.checknan = checknan
        self.warn = warn
        self.tiny = 1.5 * np.finfo(float).eps
        
        # order is important!
        if self.fixneg:
            self.f = self._fix(self.f, 'neg', copy=True)
        if self.fixzero:
            self.f = self._fix(self.f, 'zero', copy=True)
    
    def _printwarn(self, msg):
        if self.warn:
            print(msg)

    def _fix(self, arr, what='zero', copy=True):
        arr2 = arr.copy() if copy else arr
        if what == 'zero':
            idxs = np.abs(arr2) <= self.tiny
            if idxs.any():
                rms = crys.rms(arr2[idxs])
                self._printwarn("HarmonicThermo._fix: warning: "
                    "fixing zeros!, rms=%e" %rms)
            arr2[idxs] = self.tiny
        elif what == 'neg':
            idxs = arr2 < 0.0
            if idxs.any():
                rms = crys.rms(arr2[idxs])
                self._printwarn("HarmonicThermo._fix: warning: "
                    "fixing negatives!, rms=%e" %rms) 
            arr2[idxs] = 2*self.tiny
        else:
            raise StandardError("unknown method: '%s'" %what)
        return arr2
    

    def _integrate(self, y, f):
        """
        Integrate `y` along axis=1, i.e. over freq axis for all T.

        Parameters
        ----------
        y : 2d array (nT, ndos) where nT = len(self.T), ndos = len(self.dos)
        f : self.f, (len(self.dos),)

        Returns
        -------
        array (nT,)
        """
        if self.checknan:
            mask = np.isnan(y)
            if mask.any():
                self._printwarn("HarmonicThermo._integrate: warning: NaNs found!")
                if self.fixnan:
                    self._printwarn("HarmonicThermo._integrate: warning: fixing NaNs!")
                    y[mask] = 0.0
        return np.array([simps(y[i,:], f) for i in range(y.shape[0])])
    
    def _get_temp(self, temp):
        if (self.T is None) and (temp is None):
            raise ValueError("temp input and self.T are None")
        return self.T if temp is None else temp       

    def vibrational_internal_energy(self, T=None):
        """Evib [eV]"""
        h, f, kb, dos = self.h, self.f, self.kb, self.dos
        T = self._get_temp(T)
        arg = h * f / (kb*T[:,None])
        # 1/[ exp(x) -1] = NaN for x=0, that's why we use _fix('zero'): For
        # instance
        #   1/(exp(1e-17) - 1) = NaN
        #   1/(exp(3e-16) - 1) = 4503599627370496.0
        arg = self._fix(arg, 'zero', copy=False)
        y = dos * f * (0.5 + 1.0 / (np.exp(arg) - 1.0))
        eint = self._integrate(y, f) 
        return eint * (h/eV)
        
    def isochoric_heat_capacity(self, T=None):
        """Cv [R, kb]"""
        h, f, kb, dos = self.h, self.f, self.kb, self.dos
        T = self._get_temp(T)
        arg = h * f / (kb*T[:,None])
        y = dos * f**2 / np.sinh(arg / 2.0)**2.0
        fac = (h / (kb*2*T))**2
        cv = self._integrate(y, f) * fac
        return cv
    
    def vibrational_free_energy(self, T=None):
        """Fvib [eV]"""
        h, f, kb, dos = self.h, self.f, self.kb, self.dos
        T = self._get_temp(T)
        arg = h * f / (kb*T[:,None])
        y = dos * np.log(2.0*np.sinh(arg/2.0))
        ret = self._integrate(y,f) * T
        return ret * (kb / eV)
    
    def vibrational_entropy(self, T=None):
        """Svib [R, kb]"""
        h, f, kb, dos = self.h, self.f, self.kb, self.dos
        T = self._get_temp(T)
        arg = h * f / (kb*T[:,None])
        y = dos * (0.5/T[:,None] * (h / kb) * f * coth(arg/2.0) - 
                   np.log(2.0*np.sinh(arg/2.0)))
        ret = self._integrate(y,f)
        return ret
    
    def evib(self, *args, **kwargs):
        return self.vibrational_internal_energy(*args, **kwargs)

    def cv(self, *args, **kwargs):
        return self.isochoric_heat_capacity(*args, **kwargs)
    
    def fvib(self, *args, **kwargs):
        return self.vibrational_free_energy(*args, **kwargs)
    
    def svib(self, *args, **kwargs):
        return self.vibrational_entropy(*args, **kwargs)


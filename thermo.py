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
    def __init__(self, freq, dos, temp=None, skipfreq=False, 
                 eps=1.5*np.finfo(float).eps, fixnan=False, nanfill=0.0, 
                 verbose=True):
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
        skipfreq : bool, optional
            Ignore frequencies and DOS values where the frequencies are
            negative or close to zero, i.e. all DOS curve values where `freq` <
            `eps`. The number and rms of the skipped values is printed if
            `verbose=True`.
        eps : float, optional
            Threshold for `skipfreq`. Default is ~1.5*2.2e-16 . 
        fixnan : bool, optional
            Use if YKWYAD, test before using! Currently, set all NaNs occuring
            during integration to `nanfill`. This is a HACK b/c we must assume
            that these numbers should be `nanfill`.
        nanfill : float, optional
            During integration over temperature, set NaNs to this value.
        verbose : bool, optional
            Print warnings. Recommended for testing.

        Notes
        -----
        `skipfreq` and `fixnan`: Sometimes, a few frequencies (ususally the 1st
        few values only) are close to zero and negative, and the DOS is very
        small there. `skipfreq` can be used to ignore this region. The default
        is False b/c it may hide large negative frequencies (i.e. unstable
        structure), which is a perfectly valid result (but you shouldn't do
        thermodynamics with that :) Even if there are no negative frequencies,
        you can have frequencies (usually the first) beeing exactly zero or
        close to that (order 1e-17). That can cause numerical problems (NaNs)
        in some calculations so we may skip them and their DOS values, which
        must be assumed to be small. If you still encounter NaNs during
        integration, you may use `fixnan` to set them to `nanfill`. But that is a
        hack. If you cannot get rid of NaNs by `skipfreq`, then your freq-dos
        data is probably fishy!
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
        self.skipfreq = skipfreq
        self.verbose = verbose
        self.eps = eps
        self.nanfill = nanfill
        
        assert len(self.f) == len(self.dos), ("freq and dos don't have "
                                             "equal length")
        if self.verbose:
            print "number of points: %i" %len(self.f)
        
        if self.skipfreq:
            mask = self.f > self.eps
            if self.verbose:
                imask = np.invert(mask)
                nskip = len(imask.nonzero()[0])
                if len(imask) > 0:
                    frms = crys.rms(self.f[imask])
                    drms = crys.rms(self.dos[imask])
                    self._printwarn("HarmonicThermo: skipping %i frequencies: "
                        "rms=%e" %(nskip, frms))
                    self._printwarn("HarmonicThermo: skipping %i dos values: "
                        "rms=%e" %(nskip,drms))
            self.f = self.f[mask]
            self.dos = self.dos[mask]

    def _printwarn(self, msg):
        if self.verbose:
            print(msg)

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
        mask = np.isnan(y)
        if mask.any():
            self._printwarn("HarmonicThermo._integrate: warning: "
                            " %i NaNs found in y!" %len(mask))
            if self.fixnan:
                self._printwarn("HarmonicThermo._integrate: warning: "
                                "fixing %i NaNs in y!" %len(mask))
                y[mask] = self.nanfill
        return np.array([simps(y[i,:], f) for i in range(y.shape[0])])
    
    def _get_temp(self, temp):
        if (self.T is None) and (temp is None):
            raise ValueError("temp input and self.T are None")
        return self.T if temp is None else temp       

    def vibrational_internal_energy(self, temp=None):
        """Evib [eV]"""
        h, f, kb, dos = self.h, self.f, self.kb, self.dos
        T = self._get_temp(temp)
        arg = h * f / (kb*T[:,None])
        y = dos * f * (0.5 + 1.0 / (np.exp(arg) - 1.0))
        eint = self._integrate(y, f) 
        return eint * (h/eV)
        
    def isochoric_heat_capacity(self, temp=None):
        """Cv [R, kb]"""
        h, f, kb, dos = self.h, self.f, self.kb, self.dos
        T = self._get_temp(temp)
        arg = h * f / (kb*T[:,None])
        y = dos * f**2 / np.sinh(arg / 2.0)**2.0
        fac = (h / (kb*2*T))**2
        cv = self._integrate(y, f) * fac
        return cv
    
    def vibrational_free_energy(self, temp=None):
        """Fvib [eV]"""
        h, f, kb, dos = self.h, self.f, self.kb, self.dos
        T = self._get_temp(temp)
        arg = h * f / (kb*T[:,None])
        y = dos * np.log(2.0*np.sinh(arg/2.0))
        ret = self._integrate(y,f) * T
        return ret * (kb / eV)
    
    def vibrational_entropy(self, temp=None):
        """Svib [R, kb]"""
        h, f, kb, dos = self.h, self.f, self.kb, self.dos
        T = self._get_temp(temp)
        arg = h * f / (kb*T[:,None])
        y = dos * (0.5/T[:,None] * (h / kb) * f * coth(arg/2.0) - 
                   np.log(2.0*np.sinh(arg/2.0)))
        ret = self._integrate(y,f)
        return ret
    
    # aliases 
    def evib(self, *args, **kwargs):
        """Same as vibrational_internal_energy()."""
        return self.vibrational_internal_energy(*args, **kwargs)

    def cv(self, *args, **kwargs):
        """Same as isochoric_heat_capacity()."""
        return self.isochoric_heat_capacity(*args, **kwargs)
    
    def fvib(self, *args, **kwargs):
        """Same as vibrational_free_energy()."""
        return self.vibrational_free_energy(*args, **kwargs)
    
    def svib(self, *args, **kwargs):
        """Same as vibrational_entropy()."""
        return self.vibrational_entropy(*args, **kwargs)


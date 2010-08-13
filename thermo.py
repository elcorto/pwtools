import numpy as np
from scipy.integrate import simps, trapz
from pwtools.constants import kb, h, R, pi, c0

def cv(freq, dos, temp, hplanck=h, kb=kb, fixzero=True):
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
        try to handle zero frequencies which can otherwise lead to 1/sinh(0) =
        NaN
    
    returns:
    --------
    cv : 1d array
        The isochoric heat capacity from temp[0] to temp[-1] in units of R
        (universal gas constant: 8.314 J/(K*mol)).
    
    notes:
    ------
    For high T, cv should approach 3*N where N = natom = atoms in the unit
    cell. This is the Dulong-Petit limit (usually 3*N*R, here 3*N).
    """
    # Theory:
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
    # in F_QHS.f90:
    #   a3 = 1.0/8065.5/8.617e-5 # = hbar*c0*100*2*pi / kb
    #
    # Must convert freq to w=2*pi*freq for hbar*w. Or, use 
    # h*freq.
    #
    # cm^-1 -> 1/s       : * c0*100
    # 1/s   -> cm^-1     : / (c0*100)
    # s     -> cm        : * c0*100
    # => hbar or h: J*s -> J*cm : *c0*100
    
    _f = freq           # cm^-1
    _h = hplanck*c0*100 # J*s -> J*cm
    _kb = kb            # J/K
    T = temp            # K
    if fixzero:
        eps = np.finfo(float).eps
        _f = _f.copy()
        _f[_f < 1.5*eps] = 1.5*eps
    arg = _h * _f / (_kb*T[:,None])
    y = dos * _f**2 / np.sinh(arg / 2.0)**2.0
    fac = (_h / (_kb*2*T))**2
    cv = np.array([simps(y[i,:], _f) for i in range(y.shape[0])]) * fac
    return cv


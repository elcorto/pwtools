import os

import numpy as np
from scipy.integrate import simps, trapz

from pwtools.constants import kb, h, R, pi, c0
from pwtools import common, constants

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


class ExternEOS(object):
    """Base class for calling extern (Fortran) EOS-fitting apps. Write input
    file, call app, load E-V fitted data. Load or calcutate P-V data.
    >>> eos = SomeEOSClass(app='super_app.x', energy=e, volume=v)
    >>> eos.fit()
    >>> ee,vv_e = eos.get_ev()
    >>> pp,vv_p = eos.get_pv()
    >>> plot(e,v, 'o-', label='data')
    >>> plot(vv_e, ee, label='eos fit')
    >>> plot(vv_p, pp, label='-dE_eos/dV')
    """
    # Note: we distinguish between the volume "x-axis" for energy (vv_e) and
    # pressure (vv_p) b/c, depending on how P=-dE_eos/dV is calculated, these
    # may have different length.
    def __init__(self, app=None, energy=None, volume=None, dir=None):
        # str, name of the executable, make sure that it is on your PATH or
        # use an absolute path
        self.app = app
        # 1d, Ry
        self.energy = energy
        # 1d, Bohr^3
        self.volume = volume
        if dir is None:
            self.dir = os.path.basename(self.app)
        else:
            self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir) 
        print("Find your results in %s/" %self.dir)
        self.infn = os.path.join(self.dir, 'eos.in')

    def fit(self):
        # 2d array [volume, evergy]
        self.fitdata_energy = None
        # 2d array [volume, pressure]
        self.fitdata_pressure = None

    def get_ev(self):
        """Must return 2 1d arrays: energy[Ry], volume[Bohr^3]"""
        return (self.fitdata_energy[:,1], 
                self.fitdata_energy[:,0])
    
    def get_pv(self):
        """Must return 2 1d arrays: pressure[GPa], volume[Bohr^3]"""
        return (self.fitdata_pressure[:,1], 
                self.fitdata_pressure[:,0])


class BossEOSfit(ExternEOS):
    """eosfit.x from WIEN2k modified by The Boss"""
    def __init__(self, app='eosfit.x', **kwargs):
        ExternEOS.__init__(self, app=app, **kwargs)

    def fit(self):
        # data file with volume[Bohr^3] etot[Ry]
        datafn = os.path.join(self.dir,'evdata.txt')
        np.savetxt(datafn, np.array([self.volume, self.energy]).T)
        min_en_idx = np.argmin(self.energy)
        # input file for eosfit.x:
        #   <datafn>
        #   <volume at min(etot)>
        common.file_write(self.infn,
                          '%s\n%f' %(os.path.basename(datafn), \
                                     self.volume[min_en_idx]))
        common.system('cd %s && eosfit.x < %s' \
                      %(self.dir, os.path.basename(self.infn)))
        # V, E                      
        # data produced by fit, 'eos.fit' hardcoded in eosfit.f90
        self.fitdata_energy = np.loadtxt(os.path.join(self.dir,'eos.fit'))
        # V, P
        # P = -dE/dV
        # Ry / Bohr^3 -> Pa -> GPa
        fac = constants.Ry_to_J / constants.a0**3 / 1e9
        vol, dEdV = common.deriv_fd(self.fitdata_energy[:,1], 
                                    self.fitdata_energy[:,0], 
                                    n=1)
        self.fitdata_pressure = np.empty((vol.shape[0], 2), dtype=float)
        self.fitdata_pressure[:, 1] = -dEdV*fac
        self.fitdata_pressure[:, 0] = vol


class ElkEOSfit(ExternEOS):
    """eos.x from the Elk [1] and Exciting [2] codes.
    [1] http://elk.sourceforge.net/
    [2] http://exciting-code.org/
    """
    def __init__(self, app='eos.x', natom=None, name='foo', etype=1, **kwargs):
        ExternEOS.__init__(self, app=app, **kwargs)
        self.name = name
        self.natom = natom
        self.etype = etype
        # From the README:
        # ----------------
        # input file:
        #
        # cname               : name of crystal up to 256 characters
        # natoms              : number of atoms in unit cell
        # etype               : equation of state type (see below)
        # vplt1, vplt2, nvplt : volume interval over which to plot energy, pressure etc.
        #                       as well as the number of points in the plot
        # nevpt               : number of energy-volume points to be inputted
        # vpt(i) ept(i)       : energy-volume points (atomic units)
        #
        # Note that the input units are atomic - Bohr and Hartree (NOT Rydbergs).
        #
        # output files:
        #
        # All units are atomic unless otherwise stated
        # EOS parameters written to PARAM.OUT
        # Energy-volume per atom at data points written to EVPAP.OUT
        # Energy-volume per atom over interval written to EVPAI.OUT
        # Pressure(GPa)-volume per atom at data points written to PVPAP.OUT
        # Pressure(GPa)-volume per atom over interval written to PVPAI.OUT
        # Enthalpy-pressure(GPa) per atom over interval written to HPPAI.OUT
        #
        # Note that the data is normalized to natoms and that energy is in
        # Hartree.

    def fit(self):
        # volume[Bohr^3] etot[Ha] (Ry -> Ha : /2)
        data = np.array([self.volume, self.energy/2.0]).T
        npoints = len(self.volume)*15
        infn_txt =\
        """
%s
%i
%i
%f,  %f,  %i
%i
%s
        """%(self.name, 
             self.natom, 
             self.etype, 
             self.volume[0], self.volume[-1], npoints,
             len(self.volume), 
             common.str_arr(data))
        common.file_write(self.infn, infn_txt)
        common.system('cd %s && eos.x' %self.dir)
        print(open(os.path.join(self.dir,'PARAM.OUT')).read())
        # remove normalization on natoms
        # convert energy back to Ry
        fitev = np.loadtxt(os.path.join(self.dir,'EVPAI.OUT')) * self.natom
        fitev[:,1] *= 2.0
        self.fitdata_energy = fitev
        fitpv = np.loadtxt(os.path.join(self.dir,'PVPAI.OUT'))
        fitpv[:,0] *= self.natom
        self.fitdata_pressure = fitpv
        


# eos.py
#
# Interface classes for calling EOS fitting apps.
#
# Just compile the app to produce an executable, e.g. eos.x. Then use 
# app='/path/to/eos.x' in the constructor.

import os
import subprocess
import numpy as np
from pwtools import common, constants, num
from pwtools.constants import Ry, Ha, Bohr, Ang, eV
from pwtools.base import FlexibleGetters

class ExternEOS(FlexibleGetters):
    """Base class for calling extern (Fortran) EOS-fitting apps. The class
    writes an input file, calls the app, loads E(V) fitted data and loads or
    calcutates P(V), B(V).

    The number N of data points for the returned arrays (fitted curves) are
    handled by derived classes.

    We have three "representations" of the data:

    (a) input data E(V) : self.volume [Ang^3], self.energy [eV]
    (b) fitted or calculated points : self.{ev,pv,bv} -- 2d arrays (N,2)
        where N is the number of returned fitted points from the fitting app. N
        depends in the fitting app. For instance, in ElkEOSFit, you can use
        `npoints` to set N.
    (c) Splines thru fitted or calculated (N,2) data ev,pv,bv :
        self.spl_{ev,pv,bv}.        
    
    Attributes
    ----------
    ev, pv, bv, spl_ev, spl_pv, spl_bv, see fit() doc string.

    Examples
    --------
    >>> eos = SomeEOSClass(app='super_fitting_app.x', energy=ee, volume=vv)
    >>> eos.fit()
    >>> plot(vv, ee, 'o-', label='E(V) data')
    >>> plot(eos.ev[:,0], eos.ev[:,1], label='E(V) fit')
    >>> plot(eos.pv[:,0], eos.pv[:,1], label='P=-dE/dV')
    >>> plot(eos.ev[:,0], eos.spl_ev(eos.ev[:,0]), label='spline E(V)')
    >>> plot(eos.pv[:,0], eos.spl_pv(eos.pv[:,0]), label='spline P(V)')
    >>> print "min:", eos.get_min()

    For derived classes:
    Implement _fit(), which sets self.{ev,pv}.
    """
    # Notes
    # -----
    # We distinguish between the volume axis for energy ev[:,0], pressure
    # pv[:,0] and bulk modulus bv[:,0] b/c, depending on how P=-dE/dV or B(V)
    # is calculated, these may have different length. For instance, if the
    # pressure is calculated by finite differences, then ev.shape[0] == N,
    # pv.shape[0] == N-1 . This is mostly for historic reasons but also b/c
    # it's nice to have an array with x-y data right there.
    def __init__(self, app=None, energy=None, volume=None, dir=None,
                 bv_method='ev', verbose=True):
        """
        Parameters
        ----------
        app : str 
            name of the executable ([/path/to/]eos.x), make sure that it is on
            your PATH or use an absolute path
        energy : 1d array [eV]
        volume : 1d array [Ang^3]
        dir : str
            dir where in- and outfiles are written, default is the basename of
            "app" (e.g. "eos.x" for app='/path/to/eos.x')
        bv_method : str, {'pv', 'ev'}
            Based on which quantity should B(V) and minimum properties be
            calculated.
            pv: based on P(V) 
            ev: based on E(V) 
        verbose : bool
            print stdout and stderr of fitting tool
        """
        assert len(energy) == len(volume), ("volume and energy arrays have "
                                            "not the same length")
        assert (np.diff(volume) > 0.0).any(), ("volume seems to be wrongly "
            "sorted")
        self.app = app
        self.energy = energy
        self.volume = volume
        self.app_basename = os.path.basename(self.app)
        self.set_bv_method(bv_method)
        self.verbose = verbose
        if dir is None:
            self.dir = self.app_basename
        else:
            self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if self.verbose:            
            print("Find your results in %s/" %self.dir)
        self.infn = os.path.join(self.dir, 'eos.in')
    
    def _fit(self):
        """Fit E-V data (self.energy, self.volume) and set self.ev, self.pv .

        This is the interface which derived classes must implement. If pv is
        not calculated by the fitting tool, use smth like num.deriv_spl() to
        calculate the pressure P=-dE/dV.            
        """
        self.ev = None
        self.pv = None
    
    # user-callable method
    def fit(self, *args, **kwargs):
        """Fit E-V data (self.energy, self.volume).

        After calling fit(), these attrs are available:
            | self.ev : 2d array (N,2) [volume [Ang^3], energy        E(V) [eV] ]
            | self.pv : 2d array (N,2) [volume [Ang^3], pressure      P(V) [GPa]]
            | self.bv : 2d array (N,2) [volume [Ang^3], bulk modulus  B(V) [GPa]]
            | self.spl_ev : Spline thru E(V)
            | self.spl_pv : Spline thru P(V)
            | self.spl_bv : Spline thru B(V)
        
        """            
        self._fit(*args, **kwargs)
        self.bv = self.calc_bv()
        self.try_set_attr('spl_ev')
        self.try_set_attr('spl_pv')
        self.try_set_attr('spl_bv')

    def _get_spl(self, attr_name):
        # attr_name : 'ev', 'pv', 'bv'
        # spl_attr_name : 'self.spl_{ev,pv,bv}'
        spl_attr_name = "spl_%s" %attr_name
        if self.is_set_attr(spl_attr_name):
             return getattr(self, spl_attr_name)
        else:
            arr = getattr(self, attr_name)
            return num.Spline(arr[:,0], arr[:,1])
    
    def get_spl_ev(self):
        return self._get_spl('ev')
    
    def get_spl_pv(self):
        return self._get_spl('pv')
    
    def get_spl_bv(self):
        return self._get_spl('bv')

    def set_bv_method(self, bv_method):
        """Set self.bv_method, a.k.a. switch to another bv_method.
        
        Parameters
        ----------
        bv_method : str
            'ev', 'pv'
        """
        self.bv_method = bv_method

    def calc_bv(self):
        # B = -V*dP/dV
        if self.bv_method == 'pv':
            self.try_set_attr('spl_pv')
            vv = self.pv[:,0]
            return np.array([vv, -vv * self.spl_pv(vv, der=1)]).T
        # B = V*d^2E/dV^2 
        elif self.bv_method == 'ev':
            self.try_set_attr('spl_ev')
            # eV / Ang^3 -> GPa
            fac = eV * 1e21 # 160.2176487
            vv = self.ev[:,0]
            return np.array([vv, vv * self.spl_ev(vv, der=2) * fac]).T
        else:
            raise StandardError("unknown bv_method: '%s'" %bv_method)
    
    def get_min(self, behave='new'):
        """
        Calculate properites at energy minimum of E(V).

        Parameters
        ----------
        behave : str, optional, {'new', 'old'}
        
        Returns
        -------
        behave = 'new' : return a dict {v0, e0, p0, b0}
            volume, energy, pressure, bulk modulus at energy min
        behave = 'old' : array of length 4 [v0, e0, b0, p0]

        Notes
        -----
        If the pressure at the E(V) minimum is not very close to zero (say ~
        1e-10), then your E-V data is incorrect. Usually, this is because of
        poorly converged calculations (low ecut, too few k-points).
        """
        self.try_set_attr('spl_pv')
        self.try_set_attr('spl_ev')
        self.try_set_attr('spl_bv')
        if self.bv_method == 'pv':
            v0 = self.spl_pv.get_root()
        elif self.bv_method == 'ev':
            v0 = self.spl_ev.get_min()
        else:
            raise StandardError("unknown bv_method: '%s'" %bv_method)
        p0 = self.spl_pv(v0)
        e0 = self.spl_ev(v0)
        b0 = self.spl_bv(v0)
        if behave == 'old':
            return np.array([v0, e0, p0, b0])
        elif behave == 'new':            
            dct = {}
            dct['v0'] = v0
            dct['e0'] = e0
            dct['p0'] = p0
            dct['b0'] = b0
            return dct
        else:
            raise StandardError("unknown value for `behave`: %s" %str(behave))
    
    def _call(self, cmd):
        """
        Call shell command 'cmd' and merge stdout and stderr.

        Use this instead of common.backtick() if fitting tool insists on beeing
        very chatty on stderr.
        """
        pp = subprocess.Popen(cmd, 
                              shell=True,
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.STDOUT)
        out,err = pp.communicate()
        assert err is None, "stderr output is not None"                              
        return out


class BossEOSFit(ExternEOS):
    """eosfit.x from WIEN2k modified by The Boss
    
    self.{ev,bv,pv} all have the same shape[0] b/c we do not use finite
    differences for derivatives.

    Notes
    -----
    - The output file name with fitted data is hardcoded in eosfit.f to be
      'eos.fit'.
    - Also, the number of fit points is hardcoded. This is not the case for
      eos.x, see npoints in ElkEOSFit.      
    """
    def __init__(self, app='eosfit.x', **kwargs):
        raise NotImplementedError("fix units")
        ExternEOS.__init__(self, app=app, **kwargs)

    def _fit(self):
        # data file with volume[Bohr^3] etot[Ry]
        datafn = os.path.join(self.dir, 'evdata.txt')
        np.savetxt(datafn, np.array([self.volume, self.energy]).T)
        min_en_idx = np.argmin(self.energy)
        # input file for eosfit.x:
        #   <datafn>
        #   <volume at min(etot)>
        common.file_write(self.infn,
                          '%s\n%f' %(os.path.basename(datafn), \
                                     self.volume[min_en_idx]))
        out = common.backtick('cd %s && %s < %s' \
                              %(self.dir, 
                                self.app_basename, 
                                os.path.basename(self.infn)))
        if self.verbose:
            print out
        # [V [Bohr^3], E [Ry]]
        self.ev = np.loadtxt(os.path.join(self.dir,'eos.fit'))
        # [V [Bohr^3], P [GPa]]
        # P = -dE/dV
        # Ry / Bohr^3 -> Pa -> GPa
        fac = constants.Ry_to_J / constants.a0**3 / 1e9
        vol, dEdV = num.deriv_spl(self.ev[:,1], 
                                  self.ev[:,0], 
                                  n=1)
        self.pv = np.empty((vol.shape[0], 2), dtype=float)
        self.pv[:, 0] = vol
        self.pv[:, 1] = -dEdV*fac


class ElkEOSFit(ExternEOS):
    """eos.x from the Elk [1] and Exciting [2] codes.
    [1] http://elk.sourceforge.net/
    [2] http://exciting-code.org/

    Note that the data produced by eos.x is divided by natoms and that energy
    is in Hartree. We remove the normalization and convert Ha -> eV.
    
    self.{ev,bv,pv} all have the same shape[0] b/c we do not use finite
    differences for derivatives.
    """
    def __init__(self, app='eos.x', natoms=1, name='foo', etype=1, 
                 npoints=300, **kwargs):
        """
        Parameters
        ----------
        see ExternEOS.__init__()

        natoms : number of atoms in the unit cell, this is (I think) only used
            for normalization and can be set to 1 if not needed
        name : str
            some dummy name for the input file
        etype : int
            type of EOS to fit (see below)
        npoints : integer, optional
            number of E-V and P-V points of the fitted curves (`nvplt` in
            eos.x)

        Notes
        -----
        From the README:
        The equations of state currently implemented are:
         1. Universal EOS (Vinet P et al., J. Phys.: Condens. Matter 1, p1941 (1989))
         2. Murnaghan EOS (Murnaghan F D, Am. J. Math. 49, p235 (1937))
         3. Birch-Murnaghan 3rd-order EOS (Birch F, Phys. Rev. 71, p809 (1947))
         4. Birch-Murnaghan 4th-order EOS
         5. Natural strain 3rd-order EOS (Poirier J-P and Tarantola A, Phys. Earth
            Planet Int. 109, p1 (1998))
         6. Natural strain 4th-order EOS
         7. Cubic polynomial in (V-V0)
        """
        ExternEOS.__init__(self, app=app, **kwargs)
        self.name = name
        self.natoms = natoms
        self.etype = etype
        self.npoints = npoints
        assert self.npoints >= len(self.volume), ("npoints is < number of E-V "
            "input points")
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

    def _fit(self):
        # volume[Bohr^3] etot[Ha] for eos.x
        volume = self.volume*(Ang**3.0 / Bohr**3.0)
        energy = self.energy*(eV / Ha)
        data = np.array([volume, energy]).T
        infn_txt =\
        """
%s
%i
%i
%f,  %f,  %i
%i
%s
        """%(self.name, 
             self.natoms, 
             self.etype, 
             volume[0], volume[-1], self.npoints,
             len(volume), 
             common.str_arr(data))
        common.file_write(self.infn, infn_txt)
        out = common.backtick('cd %s && %s' %(self.dir, self.app_basename))
        if self.verbose:
            print out
            print(open(os.path.join(self.dir,'PARAM.OUT')).read())
        # Remove normalization on natoms. See .../eos/output.f90:
        # fitev: [volume [Bohr^3] / natoms, energy [Ha] / natoms]
        # fitpv: [volume [Bohr^3] / natoms, pressure [GPa]]
        fitev = np.loadtxt(os.path.join(self.dir,'EVPAI.OUT')) * self.natoms
        # convert energy back to [Ang^3, eV]
        fitev[:,0] *= (Bohr**3 / Ang**3)
        fitev[:,1] *= (Ha / eV)
        self.ev = fitev
        fitpv = np.loadtxt(os.path.join(self.dir,'PVPAI.OUT'))
        fitpv[:,0] *= (self.natoms * Bohr**3 / Ang**3)
        self.pv = fitpv

# backward compat
BossEOSfit = BossEOSFit
ElkEOSfit = ElkEOSFit

# eos.py
#
# Interface classes for calling EOS fitting apps.
#
# Just compile the app to produce an executable, e.g. eos.x. Then use 
# app='/path/to/eos.x' in the constructor.

import os
import subprocess
import numpy as np
from pwtools import common

class ExternEOS(object):
    """Base class for calling extern (Fortran) EOS-fitting apps. The class
    writes an input file, calls the app, loads E-V fitted data and loads or
    calcutates P-V data.
    
    The number of data points for the returned arrays (fitted curves) are
    handled by derived classes.

    methods:
    --------
    fit() : fit E-V data
    get_ev() : return tuple (ev_v, ev_e), see below
    get_pv() : return tuple (pv_v, pv_p), see below
    
    attributes:
    -----------
    After calling fit(), these attrs are available. They are also returned by
    get_{ev,pv}.
        self.ev_v : volume [Bohr^3]
        self.ev_e : evergy [Ry]
        self.pv_v : volume [Bohr^3]
        self.pv_p : pressure [GPa]

    >>> eos = SomeEOSClass(app='super_fitting_app.x', energy=e, volume=v)
    >>> eos.fit()
    >>> plot(v,e, 'o-', label='data')
    >>> plot(eos.ev_v, eos.ev_e, label='eos fit')
    >>> plot(eos.pv_v, eos.pv_p, label='-dE_eos/dV')

    notes:
    ------
    We distinguish between the volume "x-axis" for energy (ev_v) and pressure
    (pv_v) b/c, depending on how P=-dE_eos/dV is calculated, these may have
    different length (if the pressure is calculated by finite differences, for
    instance).
    """
    # Derived classes:
    # ----------------
    # Implement _fit(), which sets self.fitdata_{energy, pressure}.
    #
    # self.fitdata_{energy, pressure} : 2d array, shape (len(volume), 2):
    #       [volume[Bohr^3], {energy[Ry], pressure[GPa]}]
    #     Derived classed must conform to this in their _fit() method. We use
    #     the fitdata_* arrays b/c the fitting apps usually write their results
    #     in that format -- just np.loadtxt() that.
    def __init__(self, app=None, energy=None, volume=None, dir=None,
                 verbose=True):
        """
        args:
        -----
        app : str 
            name of the executable ([/path/to/]eos.x), make sure that it is on
            your PATH or use an absolute path
        energy : 1d array [Ry]
        volume : 1d array [Bohr^3]
        dir : str
            dir where in- and outfiles are written, default is the basename of
            "app" (e.g. "eos.x" for app='/path/to/eos.x')
        verbose : bool
            print stdout and stderr of fitting tool
        """
        assert len(energy) == len(volume), ("volume and energy arrays have "
                                            "not the same length")
        self.app = app
        self.energy = energy
        self.volume = volume
        self.app_basename = os.path.basename(self.app)
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
        """Fit E-V data (self.energy, self.volume) and set 
            self.fitdata_energy
            self.fitdata_pressure
        This is the interface which derived classes must implement.            
        """
        self.fitdata_energy = None
        self.fitdata_pressure = None

    def fit(self, *args, **kwargs):
        """Fit E-V data (self.energy, self.volume) and set 
            self.fitdata_energy
            self.fitdata_pressure
        by fitting. Set shortcut attrs
            self.ev_v
            self.ev_e
            self.pv_v
            self.pv_p
        """            
        # Assume 
        #   fitdata_*[:,0] = volume
        #   fitdata_*[:,1] = <data>
        # shortcuts
        self._fit(*args, **kwargs)   
        self.ev_v = self.fitdata_energy[:,0]
        self.ev_e = self.fitdata_energy[:,1]
        self.pv_v = self.fitdata_pressure[:,0]
        self.pv_p = self.fitdata_pressure[:,1]

    def get_ev(self):
        """
        returns:
        --------
        v,e
        v : volume [Bohr^3] 
        e : evergy [Ry]
        """
        return (self.ev_v, self.ev_e)
    
    def get_pv(self):
        """
        returns:
        --------
        v,p
        v : volume [Bohr^3] 
        p : pressure [GPa]
        """
        return (self.pv_v, self.pv_p)
    
    def call(self, cmd):
        """
        Call shell command 'cmd' and merge stdout and stderr.

        Use this instead of common.backtick() if fitting tool insists on beeing
        very chatty on stderr.
        """
        pp = subprocess.Popen(call, 
                              shell=True,
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.STDOUT)
        out,err = pp.communicate()
        assert err is None, "stderr output is not None"                              
        return out


class BossEOSFit(ExternEOS):
    """eosfit.x from WIEN2k modified by The Boss

    The returned arrays from get_pv() are 1 shorter than the results of
    get_ev() b/c the pressure is calculated by finite differences.

    notes:
    ------
    - The output file name with fitted data is hardcoded in eosfit.f to be
      'eos.fit'.
    - Also, the number of fit points is hardcoded. This is not the case for
      eos.x, see npoints in ElkEOSFit.      
    """
    def __init__(self, app='eosfit.x', **kwargs):
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
        self.fitdata_energy = np.loadtxt(os.path.join(self.dir,'eos.fit'))
        # [V [Bohr^3], P [GPa]]
        # P = -dE/dV
        # Ry / Bohr^3 -> Pa -> GPa
        fac = constants.Ry_to_J / constants.a0**3 / 1e9
        vol, dEdV = common.deriv_fd(self.fitdata_energy[:,1], 
                                    self.fitdata_energy[:,0], 
                                    n=1)
        self.fitdata_pressure = np.empty((vol.shape[0], 2), dtype=float)
        self.fitdata_pressure[:, 1] = -dEdV*fac
        self.fitdata_pressure[:, 0] = vol


class ElkEOSFit(ExternEOS):
    """eos.x from the Elk [1] and Exciting [2] codes.
    [1] http://elk.sourceforge.net/
    [2] http://exciting-code.org/

    Note that the data produced by eos.x is divided by natoms and that energy
    is in Hartree. We remove the normalization and convert Ha -> Ry.

    get_ev() and get_pv() return arrays of the same length.
    """
    def __init__(self, app='eos.x', natoms=1, name='foo', etype=1, 
                 npoints=300, **kwargs):
        """
        args:
        -----
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

        notes:
        ------
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
        # volume[Bohr^3] etot[Ha] (Ry -> Ha : / 2)
        data = np.array([self.volume, self.energy/2.0]).T
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
             self.volume[0], self.volume[-1], self.npoints,
             len(self.volume), 
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
        # convert energy back to Ry
        fitev[:,1] *= 2.0
        self.fitdata_energy = fitev
        fitpv = np.loadtxt(os.path.join(self.dir,'PVPAI.OUT'))
        fitpv[:,0] *= self.natoms
        self.fitdata_pressure = fitpv

# backward compat
BossEOSfit = BossEOSFit
ElkEOSfit = ElkEOSFit

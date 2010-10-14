# eos.py
#
# Interface classes for calling EOS fitting apps.
#
# Just compile the app to produce an executable, e.g. eos.x. Then use 
# app='/path/to/eos.x' in the constructor.

import os
import numpy as np
from pwtools import common

class ExternEOS(object):
    """Base class for calling extern (Fortran) EOS-fitting apps. The class
    writes an input file, calls the app, loads E-V fitted data and loads or
    calcutates P-V data.

    methods:
    --------
    fit() : fit E-V data
    get_ev() : return 2d array (len(volume), :) [volume [Bohr^3], evergy [Ry]]
    get_pv() : return 2d array (len(volume), :) [volume [Bohr^3], pressure [GPa]]

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
    # may have different length (if the pressure is calculated by finite
    # differences, for instance).
    #
    # self.fitdata_{energy, pressure} : We have hardcoded the way these arrays
    # must look like. See get_{ev,pv}. Derived classed must conform to this  in
    # their fit() method.
    def __init__(self, app=None, energy=None, volume=None, dir=None):
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
        """
        self.app = app
        self.energy = energy
        self.volume = volume
        self.app_basename = os.path.basename(self.app)
        if dir is None:
            self.dir = self.app_basename
        else:
            self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir) 
        print("Find your results in %s/" %self.dir)
        self.infn = os.path.join(self.dir, 'eos.in')

    def fit(self):
        """Fit E-V data and set 
            self.fitdata_energy
            self.fitdata_pressure
        """            
        self.fitdata_energy = None
        self.fitdata_pressure = None

    def get_ev(self):
        """
        returns:
        --------
        fitdata_energy : 2d array (len(volume), :) 
            [volume [Bohr^3], evergy [Ry]]
        """
        # Assume 
        #   fitdata_*[:,0] = <data>
        #   fitdata_*[:,1] = volume
        return (self.fitdata_energy[:,1], 
                self.fitdata_energy[:,0])
    
    def get_pv(self):
        """
        returns:
        --------
        fitdata_pressure : 2d array (len(volume), :) 
            [volume [Bohr^3], pressure [GPa]]
        """
        # Assume 
        #   fitdata_*[:,0] = <data>
        #   fitdata_*[:,1] = volume
        return (self.fitdata_pressure[:,1], 
                self.fitdata_pressure[:,0])


class BossEOSfit(ExternEOS):
    """eosfit.x from WIEN2k modified by The Boss
    """
    def __init__(self, app='eosfit.x', **kwargs):
        ExternEOS.__init__(self, app=app, **kwargs)

    def fit(self):
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
        common.system('cd %s && %s < %s' \
                      %(self.dir, self.app_basename, 
                        os.path.basename(self.infn)))
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

    Note that the data produced by eos.x is normalized to natoms and that
    energy is in Hartree. We remove the normalization and convert Ha -> Ry.
    """
    def __init__(self, app='eos.x', natoms=1, name='foo', etype=1, **kwargs):
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

    def fit(self):
        # volume[Bohr^3] etot[Ha] (Ry -> Ha : / 2)
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
             self.natoms, 
             self.etype, 
             self.volume[0], self.volume[-1], npoints,
             len(self.volume), 
             common.str_arr(data))
        common.file_write(self.infn, infn_txt)
        common.system('cd %s && %s' %(self.dir, self.app_basename))
        print(open(os.path.join(self.dir,'PARAM.OUT')).read())
        # remove normalization on natoms
        # convert energy back to Ry
        fitev = np.loadtxt(os.path.join(self.dir,'EVPAI.OUT')) * self.natoms
        fitev[:,1] *= 2.0
        self.fitdata_energy = fitev
        fitpv = np.loadtxt(os.path.join(self.dir,'PVPAI.OUT'))
        fitpv[:,0] *= self.natoms
        self.fitdata_pressure = fitpv

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
from pwtools.base import FlexibleGetters

class ExternEOS(FlexibleGetters):
    """Base class for calling extern (Fortran) EOS-fitting apps. The class
    writes an input file, calls the app, loads E(V) fitted data and loads or
    calcutates P(V), B(V).

    The number N of data points for the returned arrays (fitted curves) are
    handled by derived classes.

    We have three "representations" of the data:
    (a) input data E(V) : self.volume, self.energy
    (b) fitted or calculated points : self.{ev,pv,bv} -- 2d arrays (N,2)
        where N is the number of returned fitted points from the fitting app. N
        depends in the fitting app. For instance, in ElkEOSFit, you can use
        `npoints` to set N.
    (c) Splines thru fitted or calculated (N,2) data ev,pv,bv :
        self.spl_{ev,pv,bv}.        

    methods:
    --------
    fit() : call this to fit data and calculate pressure etc
    get_min() : return properties at min(energy), hint: If the
        pressure is not very close to zero (say ~ 1e-10), then your E-V data is
        incorrect. Usually, this is because of poorly converged
        calculations (low ecut, too few k-points).
    get_spl_{ev,pv,bv}() : Return a Spline object for the fitted data.

    attributes:
    -----------
    After calling fit(), these attrs are available. All are 2d arrays (N,2),
    see notes below regarding N.
        self.ev : [volume [Bohr^3], energy E(V) [Ry]]
        self.pv : [volume [Bohr^3], pressure P(V) [GPa]]
        self.bv : [volume [Bohr^3], bulk modulus B(V) [GPa]]
    Splines self.spl_* should be obtained by the self.get_spl_* methods, rather
    then attr access self.spl_*. This is because they are calculated only when
    requested, not by default in fit().

    >>> eos = SomeEOSClass(app='super_fitting_app.x', energy=ee, volume=vv)
    >>> eos.fit()
    >>> plot(vv, ee, 'o-', label='data')
    >>> plot(eos.ev[:,0], eos.ev[:,1], label='eos fit')
    >>> plot(eos.pv[:,0], eos.pv[:,1], label='P=-dE/dV')
    >>> plot(eos.ev[:,0], eos.get_spl_ev()(eos.ev[:,0]), label='spline E(V)')
    >>> plot(eos.pv[:,0], eos.get_spl_pv()(eos.pv[:,0]), label='spline P(V)')
    >>> print "min:", eos.get_min()

    notes:
    ------
    We distinguish between the volume axis for energy (ev[:,0]) and pressure
    (pv[:,0]) b/c, depending on how P=-dE/dV is calculated, these may have
    different length. For instance, if the pressure is calculated by finite
    differences, then ev.shape[0] == N, pv.shape[0] == N-1 .
    """
    # Derived classes:
    # ----------------
    # Implement _fit(), which sets self.{ev,pv}.
    #
    # ev : 2d array, shape (number_of_fit_points, 2): [volume [Bohr^3], energy [Ry]]
    # pv : 2d array, shape (number_of_fit_points, 2): [volume [Bohr^3], pressure [GPa]]
    # 
    # Derived classed must conform to this in their _fit() method.
    def __init__(self, app=None, energy=None, volume=None, dir=None,
                 method='ev', verbose=True):
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
            "app" (e.g. "/path/to" for app='/path/to/eos.x')
        method : str, {'pv', 'ev'}
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
        self.method = method
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
            self.ev
            self.pv
        This is the interface which derived classes must implement. If bv is
        not calculated by the fitting tool, use smth like num.deriv_spl() to
        calculate the pressure P=-dE/dV.            
        """
        self.ev = None
        self.pv = None
    
    def fit(self, *args, **kwargs):
        """Fit E-V data (self.energy, self.volume) and set 
            self.ev
            self.pv
        (done by self._fit()) and calculate self.bv .
        """            
        self._fit(*args, **kwargs)
        self.bv = self.calc_bv()

    # Make internal spline representations accessible to the user.
    # XXX how about using @property for another way to automatically invoke the
    # getter?
    def get_spl_ev(self):
        if self.is_set_attr('spl_ev'):
            return self.spl_ev
        else:            
            return num.Spline(*self.get_ev_tup())
    
    def get_spl_pv(self):
        if self.is_set_attr('spl_pv'):
            return self.spl_pv
        else:            
            return num.Spline(*self.get_pv_tup())
    
    def get_spl_bv(self):
        if self.is_set_attr('spl_bv'):
            return self.spl_bv
        else:            
            return num.Spline(*self.get_bv_tup())
    
    def set_method(self, method):
        """Set self.method, a.k.a. switch to another method.
        
        args:
        -----
        method : str
            'ev', 'pv'
        """
        self.method = method

    def get_ev_tup(self):
        """
        returns:
        --------
        v,e
        v : volume [Bohr^3] 
        e : energy [Ry]
        """
        return (self.ev[:,0], self.ev[:,1])
    
    def get_pv_tup(self):
        """
        returns:
        --------
        v,p
        v : volume [Bohr^3] 
        p : pressure [GPa]
        """
        return (self.pv[:,0], self.pv[:,1])
    
    def get_bv_tup(self):
        """
        returns:
        --------
        v,b
        v : volume [Bohr^3] 
        b : bulk modulus [GPa]
        """
        return (self.bv[:,0], self.bv[:,1])
    
    # XXX backward compat
    def get_ev(self):
        return self.get_ev_tup()
    
    def get_pv(self):
        return self.get_pv_tup()
    
    def get_bv(self):
        return self.get_bv_tup()
    # XXX backward compat

    def calc_bv(self):
        # B = V*d^2E/dV^2 = -V*dP/dV
        if self.method == 'pv':
            self.try_set_attr('spl_pv')
            vv = self.pv[:,0]
            return np.array([vv, -vv * self.spl_pv(vv, der=1)]).T
        elif self.method == 'ev':
            self.try_set_attr('spl_ev')
            # Ry / Bohr^3 -> GPa
            fac = constants.Ry_to_J / constants.a0**3.0 / 1e9
            vv = self.ev[:,0]
            return np.array([vv, vv * self.spl_ev(vv, der=2) * fac]).T
        else:
            raise StandardError("unknown method: '%s'" %method)
    
    # XXX behave : backward compat
    def get_min(self, behave='new'):
        """
        returns:
        --------
        a dict {v0, e0, p0, b0} : volume, energy, pressure, bulk modulus at
            energy min
        or an array of length 4 if behave=='old'.            
        """
        self.try_set_attr('spl_pv')
        self.try_set_attr('spl_ev')
        self.try_set_attr('spl_bv')
        if self.method == 'pv':
            v0 = self.spl_pv.get_root()
        elif self.method == 'ev':
            v0 = self.spl_ev.get_min()
        else:
            raise StandardError("unknown method: '%s'" %method)
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
    
    self.{ev,bv,pv} all have the same shape[0] b/c we do not use finite
    differences for derivatives.

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
    is in Hartree. We remove the normalization and convert Ha -> Ry.
    
    self.{ev,bv,pv} all have the same shape[0] b/c we do not use finite
    differences for derivatives.
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
        self.ev = fitev
        fitpv = np.loadtxt(os.path.join(self.dir,'PVPAI.OUT'))
        fitpv[:,0] *= self.natoms
        self.pv = fitpv

# backward compat
BossEOSfit = BossEOSFit
ElkEOSfit = ElkEOSFit

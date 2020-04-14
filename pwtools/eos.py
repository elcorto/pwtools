"""
EOS fitting. Use :class:`EosFit` (only Vinet EOS for now).

Also: Old interface class :class:`ElkEOSFit` and the base class
:class:`ExternEOS` for calling extern EOS fitting applications. Compile the app
to produce an executable, e.g. ``eos.x``. Then use ``app='/path/to/eos.x'`` in
the constructor. Has another API than :class:`EosFit`, e.g. the
:meth:`EosFit.get_min` method is different from :meth:`ElkEOSFit.get_min`.
"""

import os
import subprocess, types, warnings
import numpy as np
from pwtools import common, constants, num
from pwtools.constants import Ry, Ha, Bohr, Ang, eV, eV_by_Ang3_to_GPa
from pwtools.base import FlexibleGetters
from pwtools.decorators import lazyprop
from pwtools.num import Fit1D
from scipy.optimize import leastsq


def _vinet(V, params):
    """Vinet equation from PRB 70, 224107, from pymatgen."""
    E0, B0, B1, V0 = params['e0'], params['b0'], params['b1'], params['v0']
    eta = (V/V0)**(1./3.)
    return E0 + 2.*B0*V0/(B1-1.)**2 \
           * (2. - (5. +3.*B1*(eta-1.)-3.*eta)*np.exp(-3.*(B1-1.)*(eta-1.)/2.))


def _vinet_deriv1(V, params):
    """Vinet first derivative dE/dV."""
    # Thank you, Maxima!
    E0, B0, B1, V0 = params['e0'], params['b0'], params['b1'], params['v0']
    eta = (V/V0)**(1./3.)
    ex = np.exp(1.5*(-eta*B1 + B1 + eta - 1))
    return (3*eta -3) / eta**2.0 * B0 * ex


def _vinet_deriv2(V, params):
    """Vinet second derivative d^2E/dV^2."""
    # Thank you, Maxima!
    E0, B0, B1, V0 = params['e0'], params['b0'], params['b1'], params['v0']
    eta = (V/V0)**(1./3.)
    ex = np.exp(1.5*(-eta*B1 + B1 + eta - 1))
    return -B0 * (3*eta**2.0*B1 - 3*eta*B1 - 3*eta**2.0 + 5*eta - 4) / \
            (2*eta**5 * V0) * ex


class MaxDerivException(Exception):
    def __init__(self, msg=None):
        self.msg = msg
    def __str__(self):
        return self.msg


class EVFunction(object):
    """Base class for E(V) models, such as the Vinet EOS."""
    def __init__(self):
        self.param_order = ['e0', 'b0', 'b1', 'v0']

    def evaluate(self, volume, params):
        """Evaluate function for x-axis `volume`.

        Parameters
        ----------
        volume : scalar, 1d array
            volume per atom [Ang^3]
        params : dict
            {'e0', 'b0', 'b1', 'v0'} in eV, eV/Ang^3, 1, Ang^3
        """
        pass

    def deriv(self, volume, params, der=None):
        """Calculate derivative of E(V) of order `der`. ``der=1`` is the first
        deriv etc.

        Parameters
        ----------
        volume : scalar, 1d array
            volume per atom [Ang^3]
        params : dict
            {'e0', 'b0', 'b1', 'v0'} in eV, eV/Ang^3, 1, Ang^3
        der : int
            derivative order
        """
        pass

    def get_min(self):
        pass

    def __call__(self, volume, params, der=0):
        if der == 0:
            return self.evaluate(volume, params)
        else:
            return self.deriv(volume, params, der)

    def lst2dct(self, lst):
        return dict([(k, lst[ii]) for ii,k in enumerate(self.param_order)])

    def dct2lst(self, dct):
        return [dct[k] for k in self.param_order]


class Vinet(EVFunction):
    """Vinet EOS model."""
    def evaluate(self, volume, params):
        return _vinet(volume, params)

    def deriv(self, volume, params, der=None):
        if der == 1:
            return _vinet_deriv1(volume, params)
        elif der == 2:
            return _vinet_deriv2(volume, params)
        else:
            raise MaxDerivException("der %i not supported" %der)


# Before using this a fitfunc in thermo.Gibbs, test it! You may need to
# implement data scaling, as we do in num.PolyFit.
class EosFit(Fit1D):
    """E(V) fit class.

    Examples
    --------
    >>> from pwtools.eos import EosFit
    >>> from pwtools.constants import eV_by_Ang3_to_GPa
    >>> V = linspace(30, 50, 20)
    >>> E = (V-40)**2.0 / 50.0 + 30 + rand(len(V)) / 5.0
    >>> plot(V, E, 'o')
    >>> f = EosFit(V, E)
    >>> vv = linspace(V.min(), V.max(), 200)
    >>> plot(vv, f(vv), 'r-')
    >>> v0 = f.params['v0']
    >>> print("compare fit params and values obtained from methods:")
    >>> print("V0: %f Ang^3 (%f)" %(v0, f.get_min()))
    >>> print("B0: %f GPa   (%f)" %(f.params['b0']*eV_by_Ang3_to_GPa, f.bulkmod(v0)))
    >>> print("B1: %f           " %f.params['b1'])
    >>> print("E0: %f  eV   (%f)" %(f.params['e0'], f(v0)))
    >>> print("P0: %f GPa       " %f.pressure(v0))
    """

    def __init__(self, volume, energy, func=Vinet(), splpoints=500):
        """
        Parameters
        ----------
        volume : 1d array
            volume per atom [Ang^3]
        energy : 1d array
            total energy per atom [eV]
        func : EVFunction instance
        splpoints : int
            number of spline points for fallback derivative calculation
        """
        Fit1D.__init__(self, x=volume, y=energy)
        self.func = func
        self.energy = energy
        self.volume = volume
        self.splpoints = splpoints
        self.fit()

    @lazyprop
    def spl(self):
        """Spline thru the fitted E(V) b/c we are too lazy to calculate the
        analytic derivative. Fallback."""
        # use many points for accurate deriv
        vv = np.linspace(self.volume.min(), self.volume.max(), self.splpoints)
        return num.Spline(vv, self(vv, der=0), k=5, s=None)

    def fit(self):
        """Fit E(V) model, fill ``self.params``."""
        # Quadratic fit to get an initial guess for the parameters.
        # Thanks: https://github.com/materialsproject/pymatgen
        # -> pymatgen/io/abinitio/eos.py
        a, b, c = np.polyfit(self.volume, self.energy, 2)
        v0 = -b/(2*a)
        e0 = a*v0**2 + b*v0 + c
        b0 = 2*a*v0
        b1 = 4  # b1 is usually a small number like 4
        if not self.volume.min() < v0 and v0 < self.volume.max():
            raise Exception('The minimum volume of a fitted parabola is not in the input volumes')

        # need to use lst2dct and dct2lst here to keep the order of parameters
        pp0_dct = dict(e0=e0, b0=b0, b1=b1, v0=v0)
        target = lambda pp, v: self.energy - self.func(v, self.func.lst2dct(pp))
        pp_opt, ierr = leastsq(target,
                               self.func.dct2lst(pp0_dct),
                               args=(self.volume,))
        self.params = self.func.lst2dct(pp_opt)

    def __call__(self, volume, der=0):
        """
        Parameters
        ----------
        volume : scalar, 1d array
            volume per atom [Ang^3]
        der : int
            derivative order
        """
        try:
            return self.func(volume, self.params, der=der)
        except MaxDerivException:
            return self.spl(volume, der=der)

    # Fit1D compat
    def get_min(self):
        """V0 [Ang^3]"""
        if 'v0' in self.params:
            return self.params['v0']
        else:
            return super(EosFit, self).get_min()

    def pressure(self, volume):
        """P(V) [GPa]

        Parameters
        ----------
        volume : scalar, 1d array
            volume per atom [Ang^3]
        """
        return -self(volume, der=1) * eV_by_Ang3_to_GPa

    def bulkmod(self, volume):
        """B(V) [GPa]

        Parameters
        ----------
        volume : scalar, 1d array
            volume per atom [Ang^3]
        """
        if 'b0' in self.params:
            return self.params['b0'] * eV_by_Ang3_to_GPa
        else:
            return volume * self(volume, der=2) * eV_by_Ang3_to_GPa


class ExternEOS(FlexibleGetters):
    """Base class for calling extern EOS-fitting executables. The class
    writes an input file, calls the app, loads E(V) fitted data and loads or
    calcutates P(V), B(V).

    The number N of data points for the returned arrays (fitted curves) are
    handled by derived classes.

    We have three "representations" of the data:

    (a) input data E(V) : self.volume [Ang^3], self.energy [eV]
    (b) fitted or calculated points : self.{ev,pv,bv} -- 2d arrays (N,2)
        where N is the number of returned fitted points from the fitting app. N
        depends on the fitting app. For instance, in ElkEOSFit, you can use
        `npoints` to set N.
    (c) Splines thru fitted or calculated (N,2) data ev,pv,bv :
        self.spl_{ev,pv,bv}.

    Attributes
    ----------
    ev, pv, bv, spl_ev, spl_pv, spl_bv, see fit() doc string.

    Examples
    --------
    >>> from pwtools import eos
    >>> efit = eos.ElkEOSFit(app='eos.x', energy=ee, volume=vv)
    >>> efit.fit()
    >>> plot(vv, ee, 'o-', label='E(V) data')
    >>> plot(efit.ev[:,0], efit.ev[:,1], label='E(V) fit')
    >>> plot(efit.pv[:,0], efit.pv[:,1], label='P=-dE/dV')
    >>> plot(efit.ev[:,0], efit.spl_ev(efit.ev[:,0]), label='spline E(V)')
    >>> plot(efit.pv[:,0], efit.spl_pv(efit.pv[:,0]), label='spline P(V)')
    >>> print "V0={v0} E0={e0} B0={b0} P0={p0}".format(**efit.get_min())

    Notes
    -----
    For derived classes:
    Implement _fit(), which sets self.{ev,pv}. `bv` and `spl_bv` are always
    calculated from `ev` or `pv` when :meth:`fit` is called, see also
    :meth:`calc_bv` and :meth:`set_bv_method`.
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
        warnings.warn("ExternEOS is deprecated, use EosFit instead",
                      DeprecationWarning)
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
            print(("After calling the fit() method, find data from '%s' "
                  "in %s/" %(self.app, self.dir)))
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
            raise Exception("unknown bv_method: '%s'" %bv_method)

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
        We have two sources for pressure: (a) The code which calculated E(V),
        i.e. usually some ab initio code (PWscf, ...). (b) Calculated pressure
        P=-dE/dV from the EOS fit to E(V). If the (a) pressure at the E(V)
        minimum is not very close to zero (say ~ 1e-10), then your E-V data is
        incorrect. Usually, this is because of poorly converged calculations
        (low cufoff / bad basis set, too few k-points).
        """
        self.try_set_attr('spl_pv')
        self.try_set_attr('spl_ev')
        self.try_set_attr('spl_bv')
        if self.bv_method == 'pv':
            v0 = self.spl_pv.get_root()
        elif self.bv_method == 'ev':
            v0 = self.spl_ev.get_min()
        else:
            raise Exception("unknown bv_method: '%s'" %bv_method)
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
            raise Exception("unknown value for `behave`: %s" %str(behave))

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
            print(out)
            print((open(os.path.join(self.dir,'PARAM.OUT')).read()))
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

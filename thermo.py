# thermo.py
#
# (Quasi)harmonic approximation.

import numpy as np
from scipy.integrate import simps, trapz
from pwtools.constants import kb, hplanck, R, pi, c0, Ry_to_J, eV,\
    eV_by_Ang3_to_GPa
from pwtools.verbose import verbose
from pwtools import crys, num, mpl

def coth(x):
    return 1.0/np.tanh(x)


class HarmonicThermo(object):
    """Calculate vibrational internal energy (Evib [eV]), free energy (Fvib
    [eV]), entropy (Svib [R,kb]) and isochoric heat capacity (Cv [R,kb]) in the
    harmonic approximation from a phonon density of states. 
    """    
    def __init__(self, freq, dos, temp=None, skipfreq=False, 
                 eps=1.5*np.finfo(float).eps, fixnan=False, nanfill=0.0,
                 dosarea=None, integrator=trapz, verbose=True):
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
        dosarea : float or None
            If not None, then re-normalize the area int(freq) dos to `dosarea`,
            after `skipfreq` was applied if used.
        integrator : callable
            Function which integrates x-y data. Called as ``integrator(y,x)``,
            like ``scipy.integrate.{trapz,simps}``. Usually, `trapz` is
            numerically more stable for weird DOS data and accurate enough if
            the freqeuency axis resolution is big.
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
        self.dosarea = dosarea
        self.integrator = integrator
        
        assert len(self.f) == len(self.dos), ("freq and dos don't have "
                                             "equal length")
        if self.verbose:
            print "HarmonicThermo: number of dos points: %i" %len(self.f)
        
        if self.skipfreq:
            mask = self.f > self.eps
            if self.verbose:
                imask = np.invert(mask)
                nskip = len(imask.nonzero()[0])
                if len(imask) > 0:
                    frms = crys.rms(self.f[imask])
                    drms = crys.rms(self.dos[imask])
                    self._printwarn("HarmonicThermo: skipping %i dos points: "
                        "rms(f)=%e, rms(dos)=%e" %(nskip, frms, drms))
            self.f = self.f[mask]
            self.dos = self.dos[mask]
        
        if self.dosarea is not None:
            self.dos = self._norm_int(self.dos, self.f, area=float(self.dosarea))
    
    def _norm_int(self, y, x, area):
        fx = np.abs(x).max()
        fy = np.abs(y).max()
        sx = x / fx
        sy = y / fy
        _area = self.integrator(sy, sx) * fx * fy
        return y*area/_area
        

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
        return np.array([self.integrator(y[i,:], f) for i in range(y.shape[0])])
    
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

class Gibbs(object):
    """
    Calculate thermodynamic properties on a T-P grid in the quasiharmonic
    approximation, given some variation grid of unit cell axes (`axes_flat`) and
    corresponding phonon DOS data for each grid point.
    
    We have 3 cases for how unit cell axis need to be varied.

    ==== ===============  ========================  ==============   ============
    case axes_flat.shape  cell parameter grid       fitfunc          bulk mod
    ==== ===============  ========================  ==============   ============
    1d   (N,) or (N,1)    ax0 = a (-> V)            G(V), thus any   V*d^2G/dV^2 
                          (cubic)                   EOS will do      

    2d   (N,2)            (ax0,ax1) = (a,b), (a,c)  G(ax0, ax1)      not
                          or (b,c)                                   implemented
                          (e.g. hexaonal: (a,c)) 
    
    3d   not implemented                                  
    ==== ===============  ========================  ==============   ============
    
    Internally, we only deal with `ax0` (1d), `ax0` + `ax1` (2d) and
    `ax0` + `ax1` + `ax2` (3d), thus for 2d and 3d it doesn't matter which cell
    axes is which, you just have to remember :) 

    Attributes
    ----------
    axes : {'ax0': <1d array>, 'ax1': <1d array>, ...}
        The axes of the flattened (unrolled) grid `axes_flat` from np.unique().
    nax0,nax1,nax2 : int
        the length of each axis, i.e. ``len(axes['ax0']), ...``
    fitfunc : {'1d': fit G(V), '2d': fit G(ax0,ax1)}
        Dict with functions to fit G. Must return ``num.Spline``-like object with
        ``get_min()`` method. For 1d, the ``__call__`` method must accept
        the keyword ``der=2`` for the 2nd deriv (for bulk modulus).

    Notes
    -----
    The methods `calc_F` and `calc_G` return dicts with nd-arrays holding
    calculated thermodynamic properites. Naming convention for dict keys
    returned by methods: The keys (strings) mimic HDF5 path names, e.g.
    ``/group0/group1/array``, thus the last name in the path is the name of the
    array (`z` in the examples below). All previous names define the grid and
    thus the dimension of the array. A group name starting with "#" is just a
    prefix. Below, `nx=len(x)`. 
    
    ==============  ==  ================   ================================
    ``/a/z``        1d  (na,)              "z along a-grid"
    ``/a/b/z``      2d  (na,nb)            "z on a-b grid"
    ``/a/b/c/x/z``  4d  (na,nb,nc,nx)
    ``/a-b/z``      1d  (na*nb,)           "z on flattened (double-loop) a-b grid"
    ``/#foo/a/z``   1d  (na,)              "z along a-grid"
    ==============  ==  ================   ================================
    
    Units
    
    =================== =====================
    B,P                 GPa
    T                   K
    F,G                 eV
    ax{0,1,2}           Ang
    Cv,Cp               R (8.4314 J/(mol*K))
    alpha_{V,ax{0,1,2}} 1/K
    =================== =====================
    """
    def __init__(self, T=None, P=None, etot=None, phdos=None, axes_flat=None,
                 volfunc_ax=None, dosarea=None):
        """
        Parameters
        ----------
        T : 1d array 
            temperature [K]
        P : 1d array
            pressure [GPa]
        etot : 1d array, (axes_flat.shape[0],)
            Total energy [eV] for each axes_flat[i,...]
        phdos : sequence (axes_flat.shape[0],)
            Phonon dos arrays for each axes grid point.
            axes_flat[i,...] -> phdos[i] = <2d array (nfreq,2)>
        axes_flat : 1d or 2d array
            Flattened cell axes variation grid (result of nested loop over axes
            to vary). Will be cast to shape (N,1) if 1d with shape (N,) .
                | 1d: (N,) or (N,1) -> vary one cell axis, i.e. cubic cell
                |     ``itertools.product(axes['ax0']) ==  axes['ax0']``
                | 2d: (N,2) -> vary 2 (e.g. a and c for hexagonal)
                |     ``itertools.product(axes['ax0'], axes['ax1'])``
                | 3d: (N,3) -> vary a,b,c (general triclinic)
                |     ``itertools.product(axes['ax0'], axes['ax1'], axes['ax2'])``
        volfunc_ax : callable
            calculate cell volume based on cell axes, 
            V[i] = volfunc_ax(axes_flat[i,...]) where axes_flat[i,...]:
                | 1d: [a0]
                | 2d: [a0, a1]
                | 3d: [a0, a1, a2]
            with a0,a1,a2 the length of the unit cell axes.
        dosarea : see HarmonicThermo    
        """
        self.T = T
        self.P = P
        self.etot = etot
        self.phdos = phdos
        self.volfunc_ax = volfunc_ax
        self.dosarea = dosarea
        self.axes_flat = axes_flat if axes_flat.ndim == 2 else axes_flat[:,None]
        self.nT = len(self.T)
        self.nP = len(self.P)
        self.axes = dict(('ax%i' %jj, np.unique(self.axes_flat[:,jj])) \
            for jj in range(self.axes_flat.shape[1]))
        for key,val in self.axes.iteritems():
            setattr(self, 'n%s' %key, len(val))
        self.V = np.array([self.volfunc_ax(self.axes_flat[ii,...]) for ii \
                 in range(self.axes_flat.shape[0])])
        self.case = None
        if self.axes_flat.shape[1] == 1:
            self.case = '1d'
            self.axes_prefix = '/ax0'
        elif self.axes_flat.shape[1] == 2:
            self.case = '2d'
            self.axes_prefix = '/ax0-ax1'
        else:
            raise StandardError("case 3d not implemented")
        self.fitfunc = {\
            '1d': self._default_fit_1d,
            '2d': self._default_fit_2d}
     
    @staticmethod
    def _default_fit_1d(x,y):
        return num.Spline(x,y, s=None, k=5)
    
    @staticmethod
    def _default_fit_2d(xx,yy,zz):    
        dd = mpl.Data3D(xx=xx, yy=yy, zz=zz)
        return num.Interpol2D(dd=dd, what='bispl', kx=5, ky=5, s=None)

##    @staticmethod
##    def _default_fit_1d(x,y):
##        _eos = eos.ElkEOSFit(energy=y, volume=x, verbose=False)
##        _eos.fit()
##        return _eos.spl_ev
    
    def calc_F(self):
        """
        Calculate free energy properties along T axes for each axes grid point
        (ax0,ax1,ax2) in `self.axes_flat`. Also used by `calc_G()`. Uses
        HarmonicThermo.

        Returns
        -------
        ret : dict
            Keys for 1d: 
                | '/ax0/T/F'
                | '/ax0/T/Fvib'
                ...
            Keys for 2d: 
                | '/ax0-ax1/T/F'
                | '/ax0-ax1/T/Fvib'
                | ...
            Keys for 3d: 
                | '/ax0-ax1-ax2/T/F'
                | '/ax0-ax1-ax2/T/Fvib'
                | ...
            Each array has shape (product(nax0, nax1, nax2), nT), i.e. one
            (nT,) array per axes grid point.
        """
        ret = dict((self.axes_prefix + '/T/%s' %name, 
                    np.empty((self.axes_flat.shape[0], self.nT), 
                             dtype=float)) \
                    for name in ['F', 'Fvib'])
        for idx in range(self.axes_flat.shape[0]):
            ha = HarmonicThermo(freq=self.phdos[idx][:,0], 
                                dos=self.phdos[idx][:,1],
                                temp=self.T, 
                                fixnan=True,
                                skipfreq=True,
                                dosarea=self.dosarea)
            fvib = ha.fvib()
            ret[self.axes_prefix + '/T/F'][idx,:] = self.etot[idx] + fvib                            
            ret[self.axes_prefix + '/T/Fvib'][idx,:] = fvib 
##            ret[self.axes_prefix + '/T/Cv'][idx,:] = ha.cv()
        return ret

    def calc_G(self, ret=None):
        """
        Gibbs free energy and related properties on T-P grid. Uses
        self.fitfunc.

        Parameters
        ----------
        ret : dict
            Result from calc_F(). If None then calc_F() is called here.

        Returns
        -------
        ret : dict
            All keys starting with the ``/#opt`` prefix are values obtained
            from minimizing G(ax0,ax1,ax2,T,P) w.r.t. (ax0,ax1,ax2).         
        """
        if ret is None:
            ret = self.calc_F()
        names = ['ax0', 'ax1', 'ax2', 'V', 'G', 'B']
        ret.update(dict(('/#opt/T/P/%s' %name, np.empty((self.nT,self.nP))) \
                   for name in names))
        if self.case == '1d':
            ret['/T/P/ax0/G'] = np.empty((self.nT,self.nP,self.nax0), dtype=float)
        elif self.case == '2d':
            ret['/T/P/ax0/ax1/G'] = np.empty((self.nT,self.nP,self.nax0,self.nax1), dtype=float)
        for tidx in range(self.nT):
            for pidx in range(self.nP):
                gg = ret[self.axes_prefix + '/T/F'][:,tidx] + self.V * self.P[pidx]  / eV_by_Ang3_to_GPa
                if self.case == '1d':
                    ret['/T/P/ax0/G'][tidx,pidx,:] = gg
                    if self.fitfunc['1d'] is not None:          
                        inter = self.fitfunc['1d'](self.V, gg)
                        vopt = inter.get_min()
                        ret['/#opt/T/P/V'][tidx,pidx] = vopt
                        ret['/#opt/T/P/G'][tidx,pidx] = inter(vopt)
                        spl = num.Spline(self.V, self.axes['ax0'])    
                        ret['/#opt/T/P/ax0'][tidx,pidx] = spl(vopt)
                        ret['/#opt/T/P/B'][tidx,pidx] = vopt * inter(vopt, der=2) * eV_by_Ang3_to_GPa
                    else:
                        raise StandardError("no 1d fit method defined")
                elif self.case == '2d':
                    ret['/T/P/ax0/ax1/G'][tidx,pidx,:,:] = gg.reshape((self.nax0, self.nax1))
                    if self.fitfunc['2d'] is not None:          
                        ggmin = gg.min()
                        ggmax = gg.max()
                        ggscale = (gg - ggmin) / (ggmax - ggmin)
                        inter = self.fitfunc['2d'](xx=self.axes_flat[:,0],
                                                   yy=self.axes_flat[:,1], 
                                                   zz=ggscale)
                        xopt = inter.get_min()
                        ret['/#opt/T/P/ax0'][tidx,pidx] = xopt[0]
                        ret['/#opt/T/P/ax1'][tidx,pidx] = xopt[1]
                        ret['/#opt/T/P/G'][tidx,pidx] = inter(xopt) * (ggmax - ggmin) + ggmin
                        if self.volfunc_ax is not None:
                            ret['/#opt/T/P/V'][tidx,pidx] = self.volfunc_ax(xopt)
                    else:
                        raise StandardError("no 2d fit method defined")
        if self.volfunc_ax is None:
            ret['/#opt/T/P/V'] = None
        if self.case == '1d':
            ret['/#opt/T/P/ax1'] = None
            ret['/#opt/T/P/ax2'] = None
        elif self.case == '2d':            
            ret['/#opt/T/P/ax2'] = None
            ret['/#opt/T/P/B'] = None
        alpha_names = ['ax0', 'ax1', 'ax2', 'V']
        ret.update(dict(('/#opt/T/P/alpha_%s' %name, 
                         np.empty((self.nT,self.nP))) for name in \
                         alpha_names))
        for name in alpha_names:
            arr = ret['/#opt/T/P/%s' %name]
            if arr is not None:
                for pidx in range(self.nP):
                    x = arr[:,pidx]
                    ret['/#opt/T/P/alpha_%s' %name][:,pidx] = \
                        num.deriv_spl(x, self.T, fullout=False, k=5 ,s=None)/x
            else:
                ret['/#opt/T/P/alpha_%s' %name] = None
                        
        ret['/#opt/T/P/Cp'] = np.empty((self.nT, self.nP), dtype=float)
        for pidx in range(self.nP):
            ret['/#opt/T/P/Cp'][:,pidx] = \
                -self.T * num.deriv_spl(ret['/#opt/T/P/G'][:,pidx]*eV/kb, self.T, 
                                        n=2, k=5, s=None, fullout=False)
        ret['/T/T'] = self.T
        ret['/P/P'] = self.P
        for key in ['ax0', 'ax1', 'ax2']:
            name = '/%s/%s' %(key,key)
            if self.axes.has_key(key):
                ret[name] = self.axes[key]
            else:
                ret[name] = None
        return ret                


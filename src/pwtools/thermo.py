"""(Quasi)harmonic approximation. Thermal expansion tools."""

import warnings

import numpy as np
from scipy.integrate import simps, trapz
from pwtools.constants import kb, hplanck, R, pi, c0, Ry_to_J, eV,\
    eV_by_Ang3_to_GPa
from pwtools.verbose import verbose
from pwtools import num, mpl

def coth(x):
    return 1.0/np.tanh(x)


class HarmonicThermo:
    """Calculate vibrational internal energy (Evib [eV]), free energy (Fvib
    [eV]), entropy (Svib [R,kb]) and isochoric heat capacity (Cv [R,kb]) in the
    harmonic approximation from a phonon density of states.
    """
    def __init__(self, freq, dos, T=None, temp=None, skipfreq=False,
                 eps=1.5*num.EPS, fixnan=False, nanfill=0.0,
                 dosarea=None, integrator=trapz, verbose=True):
        """
        Parameters
        ----------
        freq : 1d array
            frequency f (NOT 2*pi*f) [cm^-1]
        dos : 1d array
            phonon dos such that int(freq) dos = 3*natom
        T : 1d array, optional
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
            Use if YKWYAD, test before using! Currently, set all NaNs occurring
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
            the frequency axis resolution is good.
        verbose : bool, optional
            Print warnings. Recommended for testing.

        Notes
        -----
        `skipfreq` and `fixnan`: Sometimes, a few frequencies (usually the 1st
        few values only) are close to zero and negative, and the DOS is very
        small there. `skipfreq` can be used to ignore this region. The default
        is False b/c it may hide large negative frequencies (i.e. unstable
        structure), which is a perfectly valid result (but you shouldn't do
        thermodynamics with that :) Even if there are no negative frequencies,
        you can have frequencies (usually the first) being exactly zero or
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
        if temp is None:
            self.T = T
        else:
            warnings.warn("'temp' is deprecated, use 'T'", DeprecationWarning)
            assert T is None, "Can't use both T and temp."
            self.T = temp
        self.f = freq
        self.dos = dos
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
            print("HarmonicThermo: number of dos points: %i" %len(self.f))

        if self.skipfreq:
            mask = self.f > self.eps
            if self.verbose:
                imask = np.invert(mask)
                nskip = len(imask.nonzero()[0])
                if len(imask) > 0:
                    frms = num.rms(self.f[imask])
                    drms = num.rms(self.dos[imask])
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
        # this call signature works for scipy.integrate,{trapz,simps}
        return self.integrator(y, x=f, axis=1)

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


class Gibbs:
    """
    Calculate thermodynamic properties on a T-P grid in the quasiharmonic
    approximation, given some variation grid of unit cell axes (`axes_flat`) and
    corresponding phonon DOS data for each grid point.

    The `axes_flat` approach allows us to easily calculate per-axis thermal
    expansion rates.

    We have 3 cases for how unit cell axis can be be varied.

    ==== ===============  ========================  ==============   ============
    case axes_flat.shape  cell parameter grid       fitfunc          bulk modulus
    ==== ===============  ========================  ==============   ============
    1d   (N,) or (N,1)    ax0 = a (-> V)            G(V), thus any   V*d^2G/dV^2
                          (e.g. cubic or            EOS will do
                          rhombohedral)

    2d   (N,2)            (ax0,ax1) = (a,b), (a,c)  G(ax0, ax1)      not
                          or (b,c)                                   defined
                          (e.g. hexagonal: (a,c))

    3d   (N,3)            (ax0,ax1,ax2)             no default       not
                          (e.g. triclinic)                           defined
    ==== ===============  ========================  ==============   ============

    Internally, we only deal with `ax0` (1d), `ax0` + `ax1` (2d) and
    `ax0` + `ax1` + `ax2` (3d), thus for 2d and 3d it doesn't matter which cell
    axes is which, you just have to remember :)

    Notes
    -----
    Functions for fitting G(V), V(T) etc are defined in a dictionary attribute
    ``fitfunc``. This is a dict with functions, which have the signature
    ``func(x,y)``. The functions get x-y type data and return a class instance.
    The instances must be a :class:`~pwtools.num.Spline`-like object with a
    ``get_min()`` method. The ``__call__()`` method must accept a keyword `der`
    for calculating derivatives in some cases. See `self._default_fit_*` to get
    an idea: for 1d :class:`~pwtools.num.Spline` or
    :class:`~pwtools.num.PolyFit1D`, for 2d: :class:`~pwtools.num.PolyFit` or
    :class:`~pwtools.num.Interpol2D`. See also ``scipy.interpolate``. Use
    :meth:`~pwtools.thermo.Gibbs.set_fitfunc` to change. This can (and should!)
    be used to tune fitting methods. See examples below for how to change fit
    functions.

    Here is a list of all valid `fitfunc` keys and what they do:

    ========  ==============================================================
    key       value
    ========  ==============================================================
    '1d-G'    fit G(V), ``__call__(V,der=2)`` for B(V) =
              V*d^2G/dV^2
    '2d-G'    fit G(ax0,ax1)
    '3d-G'    fit G(ax0,ax1,ax2), no default, define and test youself,
              please
    '1d-ax'   fit V(ax0)
    'alpha'   fit x_opt(T), x=ax0,ax1,ax2,V,
              ``__call__(T,der=1)`` for alpha_x = 1/x * dx/dT
    'C'       fit G_opt(T), ``__call__(T,der=2)`` for
              Cp = -T * d^G_opt(T,P)/dT^2
    ========  ==============================================================

    Hints for which fitfuncs to choose: The defaults are pretty good, but
    nevertheless testing is mandatory! Good choices are
    :class:`~pwtools.num.PolyFit` / :class:`~pwtools.num.PolyFit1D` for all
    axes-related grids (1d-G, 2d-G, 1d-ax) since you won't have too many
    points here. Then a fit is better than a :class:`~pwtools.num.Spline`.
    Always use ``PolyFit(..., scale=True)`` or scale the x-y data manually
    before fitting to the same order of magnitude! For T grids, choose a very
    fine T-axis (e.g. ``T=linspace(.., num=300)`` and use a
    :class:`~pwtools.num.Spline`. Needed to resolve details of alpha(T) and
    C(T).

    The methods :meth:`calc_F`, :meth:`calc_H` and :meth:`calc_G` return dicts
    with nd-arrays holding calculated thermodynamic properties. Naming
    convention for dict keys returned by methods: The keys (strings) mimic HDF5
    path names, e.g. ``/group0/group1/array``, thus the last name in the path
    is the name of the array (`z` in the examples below). All previous names
    define the grid and thus the dimension of the array. A group name starting
    with "#" is just a prefix. Below, `na=len(a)`, etc.

    ==============  ====  ================   ================================
    key             ndim  shape              description
    ==============  ====  ================   ================================
    ``/a/z``        1d    (na,)              z on 1d a-grid
    ``/a/b/z``      2d    (na,nb)            z on 2d a-b grid
    ``/a/b/c/z``    3d    (na,nb,nc)         z on 3d a-b-c grid
    ``/a-b/z``      1d    (na*nb,)           z on flattened a-b grid
    ``/a-b-c/z``    1d    (na*nb*nc,)        z on flattened a-b-c grid
    ``/#opt/a/z``   1d    (na,)              optimized z on a-grid
    ==============  ====  ================   ================================

    `axes_flat` : Usually, flat grids like "ax0-ax1" or "ax0-ax1-ax2" are
    created by nested loops ``[(ax0_i,ax1_i) for ax0_i in ax0 for ax1_i in
    ax1]`` and therefore have shape (nax0*nax1,2) or (nax0*nax1*nax2,3) . But
    that is not required. They can be completely unstructured (e.g. if points
    have been added later to the grid manually) -- only `fitfunc` must be able
    to handle that.

    Units

    =================== =====================
    B,P                 GPa
    T                   K
    F,G                 eV
    ax{0,1,2}           Ang
    Cv,Cp               R (8.4314 J/(mol*K))
    alpha_{V,ax{0,1,2}} 1/K
    =================== =====================

    Examples
    --------

    See ``test/test_gibbs.py`` for worked examples using fake data.
    Really, go there and have a look. Now!

    >>> from pwtools import mpl, crys, eos
    >>> # cubic cell, 1d case
    >>> volfunc_ax = lambda x: crys.volume_cc(np.array([[x[0]]*3 + [90]*3]))
    >>> gibbs=Gibbs(axes_flat=..., etot=..., phdos=...,
    ...             T=linspace(5,2500,100), P=linspace(0,20,5),
    ...             volfunc_ax=volfunc_ax)
    >>> # EOS fit for G(V), polynomial for heat capacity (default is Spline)
    >>> gibbs.set_fitfunc('C', lambda x,y: num.PolyFit1D(x,y,deg=5,scale=True))
    >>> gibbs.set_fitfunc('1d-G', lambda x,y: eos.EosFit(x,y))
    >>> g = gibbs.calc_G(calc_all=True)
    >>> # 1d case
    >>> V = g['/ax0/V']; G=g['/T/P/ax0/G']; T=g['/T/T']
    >>> # plot G(V) for all T and P=20 GPa
    >>> plot(V, G[:,-1,:].T)
    >>> # Cp(T) for all P
    >>> plot(T,g['/#opt/T/P/Cp'])
    >>> # 2d case plot G(ax0,ax1) for T=2500 K, P=0 GPa
    >>> G=g['/T/P/ax0-ax1/G']
    >>> d=mpl.Data2D(XY=axes_flat, zz=G[-1,0,:])
    >>> fig,ax=mpl.fig_ax3d(); ax.scatter(d.xx,d.yy,d.zz); show()
    """
    def __init__(self, T=None, P=None, etot=None, phdos=None, axes_flat=None,
                 volfunc_ax=None, case=None, **kwds):
        """
        Parameters
        ----------
        T : 1d array
            temperature [K]
        P : 1d array
            pressure [GPa]
        etot : 1d array, (axes_flat.shape[0],)
            Total energy [eV] for each axes_flat[i,...]
        phdos : sequence of 2d arrays (axes_flat.shape[0],)
            Phonon dos arrays (nfreq, 2) for each axes grid point.
            axes_flat[i,...] -> phdos[i] = <2d array (nfreq,2)>, units see
            :class:`HarmonicThermo`. For each array ``phdos[i][:,0] = freq``,
            ``phdos[i][:,1] = dos``. Those are passed to
            :class:`HarmonicThermo`.
        axes_flat : 1d or 2d array
            Flattened cell axes variation grid (for example result of nested
            loop over axes to vary). Will be cast to shape (N,1) if 1d with
            shape (N,) .
                | 1d: (N,) or (N,1) -> vary one cell axis, i.e. cubic cell
                | 2d: (N,2) -> vary 2 (e.g. a and c for hexagonal)
                |     example: ``itertools.product(ax0, ax1)``
                | 3d: (N,3) -> vary a,b,c (general triclinic)
                |     example: ``itertools.product(ax0, ax1, ax2)``
        volfunc_ax : callable
            calculate cell volume based on cell axes,
            V[i] = volfunc_ax(axes_flat[i,...]) where axes_flat[i,...]:
                | 1d: [a0]
                | 2d: [a0, a1]
                | 3d: [a0, a1, a2]
            with a0,a1,a2 the length of the unit cell axes.
        case : str, optional
            '1d', '2d', '3d' or None. If None then it will be determined from
            axes_flat.shape[1]. Can be used to evaluate "fake" 1d data: set
            case='1d' but let `axes_flat` be (N,2) or (N,3)
        **kwds: keywords
            passed to HarmonicThermo and added here as `self.<key>=<value>`
        """
        assert axes_flat.shape[0] == len(phdos), ("axes_flat and phdos "
            "not equally long")
        assert axes_flat.shape[0] == len(etot), ("axes_flat and etot "
            "not equally long")
        self.kwds = dict(verbose=False, fixnan=False, skipfreq=True,
                         dosarea=None)
        self.kwds.update(kwds)
        for k,v in self.kwds.items():
            setattr(self, k, v)
        self.T = T
        self.P = P
        self.etot = etot
        self.phdos = phdos
        self.volfunc_ax = volfunc_ax
        self.axes_flat = axes_flat if axes_flat.ndim == 2 else axes_flat[:,None]
        self.nT = len(self.T)
        self.nP = len(self.P)
        self.npoints = self.axes_flat.shape[0]
        self.nax = self.axes_flat.shape[1]
        self.V = np.array([self.volfunc_ax(self.axes_flat[ii,...]) for ii \
                 in range(self.npoints)])
        self.case = case
        self.fitax = None
        if self.nax == 1:
            if self.case is None:
                self.case = '1d'
            self.axes_prefix = '/ax0'
        elif self.nax == 2:
            if self.case is None:
                self.case = '2d'
            self.axes_prefix = '/ax0-ax1'
        elif self.nax == 3:
            if self.case is None:
                self.case = '3d'
            self.axes_prefix = '/ax0-ax1-ax2'
        else:
            raise ValueError(f"{self.axes_flat.shape=} not supported")
        self.ret = {}
        self.ret['/T/T'] = self.T
        self.ret['/P/P'] = self.P
        self.ret['%s%s' %((self.axes_prefix,)*2)] = self.axes_flat
        self.ret['%s/V' %self.axes_prefix] = self.V
        self.ret[self.axes_prefix + '/Etot'] = self.etot
        self.fitfunc = {\
            '1d-G': self._default_fit_1d_G,
            '2d-G': self._default_fit_2d_G,
            '3d-G': self._default_fit_3d_G,
            '1d-ax': self._default_fit_1d_ax,
            'alpha': self._default_fit_alpha,
            'C': self._default_fit_C,
            }

    def set_fitfunc(self, what, func):
        """Update dict with fitting functions: ``self.fitfunc[what] = func``.

        Parameters
        ----------
        what : str
            One of ``self.fitfunc.keys()``
        func : callable
            function with signature ``func(x,y)`` which returns a
            :class:`~pwtools.num.Spline`-like object, e.g. ``Spline(x,y,k=5)``
        """
        assert what in self.fitfunc, ("unknown key: '%s'" %what)
        self.fitfunc[what] = func

    @staticmethod
    def _default_fit_1d_G(x, y):
        return num.PolyFit1D(x, y, deg=5, scale=True)

    @staticmethod
    def _default_fit_2d_G(points, values):
        return num.PolyFit(points, values, deg=5, scale=True)

    @staticmethod
    def _default_fit_3d_G(points, values):
        ##return num.PolyFit(points, values, deg=5, scale=True)
        # PolyFit probably works, but force user to specify own 3d fit method.
        # Probably sklearn.gaussian_process.GaussianProcessRegressor or smth is
        # even better.
        raise NotImplementedError(
            "Please set a 3d fit method using Gibbs.set_fitfunc()"
            )

    @staticmethod
    def _default_fit_1d_ax(x, y):
        return num.PolyFit1D(x, y, deg=5, scale=True)

    @staticmethod
    def _default_fit_alpha(x, y):
        return num.Spline(x, y, k=5, s=None)

    @staticmethod
    def _default_fit_C(x, y):
        return num.Spline(x, y, k=5, s=None)

    def _fit_opt_store(self, ret, gg, prfx='/T/P', ghsym='G', tpidx=None):
        """For each (T,P) or (P,), fit G(ax0,...) or H(ax0,...) minimize and store
        properties in `ret`. Used in /T/P loop in calc_G and /P loop in calc_H.

        Parameters
        ----------
        ret : dict
        gg : G(ax0,...) or H(ax0,...)
        prfx : str
            '/P' for H in :meth:`calc_H`, '/T/P' for G in :meth:`calc_G`
        ghsym : str
            'H' or 'G'
        tpidx : list
            [tidx,pidx] in calc_G
            [pidx] in calc_H
        """
        tpsl = tuple(tpidx)
        if self.case == '1d':
            fit = self.fitfunc['1d-G'](self.V, gg)
            vopt = fit.get_min()
            ret['/#opt%s/V' %prfx][tpsl] = vopt
            ret['/#opt%s/%s' %(prfx,ghsym)][tpsl] = fit(vopt)
            # Loop needed for fake-1d case when we set case='1d'
            # by hand but self.axes_flat.shape = (N,2) or (N,3). Also,
            # we fit G(V) and not G(ax0) for that reason.
            for iax in range(self.nax):
                ret['/#opt%s/ax%i' %(prfx,iax)][tpsl] = self.fitax[iax](vopt)
            ret['/#opt%s/B' %prfx][tpsl] = vopt * fit(vopt, der=2) * eV_by_Ang3_to_GPa
        elif self.case == '2d':
            assert self.axes_flat.shape[1] == 2, (
                f"{self.axes_flat.shape[1]=} makes no sense for {self.case=}"
            )
            # XXX The fit function alone should take care of scaling, see
            # num.PolyFit(..., scale=True). Doing this here is OK but redundant.
            # Maybe also introduces additional numerical noise?
            ggmin = gg.min()
            ggmax = gg.max()
            ggscale = (gg - ggmin) / (ggmax - ggmin)
            fit = self.fitfunc['2d-G'](self.axes_flat, ggscale)
            xopt = fit.get_min()
            ret['/#opt%s/ax0' %prfx][tpsl] = xopt[0]
            ret['/#opt%s/ax1' %prfx][tpsl] = xopt[1]
            ret['/#opt%s/%s' %(prfx, ghsym)][tpsl] = fit(xopt) * (ggmax - ggmin) + ggmin
            if self.volfunc_ax is not None:
                ret['/#opt%s/V' %prfx][tpsl] = self.volfunc_ax(xopt)
        elif self.case == '3d':
            ##assert self.axes_flat.shape[1] == 3, (
            ##    f"{self.axes_flat.shape[1]=} makes no sense for {self.case=}"
            ##)
            # We have _default_fit_3d_G (raises NotImplementedError ATM), but
            # need to test a modification of the fitting business of case=2d
            # with a real-world use case before implementing this feature
            # publicly.
            raise ValueError("Fits for 3d-G not yet supported, only fake-1d where "
                             "case='1d' but axes_flat.shape[1] > 1.")
        else:
            raise ValueError(f"Unknown {case=}")

    def _set_not_calc_none(self, ret, prfx='/T/P'):
        """We know that the stuff below was not calculated, so set them to
        None. Needs to be done since all arrays are initialized to be np.empty().
        Little hackish, but OK For Me (tm)."""
        if self.volfunc_ax is None:
            ret['/#opt%s/V' %prfx] = None
        if self.nax == 1:
            ret['/#opt%s/ax1' %prfx] = None
            ret['/#opt%s/ax2' %prfx] = None
        elif self.nax == 2:
            ret['/#opt%s/ax2' %prfx] = None
            if self.case != '1d':
                ret['/#opt%s/B' %prfx] = None

    def _set_fitax(self):
        """Store list of Spline-like fit objects, one for each ax. Each
        fit(V,ax) maps the volume to each axis. Only used if self.case=='1d'."""
        if self.fitax is None:
            self.fitax = []
            for iax in range(self.nax):
                self.fitax.append(self.fitfunc['1d-ax'](self.V,
                                                        self.axes_flat[:,iax]))

    def calc_F(self, calc_all=True):
        """Free energy properties along T axis for each axes grid point
        (ax0,ax1,ax2) in `self.axes_flat`. Also used by :meth:`calc_G`. Uses
        :class:`~pwtools.thermo.HarmonicThermo`.

        Parameters
        ----------
        calc_all : bool
            Calculate only F, Fvib or all (F, Fvib, Evib, Svib, Cv)

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
        """
        self._set_fitax()
        if calc_all:
            names = ['F', 'Fvib', 'Evib', 'Svib', 'Cv']
        else:
            names = ['F', 'Fvib']
        ret = dict((self.axes_prefix + '/T/%s' %name,
                    np.empty((self.npoints, self.nT), dtype=float)) \
                    for name in names)
        for idx in range(self.npoints):
            if self.verbose:
                print("calc_F: axes_flat idx = %i" %idx)
            ha = HarmonicThermo(freq=self.phdos[idx][:,0],
                                dos=self.phdos[idx][:,1],
                                T=self.T,
                                **self.kwds)
            fvib = ha.fvib()
            ret[self.axes_prefix + '/T/F'][idx,:] = self.etot[idx] + fvib
            ret[self.axes_prefix + '/T/Fvib'][idx,:] = fvib
            if calc_all:
                svib = ha.svib()
                cv = ha.cv()
                evib = fvib + self.T * svib # or call ha.evib()
                ret[self.axes_prefix + '/T/Svib'][idx,:] = svib
                ret[self.axes_prefix + '/T/Evib'][idx,:] = evib
                ret[self.axes_prefix + '/T/Cv'][idx,:] = cv
        ret.update(self.ret)
        return ret

    def calc_H(self, calc_all=True):
        """Enthalpy H=E+P*V, B and related properties on P grid
        without zero-point contributions. Doesn't need any other method's
        results.

        The results of :meth:`calc_G` will in general be _different_ from those
        calculated here in the limit T->0 (typical use case: T=5K) because of
        zero-point contributions. Check the bulk modulus, for example.
        """
        ret = {}
        self._set_fitax()
        ret['/P' + self.axes_prefix + '/H'] = np.empty((self.nP,self.npoints), dtype=float)
        for pidx in range(self.nP):
            gg = self.etot + self.V * self.P[pidx]  / eV_by_Ang3_to_GPa
            ret['/P' + self.axes_prefix + '/H'][pidx,:] = gg

        if calc_all:
            names = ['ax0', 'ax1', 'ax2', 'V', 'H', 'B']
            ret.update(dict(('/#opt/P/%s' %name, np.empty((self.nP,))) \
                       for name in names))
            for pidx in range(self.nP):
                if self.verbose:
                    print("calc_H: pidx = %i" %(pidx))
                gg = ret['/P' + self.axes_prefix + '/H'][pidx,:]
                self._fit_opt_store(ret, gg, prfx='/P',
                                    tpidx=[pidx], ghsym='H')
        self._set_not_calc_none(ret, prfx='/P')
        ret.update(self.ret)
        return ret

    def calc_G(self, ret=None, calc_all=True):
        """Gibbs free energy and related properties on T-P grid.
        Uses self.fitfunc.

        Needs :meth:`calc_F` results. Called here if not provided.
        Also calls :meth:`calc_H` if `calc_all` is ``True``, i.e. this is the
        you-get-it-all method and the only one you should really use.

        Parameters
        ----------
        ret : dict, optional
            Result from calc_F(). If None then calc_F() is called here. Can be
            used to add additional contributions to F, such as electronic
            entropy.
        calc_all : bool
            Calculate thermal properties from G(ax0,ax1,ax2,T,P): Cp,
            alpha_x, B. If False, then calculate and store only G.

        Returns
        -------
        ret : dict
            All keys starting with the ``/#opt`` prefix are values obtained
            from minimizing G(ax0,ax1,ax2,T,P) w.r.t. (ax0,ax1,ax2).
        """
        self._set_fitax()
        if ret is None:
            ret = self.calc_F(calc_all=calc_all)
        ret['/T/P' + self.axes_prefix + '/G'] = np.empty((self.nT,self.nP,self.npoints), dtype=float)
        for tidx in range(self.nT):
            for pidx in range(self.nP):
                gg = ret[self.axes_prefix + '/T/F'][:,tidx] + self.V * self.P[pidx]  / eV_by_Ang3_to_GPa
                ret['/T/P' + self.axes_prefix + '/G'][tidx,pidx,:] = gg

        if calc_all:
            ret.update(self.calc_H(calc_all=calc_all))
            names = ['ax0', 'ax1', 'ax2', 'V', 'G', 'B']
            ret.update(dict(('/#opt/T/P/%s' %name, np.empty((self.nT,self.nP))) \
                       for name in names))

            for tidx in range(self.nT):
                for pidx in range(self.nP):
                    if self.verbose:
                        print("calc_G: tidx = %i, pidx = %i" %(tidx,pidx))
                    gg = ret['/T/P' + self.axes_prefix + '/G'][tidx,pidx,:]
                    self._fit_opt_store(ret, gg, prfx='/T/P',
                                        tpidx=[tidx,pidx], ghsym='G')
            self._set_not_calc_none(ret, prfx='/T/P')

            alpha_names = ['ax0', 'ax1', 'ax2', 'V']
            ret.update(dict(('/#opt/T/P/alpha_%s' %name,
                             np.empty((self.nT,self.nP))) for name in \
                             alpha_names))
            for name in alpha_names:
                arr = ret['/#opt/T/P/%s' %name]
                if arr is not None:
                    for pidx in range(self.nP):
                        x = arr[:,pidx]
                        fit = self.fitfunc['alpha'](self.T, x)
                        ret['/#opt/T/P/alpha_%s' %name][:,pidx] = fit(self.T, der=1) / x
                else:
                    ret['/#opt/T/P/alpha_%s' %name] = None

            ret['/#opt/T/P/Cp'] = np.empty((self.nT, self.nP), dtype=float)
            for pidx in range(self.nP):
                fit = self.fitfunc['C'](self.T, ret['/#opt/T/P/G'][:,pidx]*eV/kb)
                ret['/#opt/T/P/Cp'][:,pidx] = -self.T * fit(self.T, der=2)
        ret.update(self.ret)
        return ret


def debye_func(x, nstep=100, zero=1e-8):
    r"""Debye function

    :math:`f(x) = 3 \int_0^1 t^3 / [\exp(t x) - 1] dt`

    Parameters
    ----------
    x : float or 1d array
    nstep : int
        number of points for integration
    zero : float
        approximate the 0 in the integral by this (small!) number
    """
    x = np.atleast_1d(x)
    if x.ndim == 1:
        x = x[:,None]
    else:
        raise Exception("x is not 1d array")
    tt = np.linspace(zero, 1.0, nstep)
    return 3.0 * trapz(tt**3.0 / (np.exp(tt*x) - 1.0), tt, axis=1)


def einstein_func(x):
    r"""Einstein function

    :math:`f(x) = 1 / [ \exp(x) - 1 ]`

    Parameters
    ----------
    x : float or 1d array
    """
    x = np.atleast_1d(x)
    return 1.0 / (np.exp(x) - 1.0)


def expansion(temp, alpha, theta, x0=1.0, func=debye_func):
    """Calculate thermal expansion according to the model in `func`.

    Parameters
    ----------
    temp : array_like
        temperature
    alpha : float
        high-T limit expansion coeff
    theta
        Debye or Einstein temperature
    x0 : float
        axis length at T=0
    func : callable
        Usually :func:`debye_func` or :func:`einstein_func`

    Examples
    --------
    >>> # thermal expansion coeff alpha_x = 1/x(T) * dx/dT
    >>> from pwtools.thermo import expansion, debye_func, einstein_func
    >>> from pwtools import num
    >>> T=linspace(5,2500,100)
    >>> for zero in [1e-3, 1e-8]:
    >>>     x = expansion(T, 5e-6, 1200, 3, lambda x: debye_func(x,zero=zero))
    >>>     plot(T, num.deriv_spl(x, T, n=1)/x)
    >>> x = expansion(T, 5e-6, 1200, 3, einstein_func)
    >>> plot(T, num.deriv_spl(x, T, n=1)/x)

    References
    ----------
    [1] Figge et al., Appl. Phys. Lett. 94, 101915 (2009)
    """
    return x0 * (1.0 + alpha * theta * func(theta / temp))

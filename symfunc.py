import cPickle
from math import cos, acos, exp, pi
import numpy as np
from pwtools import crys, comb, _flib, _fsymfunc

# MD engines
# ----------
#
# To start, simply use ASE.
#
# lammps: Look at their python interface. We really only want the MD integrator
# from lammps: time step loop in python, at each step we need to somehow
# set_forces() into lammps (instead of predefining any force field), let
# lammps do *one* step with these forces (using thermostats and all that crazy
# shit), get new struct, query net, new forces, ... That would be perfect!

def precond(arr, rng=[0,1], shift=True, scale=True, atol=1e-8):
    """Precondition `arr` by `shift`-ing the mean value (of each row for 2d array)
    to zero and/or `scale` all array values to range `rng`, based on overall min
    and max.

    Parameters
    ----------
    arr : 1d or 2d array
    rng : seq of [min,max], optional
    shift, scale : bool, optional
    atol : float, optional
        Don't apply scaling if all values in the array are closer then `atol`.
        This is b/c if all array elements almost equal, we would divide zero.
    
    Returns
    -------
    out : same shape as `arr`

    Examaples
    ---------
    >>> precond(array([[1,2,3],[4,5,6]]))
    array([[ 0. ,  0.5,  1. ],
           [ 0. ,  0.5,  1. ]])
    >>> precond(array([[1,2,3],[4,6,8]]))
    array([[ 0.25,  0.5 ,  0.75],
           [ 0.  ,  0.5 ,  1.  ]])
    >>> precond(array([1,2,3]))
    array([ 0. ,  0.5,  1. ])
    >>> precond(array([1,2,3])[:,None])
    array([[ 0. ],
           [ 0.5],
           [ 1. ]])
    >>> precond(array([1,2,3])[None,:])
    array([[ 0. ,  0.5,  1. ]])
    """
    assert arr.ndim <= 2
    if shift:
        # shape (N,M) with M > 1
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr - arr.mean(axis=1)[:,None]
        else:
            # shape (N,) or (N,1)
            arr = arr - arr.mean()
    if scale:
        amin = arr.min()
        amax = arr.max()
        if abs(amax - amin) > atol:
            arr = (arr - amin) / (amax - amin) * (rng[1] - rng[0]) + rng[0]
    return arr

def l2a(lists):
    """Shorcut function. list of lists -> array of nested loops results.
    
    Examples
    --------
    >>> l2a([[1,2],[3,4]])
    array([[1, 3],
           [1, 4],
           [2, 3],
           [2, 4]])
    """
    return np.array(comb.nested_loops(lists))

# function cores, without cutfunc multiplied, no sum

def cutfunc(rij, p0):
    ret = 0.5*(np.cos(pi*rij/p0) + 1.0)
    ret[rij > p0] = 0.0
    return ret

def c_g2(rij, p1, p2):
    return np.exp(-p1*(rij - p2)**2.0)

def c_g3(rij, p1):
    return np.cos(p1*rij)

def c_g45_ang(cos_anglesijk, p1, p2):
    return 2.0**(1.0-p1) * (1.0+p2*cos_anglesijk)**p1


class SymFunc(object):
    """
    Symmetry functions from [Behler]_. 

    Examples
    --------
    >>> from pwtools import comb, crys, symfunc
    >>> # make some structure
    >>> rs=crys.RandomStructure(['Si']*100) 
    >>> st=rs.get_random_struct()
    >>> # create SymFunc, can also use sf=SymFunc(); sf.set_struct(st)
    >>> sf=symfunc.SymFunc(st)
    >>> # create and set set parameters
    >>> rcut=[6.0] 
    >>> eta=np.logspace(-3,-0,8,base=10.0) 
    >>> rs=np.linspace(0.1,6.0,5)
    >>> kappa=np.linspace(1,6.0,3)
    >>> params2=np.array(comb.nested_loops([rcut,eta,rs]))
    >>> params3=np.array(comb.nested_loops([rcut,kappa]))
    >>> sf.set_params(params2,2)
    >>> sf.set_params(params3,3)
    >>> # calculate
    >>> vals=sf.gall(sel=[2,3])
    >>> plot(vals.T)
    >>> # or individual funcs, then you can skip set_params() and use the
    >>> # params arg, else the value set by set_params() is used
    >>> vals2=sf.g2(params=params2)

    References
    ----------
    .. [Behler] Behler, J. "Atom-centered symmetry functions for constructing
                high-dimensional neural network potentials", Journal of
                Chemical Physics, 2011, 134, 074106
    """
    def __init__(self, struct=None, precond=True):
        if struct is not None:
            self.set_struct(struct)
        self.precond = precond
        self.params_all = dict((x,None) for x in [1,2,3,4,5])
        self.npsets_all = dict((nn, 0) for nn in self.params_all.keys())
    
    def _precond(self, arr):
        if self.precond:
            return precond(arr, rng=[0,1], scale=True, shift=True, atol=1e-13)
        else:
            return arr
    
    def set_struct(self, struct):
        """
        Set a Structure, calculate distances and angles.

        Parameters
        ----------
        struct : Structure instance
        """
        self.struct = struct
        self.rcut_max = crys.rmax_smith(struct.cell)
        self.distsq, self.distvecs, self.distvecs_frac = \
            crys.distances(self.struct,
                           pbc=True,
                           squared=True,
                           fullout=True)
        del self.distvecs_frac                              
        self.dists = np.sqrt(self.distsq)
        self.min_dist = self.dists[self.dists > 0].min()
    
    def dump(self, fn, protocol=2):
        self.struct = None
        cPickle.dump(self, open(fn, 'wb'), protocol=protocol)

    def get_default_params_all(self):
        """Set some default params for all symfuncs. 
        
        The values are probably not very useful. Use this as an example for how
        to generate parameters for the individual symfuncs.

        Returns
        -------
        params_all : dict
            keys = [1,2,3,4,5], values are 2d arrays of shape (npsets,X) where
            npsets = the number of generated parameter sets, X=the number of
            parameters each function takes (1..4, see the function's doc
            strings). Note that `npsets` can be different for each fucntion.
        """
        rmx = self.rcut_max
        ls = np.linspace
        rcut = [rmx]
        zeta = ls(1,30,8)
        eta = np.logspace(-3,-0,8,base=10.0)
        rs = ls(0.1,rmx,5)
        kappa = ls(1,rmx,3)
        lamb = [1]
        tmp = \
            {1: [rcut],
             2: [rcut, eta, rs],
             3: [rcut, kappa],
             4: [rcut, zeta, lamb, eta],
             5: [rcut, zeta, lamb, eta],
             }
        params_all = {}
        for nn, lst in tmp.iteritems():
            params_all[nn] = np.array(comb.nested_loops(lst))
        return params_all
    
    def set_params(self, params, what):
        """Define params array for a symfunc. You can also call each symfunc
        directly as ``sf=SymFunc(); sf.g1(params=params)``.

        Parameters
        ----------
        params : 2d array
            See g1() .. gN() for dimensions.
        what : int
            1...N, select sym func 
        """
        self.params_all[what] = params
        self.npsets_all[what] = params.shape[0]
    
    def set_params_all(self, params_all):
        """Define `params_all` dict.

        Parameters
        ----------
        params_all : dict
        """
        for what,params in params_all.iteritems():
            self.params_all[what] = params
            self.npsets_all[what] = params.shape[0]

    # XXX maybe remove later ...
    def cos_angle(self, ii, jj, kk):
        dvij = self.distvecs[ii,jj,:]
        dvik = self.distvecs[ii,kk,:]
        cos_angle = np.dot(dvij, dvik) \
                    / self.dists[ii,jj] \
                    / self.dists[ii,kk]
        return cos_angle                    

    def cutfunc(self, rcut):
        """
        Cutoff function applied to each distance in self.dists

        Parameters
        ----------
        rcut : float

        Returns
        -------
        2d array (natoms,natoms), same shape as self.dists
        """
        ret = 0.5*(np.cos(pi*self.dists/rcut) + 1.0)
        ret[self.dists > rcut] = 0.0
        return ret

    def g1(self, params=None):
        """
        Parameters
        ----------
        params : 2d array (npsets, 1), optional

        Returns
        -------
        2d array (natoms,npsets)
        """
        params = self.params_all[1] if params is None else params
        assert params.shape[1] == 1
        ret = np.empty((self.dists.shape[0], params.shape[0]), dtype=float)
        idx = 0
        for p0 in params[:,0]:
            ret[:,idx] = self.cutfunc(p0).sum(axis=1)
            idx += 1
        return self._precond(ret)
       
    def g2(self, params=None):
        """
        Parameters
        ----------
        params : 2d array (npsets, 3), optional

        Returns
        -------
        2d array (natoms, npsets)
        """
        params = self.params_all[2] if params is None else params
        assert params.shape[1] == 3
        ret = np.empty((self.dists.shape[0], params.shape[0]), dtype=float)
        idx = 0
        for p0,p1,p2 in params:
            tmp = np.exp(-p1*(self.dists-p2)**2.0) * self.cutfunc(p0)
            ret[:,idx] = tmp.sum(axis=1)
            idx += 1
        return self._precond(ret)

    def g3(self, params=None):
        """
        Parameters
        ----------
        params : 2d array (npsets, 2), optional

        Returns
        -------
        2d array (natoms, npsets)
        """
        params = self.params_all[3] if params is None else params
        assert params.shape[1] == 2
        ret = np.empty((self.dists.shape[0], params.shape[0]), dtype=float)
        idx = 0
        for p0,p1 in params:
            tmp = np.cos(p1*self.dists) * self.cutfunc(p0)
            ret[:,idx] = tmp.sum(axis=1)
            idx += 1
        return self._precond(ret)

    def g4_py(self, params=None):
        """
        Python reference implementation.

        Parameters
        ----------
        params : 2d array (npsets, 4), optional

        Returns
        -------
        2d array (natoms, npsets)
        """
        params = self.params_all[4] if params is None else params
        assert params.shape[1] == 4
        ret = np.zeros((self.distsq.shape[0], params.shape[0]), dtype=float)
        idx = 0
        for p0,p1,p2,p3 in params:
            cf = self.cutfunc(p0)
            for ii in range(self.struct.natoms): 
                for jj in range(self.struct.natoms): 
                    for kk in range(self.struct.natoms): 
                        if (ii != jj and ii != kk and jj != kk):
                            val = 2.0**(1.0-p1) * \
                                  (1.0+p2*self.cos_angle(ii,jj,kk))**p1 * \
                                  exp(-p3*(self.distsq[ii,jj] + \
                                           self.distsq[ii,kk] + \
                                           self.distsq[jj,kk])) * \
                                  cf[ii,jj]*cf[ii,kk]*cf[jj,kk]
                            ret[ii,idx] += val
            idx += 1
        return self._precond(ret)
    
    def g5_py(self, params=None):
        """
        Python reference implementation.

        Parameters
        ----------
        params : 2d array (npsets, 4), optional

        Returns
        -------
        2d array (natoms, npsets)
        """
        params = self.params_all[5] if params is None else params
        assert params.shape[1] == 4
        ret = np.zeros((self.distsq.shape[0], params.shape[0]), dtype=float)
        idx = 0
        for p0,p1,p2,p3 in params:
            cf = self.cutfunc(p0)
            for ii in range(self.struct.natoms): 
                for jj in range(self.struct.natoms): 
                    for kk in range(self.struct.natoms): 
                        if (ii != jj and ii != kk and jj != kk):
                            val = 2.0**(1.0-p1) * \
                                  (1.0+p2*self.cos_angle(ii,jj,kk))**p1 * \
                                  exp(-p3*(self.distsq[ii,jj] + \
                                           self.distsq[ii,kk])) * \
                                  cf[ii,jj]*cf[ii,kk]
                            ret[ii,idx] += val
            idx += 1
        return self._precond(ret)
    
    def g45_f(self, what=None, params=None):
        """
        Wrapper for Fortran version of g4() and g5().

        Parameters
        ----------
        params : 2d array (npsets, 4), optional
        what : int
            4 or 5, select g4 or g5 fortran implementation

        Returns
        -------
        2d array (natoms, npsets)
        """
        params = self.params_all[what] if params is None else params
        assert params.shape[1] == 4
        ret = np.empty((self.distsq.shape[0], params.shape[0]), dtype=float,
                        order='F')
        # symfunc_45(distsq,cos_anglesijk,ret,params,what,[natoms,npsets,nparams,overwrite_ret])
        # symfunc_45_fast(distvecs,dists,ret,params,what,[natoms,npsets,nparams,overwrite_ret])
        _fsymfunc.symfunc_45_fast(self.distvecs, self.dists, ret, params, what)
        return self._precond(ret)
    
    def g4(self, params=None):
        """
        Wrapper for g45_f(..., what=4).

        Parameters
        ----------
        params : 2d array (npsets, 4), optional

        Returns
        -------
        2d array (natoms, npsets)
        """
        params = self.params_all[4] if params is None else params
        return self.g45_f(what=4, params=params)

    def g5(self, params=None):
        """
        Wrapper for g45_f(..., what=5).

        Parameters
        ----------
        params : 2d array (npsets, 4), optional

        Returns
        -------
        2d array (natoms, npsets)
        """
        params = self.params_all[5] if params is None else params
        return self.g45_f(what=5, params=params)
    
    def gall(self, sel=[1,2,3,4,5]):
        """
        Calculate one or more symfuncs (call self.g1, self.g2, ...) based on
        `sel`. 
         
        Return array which holds arrays from all calculated symfuncs
        concatenated along axis=1. Thus, the number of columns equals the total
        number of symfunc values (sum of the number of all parameter sets).

        For instance, if ``self.npsets_all = {1: 1, 2: 40, 3: 3, 4: 64, 5:
        64}`` and ``sel=[2,3,4]``, then we return an array of shape (natoms,
        40+3+64).

        Parameters
        ----------
        sel : sequence of ints 1..5
            Select which symfunc(s) to calculate.
        
        Returns
        -------
        arr : 2d array (natoms, npsets_total)
        """
        nrows = self.struct.natoms
        ncols = sum([nc for iparam,nc in self.npsets_all.iteritems() \
                     if iparam in sel])
        ret = np.empty((nrows, ncols), dtype=float) 
        last = 0
        for ii in sel:
            ncols = self.npsets_all[ii]
            func = getattr(self, 'g%i' %ii)
            ret[:,last:last+ncols] = func()
            last = last+ncols
        return ret


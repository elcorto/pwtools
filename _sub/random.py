# Random structure generation.

from __future__ import absolute_import

from math import acos, pi, sin, cos, sqrt
import numpy as np
from numpy.random import uniform
from pwtools import atomic_data, _flib
from pwtools.crys import cc2cell, volume_cc, Structure
from pwtools.num import fempty

class RandomStructureFail(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg


class RandomStructure(object):
    """Create a random structure, based on atom number and composition alone
    (`symbols`).

    The mean target cell volume and cell side lengths are determined from covalent
    radii of all atoms (see pwtools.atomic_data.covalent_radii).

    Examples
    --------
    >>> rs = crys.RandomStructure(['Si']*50)
    >>> st = rs.get_random_struct()

    For many atoms (~ 200), the creation takes a while b/c as more and more
    atoms are already present, we need many more tries to get another random
    atom into the struct. Then, _atoms_too_close() is called a lot, which is the
    bottleneck. Hint: ``plot(rs.counters['coords'])``.
    """
    # Notes
    # -----
    # * Maybe add outer loop, create new cryst_const and try again if creating
    #   struct failed once. 
    # * Maybe don't raise exceptions if struct creation fails, only print
    #   warning / let user decide -- add arg error={True,False}.
    # * The bottleneck is get_random_struct() -- the loop over
    #   _atoms_too_close().   
    def __init__(self, 
                 symbols, 
                 vol_scale=4, 
                 angle_range = [60.0, 120.0],
                 vol_range_scale=[0.7,1.3],
                 length_range_scale=[0.7,1.3],
                 close_scale=1,
                 cell_maxtry=100,
                 atom_maxtry=1000,
                 verbose=False):
        """
        Parameters
        ----------
        symbols : list of strings
            Atom symbols. Defines number of atoms and composition.
        vol_scale : float
            Scale volume estimated from sum of covalent spheres (lower bound
            for volume). Only values > 1 make sense. This is used
            to get the mean target volume (increase "the holes" between spheres).
            Large values will create large cells with much space between atoms.
        angle_range : sequence of 2 floats [min, max]
            Range of allowed random cell angles.
        vol_range_scale : sequence of 2 floats [min, max]
            Scale estimated mean volume by `min` and `max` to get allowed
            volume range.
        length_range_scale : sequence of 2 floats [min, max]
            Scale estimated mean cell side length (3rd root of mean volume) 
            by `min` and `max` to get allowed cell side range.
        close_scale : float
            Scale allowed distance between atoms (from sum of covalent radii).
            Use < 1.0 to make tightly packed structures, i.e. small values will
            allow close atoms, large values make big spaces but will also
            make the structure generation more likely to fail.
        cell_maxtry / atom_maxtry : integer, optional
            Maximal attempts to create a random cell / insert random atom into
            cell before RandomStructureFail is raised.
        verbose : bool, optional
            Print stuff while trying to generate structs.
        """
        self.symbols = symbols
        self.close_scale = close_scale
        self.angle_range = angle_range
        self.natoms = len(self.symbols)
        self.cell_maxtry = cell_maxtry
        self.atom_maxtry = atom_maxtry
        self.verbose = verbose
        self.cov_radii = np.array([atomic_data.pt[sym]['cov_rad'] for \
                                   sym in symbols])
        self.dij_min = (self.cov_radii + self.cov_radii[:,None]) * self.close_scale
        self.vol_mean = 4.0/3.0 * pi * (self.cov_radii**3.0).sum() * vol_scale
        self.length_mean = self.vol_mean ** (1.0/3.0)
        self.vol_range = [self.vol_mean*vol_range_scale[0], 
                          self.vol_mean*vol_range_scale[1]]
        self.length_range = [self.length_mean * length_range_scale[0],
                             self.length_mean * length_range_scale[1]]
        self.counters = {'cryst_const': None, 'coords': None} 

    def get_random_cryst_const(self):
        """Create random cryst_const.

        Returns
        -------
        cryst_const : 1d array (6,)

        Raises
        ------
        RandomStructureFail
        """
        def _get():
            return np.concatenate((uniform(self.length_range[0], 
                                           self.length_range[1], 3),
                                   uniform(self.angle_range[0], 
                                           self.angle_range[1], 3)))
        cnt = 1
        while cnt <= self.cell_maxtry:
            cc = _get()
            vol = volume_cc(cc)
            if not self.vol_range[0] <= vol <= self.vol_range[1]:
                cc = _get()
                cnt += 1                                 
            else:
                self.counters['cryst_const'] = cnt
                return cc
        raise RandomStructureFail("failed creating random cryst_const")            
    
    def _atoms_too_close(self, iatom):
        """Check if any two atoms are too close, i.e. closer then the sum of
        their covalent radii, scaled by self.close_scale.
        
        Minimum image distances are used.

        Parameters
        ----------
        iatom : int
            Index into self.coords_frac, defining the number of atoms currently
            in there (iatom +1).
        """
        # dist: min image dist correct only up to rmax_smith(), but we check if
        #   atoms are too close; too big dists are no problem; choose only upper
        #   triangle from array `dist`
        natoms_filled = iatom + 1
        coords_frac = self.coords_frac[:natoms_filled,:]
        # This part is 10x slower than the fortran version  --------
        # distvecs_frac: (natoms_filled, natoms_filled, 3)
        # distvecs:      (natoms_filled, natoms_filled, 3)
        # dist:          (natoms_filled, natoms_filled)
        ##distvecs_frac = coords_frac[:,None,:] - coords_frac[None,:,:]
        ##distvecs_frac = min_image_convention(sij)
        ##distvecs = np.dot(sij, self.cell)
        ##dist = np.sqrt((rij**2.0).sum(axis=2))
        #-----------------------------------------------------------
        nn = natoms_filled
        # XXX For maximum speed: dummy1 and dummy2 may be big and are allocated
        # each time this method is called, which may be *many* times. Either
        # re-write the fortran code to not require them as input or allocate
        # them in __init__().
        distsq,dummy1,dummy2 = fempty((nn,nn)), fempty((nn,nn,3)), \
                               fempty((nn,nn,3))
        distsq, dummy1, dummy2 = _flib.distsq_frac(coords_frac,
                                                   self.cell,
                                                   1,
                                                   distsq,
                                                   dummy1,
                                                   dummy2)
        dist = np.sqrt(distsq)                                                            
        # This part is fast
        dij_min_filled = self.dij_min[:natoms_filled,:natoms_filled]
        return np.triu(dist < dij_min_filled, k=1).any()
    
    def _add_random_atom(self, iatom):
        self.coords_frac[iatom,:] = np.random.rand(3)

    def _try_random_struct(self):
        try:
            st = self._get_random_struct()
            return st
        except RandomStructureFail as err:
            if self.verbose:
                print err.msg
            return None
    
    def _get_random_struct(self):
        """Generate random cryst_const and atom coords.

        Returns
        -------
        Structure

        Raises
        ------
        RandomStructureFail
        """
        self.cell = cc2cell(self.get_random_cryst_const())
        self.coords_frac = np.empty((self.natoms, 3))
        self.counters['coords'] = []
        for iatom in range(self.natoms):
            if iatom == 0:
                cnt = 1
                self._add_random_atom(iatom)
            else:                
                cnt = 1
                while cnt <= self.atom_maxtry:
                    self._add_random_atom(iatom)
                    if self._atoms_too_close(iatom):
                        self._add_random_atom(iatom)
                        cnt += 1
                    else:
                        break
            self.counters['coords'].append(cnt)
            if cnt > self.atom_maxtry:
                raise RandomStructureFail("failed to create random coords for "
                                          "iatom=%i of %i" %(iatom,
                                                             self.natoms-1))
        st = Structure(symbols=self.symbols,
                       coords_frac=self.coords_frac,
                       cell=self.cell)        
        return st
    
    def _get_random_struct_nofail(self):
        """Same as _get_random_struct(), but if RandomStructureFail is raised,
        start over.
        
        Returns
        -------
        Structure
        """
        st = self._try_random_struct()
        cnt = 0
        while st is None:
            if self.verbose:
                print "  try: %i" %cnt
            st = self._try_random_struct()
            cnt += 1          
        return st             
    
    def get_random_struct(self, fail=True):
        """Generate random cryst_const and atom coords.
        
        If `fail=True`, then RandomStructureFail may be raised if structure
        generation fails (`cell_maxtry` or  `atom_maxtry` exceeded).

        If `fail=False`, then the process is repeated over and over (may run
        forever). Use this if you know that your settings (all `*_scale`
        inputs) are sensible + struct generation fails "sometimes" and you
        absolutely want to create a struct.

        Parameters
        ----------
        fail : bool, optional

        Returns
        -------
        Structure

        Raises
        ------
        RandomStructureFail (if `fail=True`).
        """
        if fail:
            return self._get_random_struct()
        else:            
            return self._get_random_struct_nofail()

# structure.py
#
# Container classes for crystal structures and trajectories.

from math import pi, cos
import numpy as np
from pwtools.parse import FlexibleGetters
from pwtools import crys, decorators
from pwtools.pwscf import atpos_str
from pwtools.verbose import verbose

# Use this as an interface for all functions/classes which take/operate
# on/return a crystal structure.
#
# This is not used ATM. If we use it as an interface (i.e. all functions in
# crys.py take a Structure instance as input), then also add a length unit,
# i.e. require that cartesian coords `coords` and `cell` are always in Bohr,
# for instance.
#
# The best usage pattern would be if each StructureFileParser derived class
# gets an attr `structure`, which is an instance of Structure.
#
# Problem: The parsing classes derived from StructureFileParser must know in
# which units coords, cell etc are in order to return a Structure instance with
# defined units. This a lot of work and magic b/c each code does it in it's own
# way and usually there are a number different units possible in the output
# files. 
#
# The only solution is that the user must tell StructureFileParser at parsing
# time which units are present in the parsed file. There is no functionality
# there for this, that's way this class here is not used much.
#
# Actually, if one goes this way, one could deprecate Structure just as well
# and include the whole machinery in StructureFileParser's derived classes b/c
# then, conversion to pwtools standard units can be made in each parsing class
# right away. Actually it is easy for Abinit, not so much for PWscf.

class Structure(FlexibleGetters):
    """Container for a single crystal structure (unit cell + atoms).

    This is a defined minimal interface for how to store a crystal structure in
    pwtools, in contrast to the classes in parse.py, which may extract and
    store structure information in any number of ways.

    Similar to parse.StructureFileParser, but with strict attribute checking
    and automatic calculation of some quantities.
    
    In the parsing classes (parse.py), getters are "sloppy". They can return
    None if something is not found. Here all getters return something or raise
    an error.
    """
    @decorators.crys_add_doc
    def __init__(self, 
                 coords=None, 
                 coords_frac=None, 
                 symbols=None, 
                 cell=None,
                 cryst_const=None, 
                 ):
        """
        args:
        -----
        coords : 2d array (natoms, 3)
            Cartesian coords.
            Optional if `coords_frac` given.
        coords_frac : 2d array (natoms, 3)
            Fractional coords w.r.t. `cell`.
            Optional if `coords` given.
        symbols : sequence of strings (natoms,)
            atom symbols
        %(cell_doc)s 
            Vectors are rows.
            Unit is Bohr, Angstrom or any other length.
            Optional if `cryst_const` given.
        %(cryst_const_doc)s
            Optional if `cell` given.

        notes:
        ------
        cell, cryst_const : Provide either `cell` or `cryst_const`, or both
            (which is redundant). If only one is given, the other is calculated
            from it. See crys.{cell2cc,cc2cell}.
        coords, coords_frac : Provide either `coords` or `coords_frac`, or both
            (which is redundant). If only one is given, the other is calculated
            from it. See crys.coord_trans().
        """
        FlexibleGetters.__init__(self)
        
        # from input
        self.coords = coords
        self.coords_frac = coords_frac
        self.symbols = symbols
        self.cell = cell
        self.cryst_const = cryst_const
        
        # derived
        self.atpos_str = None
        self.celldm = None
        self.natoms = None

    def get_coords(self):
        if not self.is_set_attr('coords'):
            req = ['cell', 'coords_frac']
            self.check_get_attrs(req)
            self.assert_attrs(req)
            return crys.coord_trans(coords=self.coords_frac,
                                    old=self.cell,
                                    new=np.identity(3),
                                    align='rows')
        else:
            return self.coords
    
    def get_coords_frac(self):
        if not self.is_set_attr('coords_frac'):
            req = ['cell', 'coords']
            self.check_get_attrs(req)
            self.assert_attrs(req)
            return crys.coord_trans(coords=self.coords,
                                    old=np.identity(3),
                                    new=self.cell,
                                    align='rows')
        else:
            return self.coords_frac
    
    def get_symbols(self):
        self.assert_attr('symbols')
        return self.symbols
    
    def get_cell(self):
        if not self.is_set_attr('cell'):
            self.assert_attr('cryst_const')
            return crys.cc2cell(self.cryst_const)
        else:
            return self.cell
    
    def get_cryst_const(self):
        if not self.is_set_attr('cryst_const'):
            self.assert_attr('cell')
            return crys.cell2cc(self.cell)
        else:
            return self.cryst_const
    
    def get_natoms(self):
        self.assert_attr('symbols') 
        return len(self.symbols)
    
    # XXX if we do this here (Pwscf), then we shall also calculate Abinit stuff
    # (see abinit.AbinitInput).
    def get_atpos_str(self, coords_attr='coords'):
        req = [coords_attr, 'symbols']
        self.check_get_attrs(req)
        self.assert_attrs(req)
        return atpos_str(self.symbols, getattr(self, coords_attr))
    
    def get_ase_atoms(self):
        """Return ASE Atoms object. Obviously, you must have ASE installed."""
        req = ['coords_frac', 'cell', 'symbols']
        self.check_get_attrs(req)
        self.assert_attrs(req)
        # We don't wanna make ase a dependency. Import only when needed.
        from ase import Atoms
        return Atoms(symbols=self.symbols,
                     scaled_positions=self.coords_frac,
                     cell=self.cell,
                     pbc=[1,1,1])

    def verify(self):
        assert self.get_natoms() == self.get_coords.shape[0]


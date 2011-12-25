# structure.py
#
# Container classes for crystal structures and trajectories.

from math import pi, cos
import numpy as np
from pwtools.parse import FlexibleGetters
from pwtools import crys, decorators
from pwtools.pwscf import atpos_str
from pwtools.verbose import verbose
from pwtools import periodic_table
from pwtools.constants import Bohr, Angstrom

# Use this as an interface for all functions/classes which take/operate
# on/return a crystal structure.
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

    All length units (coords, cell, cryst_const[:3]) must be in Bohr. If not,
    then use Structure(..., to_bohr=<factor>) to convert.

    This is a defined minimal interface for how to store a crystal structure in
    pwtools, in contrast to the classes in parse.py, which may extract and
    store structure information in any number of ways (and units).

    Similar to parse.StructureFileParser, but with strict attribute checking
    and automatic calculation of some quantities. In the parsing classes
    (parse.py), getters are "sloppy". They can return None if something is not
    found. Here all getters return something or raise an error.

    This class is very much like ase.Atoms, but without the "calculators".
    You can use get_ase_atoms() to get an Atoms object.

    example:
    --------
    >>> symbols=['N', 'Al', 'Al', 'Al', 'N', 'N', 'Al']
    >>> coords_frac=rand(len(symbols),3)
    >>> cryst_const=np.array([5,5,5,90,90,90.0])
    >>> st=structure.Structure(coords_frac=coords_frac, 
    ...                        cryst_const=cryst_const, 
    ...                        symbols=symbols)
    >>> st.symbols
    ['N', 'Al', 'Al', 'Al', 'N', 'N', 'Al']
    >>> st.symbols_unique
    ['Al', 'N']
    >>> st.order
    2}
    >>> st.typat
    [2, 1, 1, 1, 2, 2, 1]
    >>> st.znucl
    [13, 7]
    >>> st.ntypat
    2
    >>> st.nspecies
    3}
    >>> st.coords
    array([[ 1.1016541 ,  4.52833103,  0.57668453],
           [ 0.18088339,  3.41219704,  4.93127985],
           [ 2.98639824,  2.87207221,  2.36208784],
           [ 2.89717342,  4.21088541,  3.13154023],
           [ 2.28147351,  2.39398397,  1.49245281],
           [ 3.16196033,  3.72534409,  3.24555934],
           [ 4.90318748,  2.02974457,  2.49846847]])
    >>> st.coords_frac
    array([[ 0.22033082,  0.90566621,  0.11533691],
           [ 0.03617668,  0.68243941,  0.98625597],
           [ 0.59727965,  0.57441444,  0.47241757],
           [ 0.57943468,  0.84217708,  0.62630805],
           [ 0.4562947 ,  0.47879679,  0.29849056],
           [ 0.63239207,  0.74506882,  0.64911187],
           [ 0.9806375 ,  0.40594891,  0.49969369]])
    >>> st.cryst_const
    array([  5.,   5.,   5.,  90.,  90.,  90.])
    >>> st.cell
    array([[  5.00000000e+00,   0.00000000e+00,   0.00000000e+00],
           [  3.06161700e-16,   5.00000000e+00,   0.00000000e+00],
           [  3.06161700e-16,   3.06161700e-16,   5.00000000e+00]])
    >>> st.get_ase_atoms()
    Atoms(symbols='NAl3N2Al', positions=..., cell=[[2.64588604295, 0.0, 0.0],
    [1.6201379367036871e-16, 2.64588604295, 0.0], [1.6201379367036871e-16,
    1.6201379367036871e-16, 2.64588604295]], pbc=[True, True, True])
    """
    
    @decorators.crys_add_doc
    def __init__(self, 
                 coords=None, 
                 coords_frac=None, 
                 symbols=None, 
                 cell=None,
                 cryst_const=None,
                 forces=None,
                 to_bohr=1.0,
                 ):
        """
        args:
        -----
        coords : 2d array (natoms, 3)
            Cartesian coords in Bohr. See also `to_bohr`.
            Optional if `coords_frac` given.
        coords_frac : 2d array (natoms, 3)
            Fractional coords w.r.t. `cell`.
            Optional if `coords` given.
        symbols : sequence of strings (natoms,)
            atom symbols
        %(cell_doc)s 
            Vectors are rows.
            Unit is Bohr. See also `to_bohr`.
            Optional if `cryst_const` given.
        %(cryst_const_doc)s
            cryst_const[:3] = [a,b,c] in Bohr. See also `to_bohr`.
            Optional if `cell` given.
        forces : optional, 2d array (natoms, 3) with forces [Ha/Bohr]            
        to_bohr : conversion factor to Bohr for all lengths

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
        self.coords = coords * to_bohr if coords is not None else coords
        self.coords_frac = coords_frac
        self.symbols = symbols
        self.cell = cell * to_bohr if cell is not None else cell
        if cryst_const is not None:
            cc = cryst_const.copy()
            cc[:3] *= to_bohr
            self.cryst_const = cc
        else:            
            self.cryst_const = cryst_const
        
        # from input
        input_attr_lst = [\
            'coords',
            'coords_frac',
            'symbols',
            'cell',
            'cryst_const',
            ]
        # derived, all attrs *except" ase_atoms
        self.derived_attr_lst = [\
            'natoms',
            'symbols_unique',
            'order',
            'typat',
            'znucl',
            'ntypat',
            'nspecies',
            ]
        self.attr_lst = input_attr_lst + self.derived_attr_lst            
        for attr in self.derived_attr_lst:
            setattr(self, attr, None)
        
        for attr in self.attr_lst:
            self.assert_get_attr(attr)

    def get_coords(self):
        if not self.is_set_attr('coords'):
            req = ['cell', 'coords_frac']
            self.assert_get_attrs(req)
            return crys.coord_trans(coords=self.coords_frac,
                                    old=self.cell,
                                    new=np.identity(3))
        else:
            return self.coords
    
    def get_coords_frac(self):
        if not self.is_set_attr('coords_frac'):
            req = ['cell', 'coords']
            self.assert_get_attrs(req)
            return crys.coord_trans(coords=self.coords,
                                    old=np.identity(3),
                                    new=self.cell)
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
        self.assert_get_attr('symbols') 
        return len(self.symbols)
    
    def get_ase_atoms(self):
        """Return ASE Atoms object. Obviously, you must have ASE installed. We
        use scaled_positions=self.coords_frac, so only self.cell must be in the
        correct unit, see `to_bohr` in __init__().
        """
        req = ['coords_frac', 'cell', 'symbols']
        self.assert_get_attrs(req)
        # We don't wanna make ase a dependency. Import only when needed.
        from ase import Atoms
        return Atoms(symbols=self.symbols,
                     scaled_positions=self.coords_frac,
                     cell=self.cell * Bohr / Angstrom,
                     pbc=[1,1,1])

    def get_symbols_unique(self):
        self.assert_get_attr('symbols')     
        return np.unique(self.symbols).tolist()

    def get_order(self):
        self.assert_get_attr('symbols_unique')
        return dict([(sym, num+1) for num, sym in
                     enumerate(self.symbols_unique)])

    def get_typat(self):
        self.assert_get_attrs(['symbols', 'order'])     
        return [self.order[ss] for ss in self.symbols]
    
    def get_znucl(self):
        self.assert_get_attr('symbols_unique')
        return [periodic_table.pt[sym]['number'] for sym in self.symbols_unique]

    def get_ntypat(self):
        self.assert_get_attr('order')
        return len(self.order.keys())
    
    def get_nspecies(self):
        self.assert_get_attrs(['order', 'typat'])
        return dict([(sym, self.typat.count(idx)) for sym, idx in 
                     self.order.iteritems()])
    
    def verify(self):
        assert self.get_natoms() == self.get_coords.shape[0]


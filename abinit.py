# abinit.py
#
# Tools for building Abinit input files.

import numpy as np
from pwtools import periodic_table
from pwtools.common import seq2str

class AbinitInput(object):
    """Class which builds some useful input variables based on `symbols`
    alone.
    
    example:
    --------
    >>> aa = AbinitInput(['Al']*5 + ['N']*5 + ['O']*2 + ['N'])
    >>> aa.print_input()
    natom 13
    ntypat 3
    typat 1 1 1 1 1 2 2 2 2 2 3 3 2
    znucl 13 7 8
    """
    def __init__(self, symbols):
        # XXX order of getters is important, use FelxibleGetters as base class
        # if you extend this
         
        # ['N', 'Al', 'Al', 'Al', 'N', 'N', 'Al']]
        self.symbols = symbols
        # ['Al', 'N']
        self.symbols_unique = self.get_symbols_unique()
        # {'Al': 1, 'N': 2}
        self.order = self.get_order()
        # [2,1,1,1,2,2,1]
        self.typat = self.get_typat()
        # [13,7]
        self.znucl = self.get_znucl()
        # 2
        self.ntypat = self.get_ntypat()

    def get_symbols_unique(self):
        return np.unique(self.symbols).tolist()

    def get_order(self):
        return dict([(sym, num+1) for num, sym in
                     enumerate(self.symbols_unique)])

    def get_typat(self):
        """
        returns:
        --------
        list of ints

        example:
        --------
        >>> get_typat(['Al']*3 + ['N']*2, {'Al':1, 'N':2})
        [1,1,1,2,2]
        """
        return [self.order[ss] for ss in self.symbols]
    
    def get_znucl(self):
        """
        returns:
        --------
        list of ints

        example:
        --------
        >>> get_znucl({'Al':1, 'N':2})
        [13,7]
        """
        return [periodic_table.pt[sym]['number'] for sym in self.symbols_unique]

    def get_ntypat(self):
        return len(self.order.keys())
    
    def print_input(self):
        print "natom %i" %len(self.symbols)    
        print "ntypat %i" %self.ntypat
        print "typat %s" %seq2str(self.typat)
        print "znucl %s" %seq2str(self.znucl)


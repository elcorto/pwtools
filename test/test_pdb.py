import numpy as np
from pwtools.parse import PDBFile
from pwtools import common

def test_pdb():
    struct = PDBFile('files/pdb_struct.pdb', 
                     units={'length':    1.0}).get_struct()
     
    assert struct.cell is not None
    assert struct.cryst_const is not None
    assert struct.symbols is not None
    assert struct.coords is not None
    assert struct.coords_frac is not None    

    assert struct.symbols == ['C' ,'O' ,'O' ,'Na','H' ,'H' ,'O' ,'O']
    coords = np.array(\
        [[5.759,   5.581,   5.339], 
         [5.759,   6.951,   5.339],
         [6.980,   4.876,   5.339],
         [7.406,   6.575,   5.339],
         [5.701,   2.442,   7.733],
         [5.908,   0.887,   7.280],
         [6.008,   1.840,   6.996],
         [2.880,   7.979,   2.600]])
    cryst_const = np.array([10.678, 10.678,10.678,90.00,90.00, 90.00])   
    assert np.allclose(coords, struct.coords)             
    assert np.allclose(cryst_const, struct.cryst_const)             

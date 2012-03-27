def test():
    from pwtools.parse import PDBFile
    from pwtools import common

    struct = PDBFile('files/pdb_struct.pdb', 
                     units={'length':    1.0}).get_struct()
     
    assert struct.cell is not None
    assert struct.cryst_const is not None
    assert struct.symbols is not None
    assert struct.coords is not None
    assert struct.coords_frac is not None    

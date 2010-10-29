def test():
    from pwtools.parse import PDBFile
    from pwtools import common

    p = PDBFile('files/pdb_struct.pdb')
    p.parse()

    common.print_dct(p.__dict__)

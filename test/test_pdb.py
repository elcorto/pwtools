from pwtools.lib.crys import PDBFile
from pwtools import common

p = PDBFile('pdb_struct.pdb')

common.print_dct(p.__dict__)

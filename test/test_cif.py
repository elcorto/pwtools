from pwtools.parse import CifFile
from pwtools import common

c = CifFile('files/cif_struct.cif')
c.parse()

# Note the conversion A -> Bohr in c.celldm[0] == alat.
common.print_dct(c.__dict__)

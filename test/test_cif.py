from pwtools.lib.crys import CifFile
from pwtools import common

c = CifFile('cif_struct.cif')

# Note the conversion A -> Bohr in c.celldm[0] == alat.
common.print_dct(c.__dict__)

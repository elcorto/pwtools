from pwtools.lib.parse import PwInputFile
from pwtools import common

filename = 'files/pw.scf.in.2'

c = PwInputFile(filename=filename)
c.parse()
common.print_dct(c.__dict__)

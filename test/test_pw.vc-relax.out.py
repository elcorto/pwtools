from pwtools.parse import PwOutputFile
from pwtools import common

filename = 'files/pw.vc-relax.out'
infile = 'files/pw.vc-relax.in'

common.system('gunzip %s.gz' %filename)
c = PwOutputFile(filename=filename, infile=infile)
c.parse()
common.print_dct(c.__dict__)
common.system('gzip %s' %filename)

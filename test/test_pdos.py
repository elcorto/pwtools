from pwtools.lib.parse import PwOutputFile
from pwtools import common
from pwtools import pydos as pd

filename = 'files/pw.md.out'
infile = 'files/pw.md.in'
dumpfile = '/tmp/pw.md.pk'

common.system('gunzip %s.gz' %filename)
c = PwOutputFile(filename=filename, infile=infile)
c.parse()

##c.dump(dumpfile)
##c2 = PwOutputFile()
##c2.load(dumpfile)
c2 = c

V = pd.velocity(c2.coords)
m = c2.infile.massvec
f, d = pd.direct_pdos(V, m=m)
##f, d = pd.vacf_pdos(V, m=m, mirr=True)

from matplotlib import pyplot as plt
plt.plot(f,d)
plt.show()

common.system('gzip %s' %filename)

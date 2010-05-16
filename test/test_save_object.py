# Data persistence. Parse some data into a PwOutputFile object and save the
# whole object in binary to disk using the dump() method, which actually uses
# cPickle. 

import numpy as np

from pwtools.lib.parse import PwOutputFile
from pwtools import common
from pwtools import pydos as pd

def check(is_true):
    if is_true:
        print "... ok"
    else:        
        print "... FAILED!"
        

filename = 'files/pw.md.out'
infile = 'files/pw.md.in'
dumpfile = '/tmp/pw.md.pk'

common.system('gunzip %s.gz' %filename)
c = PwOutputFile(filename=filename, infile=infile)
print ">>> parsing ..."
c.parse()
print ">>> ... done"

print ">>> saving %s ..." %dumpfile
c.dump(dumpfile)
print ">>> ... done"

print ">>> loading ..."
c2 = PwOutputFile()
c2.load(dumpfile)
print ">>> ... done"

print ">>> checking equalness of attrs in loaded object ..."
known_fails = {'file': 'closed/uninitialized file',
               'infile': 'same object, just new memory address'}
arr_t = type(np.array([1]))
for attr in c.__dict__.iterkeys():
    print attr
    c_val = getattr(c, attr)
    c2_val = getattr(c2, attr)
    if type(c_val) == arr_t:
        check((c_val == c2_val).all())
    else:
        check(c_val == c2_val)
    for name, string in known_fails.iteritems():
        if name == attr:
            print "    KNOWN FAIL: %s: %s" %(name, string)
common.system('gzip %s' %filename)

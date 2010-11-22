# Data persistence. Parse some data into a PwOutputFile object and save the
# whole object in binary to disk using the dump() method, which actually uses
# cPickle. 

def test():
    import numpy as np
    from pwtools.parse import PwOutputFile
    from pwtools import common
    from pwtools import pydos as pd

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
    known_fails = {'fd': 'closed/uninitialized file',
                   'infile': 'same object, just new memory address'}
    arr_t = type(np.array([1]))
    for attr in c.__dict__.iterkeys():
        c_val = getattr(c, attr)
        c2_val = getattr(c2, attr)
        dotest = True
        for name, string in known_fails.iteritems():
            if name == attr:
                print "%s: KNOWNFAIL: %s: %s" %(name, string, attr)
                dotest = False
        if dotest:                
            if type(c_val) == arr_t:
                assert (c_val == c2_val).all(), "fail: %s: %s, %s" \
                                                %(name, c_val, c2_val)
            else:
                assert c_val == c2_val, "fail: %s: %s, %s" \
                                        %(name, c_val, c2_val)
    common.system('gzip %s' %filename)

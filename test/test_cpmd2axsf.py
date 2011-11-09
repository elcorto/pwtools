import os.path
from pwtools import common
from testenv import testdir


def run(filename):
    # filename = 'files/cpmd/md_bo/cpmd.bo.out'
    # dr       = 'files/cpmd/md_bo'
    # basedr   = 'files/cpmd'
    # archive  = 'files/cpmd/md_bo.tgz'
    bar = '='*78
    print bar
    print "@@testing: %s" %filename
    print bar
    dr = os.path.dirname(filename)
    basedr = os.path.dirname(dr)
    # Use backtick() instead of system() b/c apparently nosetests catches only
    # stdout from Python (print ...). system()'s stdout is from the shell and
    # that isn't catched. 
    print common.backtick('tar -C %s -xzf %s.tgz' %(basedr, dr))
    print common.backtick('../bin/cpmd2axsf.py %s %s.axsf' %(filename, filename))
    print common.backtick('../bin/cpmd2axsf.py -r 2,2,2 -f 0.1 -t 5: %s %s.axsf' %(filename, filename))
    print common.backtick('rm -r %s' %dr)

def test():
    run(filename='files/cpmd/md_bo_odiis/cpmd.bo.out')
    run(filename='files/cpmd/md_bo_odiis_npt/cpmd.out')        
    run(filename='files/cpmd/md_bo_lanczos/cpmd.bo.out')               

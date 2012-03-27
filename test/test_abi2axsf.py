import shutil, os
from testenv import testdir
from pwtools.common import backtick

def test_abinit_to_axsf():
    here = os.path.abspath(os.curdir)
    workdir = os.path.join(testdir, 'test_abi2axsf')
    shutil.copytree('files/abinit_md', workdir)
    print backtick('gunzip %s/*.gz' %workdir)
    for ionmov, typ in {2: 'md', 8: 'md', 13: 'vcmd'}.iteritems():
        cmd = "cd %s; \
               for fn in *ionmov%s*.out; do \
                   echo $fn; \
                   %s/../bin/abi2axsf.py -t %s $fn ${fn}.axsf; \
               done "%(workdir, ionmov, here, typ)
        print backtick(cmd)

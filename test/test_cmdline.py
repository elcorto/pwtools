#!/usr/bin/env python

# Simple test of common functionallity. We don't jet test correctness of
# results (i.e. compare numbers).

import subprocess as sp
import os
import shutil
import textwrap


def system(call):
    print("[system] calling: %s" %call)
    proc = sp.Popen(call, shell=True)    
    os.waitpid(proc.pid, 0)

class Test(object):
    def __init__(self, call, idx, copy_from=None):
        self.call = call
        self.idx = idx
        self.copy_from = copy_from
    
    def setup(self, basedir, exe, infile, outfile):
        self.basedir = basedir 
        self.dst = os.path.join(self.basedir, str(self.idx))
        # make cmd line complete
        self.call = '%s -i %s -o %s ' %(exe, infile, outfile) + self.call
        self.call += ' -x %s' %self.dst
        
    def execute(self):                    
        if self.copy_from is not None:
            src = os.path.join(self.basedir, str(self.copy_from))
            print "[Test]: copy: %s -> %s" %(src, self.dst)
            shutil.copytree(src, self.dst)
        system(self.call)

if __name__ == '__main__':
    
    print('\n' + "*"*78)
    print(textwrap.dedent(
    """\
    Testing cmd line. There should some text output but NO ERROR MESSAGES. If
    in doubt, run `python %s 2>&1 | egrep -i 'error|warn'`\
        """ %__file__))
    print("*"*78)

    
    tests = []
    #-------------------------------------------------------------------------
    # define tests here
    #-------------------------------------------------------------------------
    # parse, write bin
    idx = 0
    tests.append(Test("-p", idx))
    
    # read written data, calculate dos
    idx = 1
    tests.append(Test("-d -m -M", idx, copy_from=0))
    
    # read written data, calculate dos direct, no mirroring
    idx = 2
    tests.append(Test("-d -m 0 -M -t 'direct'", idx, copy_from=0))
                 
    # new dir, parse, write txt
    idx = 3
    tests.append(Test("-p -f txt", idx))

    # read txt, calculate dos
    idx = 4
    tests.append(Test("-d -f txt", idx, copy_from=3))
    
    # new dir, parse, write bin, dos in one run
    idx = 5
    tests.append(Test("-p -d -f txt", idx))
    #-------------------------------------------------------------------------
    
    # exec tests
    infile = "files/pw.md.in"
    outfile = "files/pw.md.out"
    outdir = "/tmp/test_pdos"
    exe = '../lib/pydos.py'
    
    # We do no longer support reading .gz files directly.
    system('gunzip ' + outfile + '.gz')

    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    for test in tests:
        print('\n'+ '+'*78)
        test.setup(outdir, exe, infile, outfile)
        test.execute()
    
    system('gzip ' + outfile)

##    print('\n' + "*"*78)
##    print("testing import of pwtools package")
##    print("*"*78)
##    system('cd $HOME && python -c "import pwtools"')

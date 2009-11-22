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

if __name__ == '__main__':
    
    print('\n' + "*"*78)
    print(textwrap.dedent(
    """\
    Testing cmd line. There should some text output but NO ERROR MESSAGES. If
    in doubt, run `./test.py 2>&1 | egrep -i 'error|warn'`\
        """))
    print("*"*78)

    infile = "AlN.md.in"
    outfile = "AlN.md.out"
    outdir = "/tmp/test_pdos"

    exe = ' ../lib/pydos.py'
    
    # We do no longer support reading .gz files directly.
    system('gunzip ' + outfile + '.gz')

    calls =[]
    # parse, write bin
    idx = 0
    calls.append("%s -i %s -o %s -p -x %s" %(exe, infile, outfile, \
        os.path.join(outdir, str(idx))))
    
    # read written data, calculate dos
    calls.append("%s -i %s -o %s -d -m -M -x %s" %(exe, infile, outfile,\
        os.path.join(outdir, str(idx))))
    
    # read written data, calculate dos direct, no mirroring
    calls.append("%s -i %s -o %s -d -m 0 -M -t 'direct' -x %s" %(exe, infile,\
        outfile, os.path.join(outdir, str(idx))))
    
    # new dir, parse, write txt
    idx += 1
    calls.append("%s -i %s -o %s -p -f txt -x %s" %(exe, infile, outfile,\
        os.path.join(outdir, str(idx))))
    
    # read txt, calculate dos
    calls.append("%s -i %s -o %s -d -f txt -x %s" %(exe, infile, outfile,\
        os.path.join(outdir, str(idx))))
    
    # new dir, parse, write bin, dos in one run
    idx += 1
    calls.append("%s -i %s -o %s -p -d -f txt -x %s" %(exe, infile, outfile,\
            os.path.join(outdir, str(idx))))
    
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    
    for call in calls:
        print('\n'+ '+'*78)
        system(call)
    
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    
    system('gzip ' + outfile)

    #------------------------------------------------------------------------

##    print('\n' + "*"*78)
##    print("testing import of pwtools package")
##    print("*"*78)
##    system('cd $HOME && python -c "import pwtools"')

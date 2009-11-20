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
    outfile_gz = outfile + '.gz'
    outdir = "/tmp/test_pdos"

    delete_cmd = 'rm -rf %s/* &&' %outdir
    exe = ' ../lib/pydos.py'

    calls =[]
    # parse, write bin
    calls.append("%s -i %s -o %s -x %s -p" %(exe, infile, outfile_gz, outdir) )
    # read written data, calculate dos
    calls.append("%s -i %s -o %s -x %s -d -m -M" %(exe, infile, outfile_gz, outdir))
    # read written data, calculate dos direct, no mirroring
    calls.append("%s -i %s -o %s -x %s -d -m 0 -M -t 'direct'" %(exe, infile, outfile_gz, outdir))
    # delete, parse, write txt
    calls.append("%s %s -i %s -o %s -x %s -p -f txt" %(delete_cmd, exe, infile, outfile_gz, outdir))
    # read txt, calculate dos
    calls.append("%s -i %s -o %s -x %s -d -f txt" %(exe, infile, outfile_gz, outdir))
    # delete, parse, write bin, dos in one run
    calls.append("%s %s -i %s -o %s -x %s -p -d -f txt" %(delete_cmd, exe, infile, outfile_gz, outdir))
    
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    for call in calls:
        print('\n'+ '+'*78)
        system(call)
##    if os.path.exists(outdir):
##        shutil.rmtree(outdir)
    
    #------------------------------------------------------------------------

    print('\n' + "*"*78)
    print("testing import of pwtools package")
    print("*"*78)
    system('cd $HOME && python -c "import pwtools"')

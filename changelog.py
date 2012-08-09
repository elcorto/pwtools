#!/usr/bin/python

import sys, re, textwrap
from pwtools import common

def usage():
    print textwrap.dedent("""
    Call "hg log" and make a changelog separated by pattern in the
    commit message (API, ENH, BUG, ...).

    usage:
    ------
    ./changelog.py [-r <start>:<end>]

    examples
    --------
    ./changelog.py
    ./changelog.py -r 0.30.1:tip

    notes:
    ------
    We have the convention to format commit messages like that:

    ENH: foo bar
         baz
    BUG: foo bar
         baz
    API: foo bar
         baz
    INT: foo bar
         baz

    where ENH = enhancement, BUG = bug fix, API = API change, INT = internal
    refactoring. If a commit message doesn't fit into that pattern, it will be
    ignored. If you want to saerch the history by yourself, then use  "hg log"
    directly.
    """)

dbar="==============================================================================="
argv = sys.argv[1:]

if '-h' in argv or '--help' in argv:
    usage()
    sys.exit()

if argv == []:
    args = "-r 0:tip"
else:    
    args = ' '.join(sys.argv[1:])

prefix_lst = ['ENH', 'BUG', 'API', 'INT']
rex = '^(' + '|'.join(prefix_lst) + ').*'
rex = re.compile(rex)
cmd = r'hg log -v %s --template "{desc}\n"' %args
lines = common.backtick(cmd).splitlines()
for prefix in prefix_lst:
    print dbar
    go = False
    for line in lines:
        if line.startswith(prefix):
            print line
            go = True
        else:
            if go and (line.startswith(' ') and (rex.match(line) is None)):
                print line
            else:
                go = False
                continue

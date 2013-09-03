#!/usr/bin/python

import sys, re, textwrap
from pwtools import common

def usage():
    print textwrap.dedent("""
    Call "hg log" and make a changelog separated by pattern in the
    commit message (API, ENH, BUG, ...).

    usage:
    ------
    ./changelog.py [<hg options>]

    examples
    --------
    ./changelog.py
    ./changelog.py -r 0.30.1:tip

    notes:
    ------
    We have the convention to format commit messages like that:

    KIND: foo bar
          baz
    KIND: lala 1 2           

    where KIND is one of:

        API = API change: code will not run w/o change (e.g. function renamed
              or removed) 
        BEH = behavior change (e.g. different default values), but no API
              change, code will run unchanged but might give silghtly
              different results
        ENH = enhancement, addition of new features 
        BUG = bug fix 
        INT = internal refactoring w/o BEH or API change

    If a commit message doesn't fit into that pattern, it will be
    ignored. If you want to saerch the history by yourself, then use  "hg log"
    directly.
    """)

if __name__ == '__main__':
    dbar="="*79
    argv = sys.argv[1:]
    if '-h' in argv or '--help' in argv:
        usage()
        sys.exit()
    if argv == []:
        args = "-r 0:tip"
    else:    
        args = ' '.join(sys.argv[1:])
    prefix_lst = ['ENH', 'BUG', 'API', 'BEH', 'INT']
    rex = re.compile('^(' + '|'.join(prefix_lst) + ').*')
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

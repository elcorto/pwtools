# Test the class parse.Grep .

import re
import os
from pwtools.parse import Grep
from pwtools.regex import float_re

def test():

    dir = 'files'
    basename = 'togrep.txt'
    filename = os.path.join(dir, basename) 

    # scalar value, grepfile(), default: 
    #   func=re.search 
    #   handle=lambda m: m.group(1)
    glist = [Grep(regex=r'foo\s+=\s+(' + float_re + ')', 
                  func=re.search, 
                  handle=lambda m: m.group(1)),
             Grep(r'foo\s+=\s+(' + float_re + ')')]
    for grep in glist:
        assert grep.grepfile(filename) == '1.23'

    # scalar value, grepdir()
    grep = Grep(regex=r'foo\s+=\s+(' + float_re + ')', basename=basename)
    assert grep.grepdir(dir) == '1.23'

    # array values
    tgt = map(str, [1,2,3])
    rex = r'bar\s*=\s*(' + float_re + ')'
    assert re.findall(rex, open(filename).read()) == tgt
    assert Grep(regex=rex, func=re.findall).grepfile(filename) == tgt
    assert Grep(regex=rex, 
                func=re.findall, 
                basename=basename).grepdir(dir) == tgt

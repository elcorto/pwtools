# Test the class parse.Grep . We only use re.search. In some examples, we do
# not compile the regex for code simplicity, but in general, you should do
# that.

def test():
    import re
    import os
    from pwtools.parse import Grep
    from pwtools.regex import float_re

    dir = 'files'
    fn = 'togrep.txt'
    filename = os.path.join(dir, fn) 

    # scalar value, grepfile()
    glist = [Grep(regex=r'foo\s+=\s+(' + float_re + ')',
                 func=re.search,                 # default
                 handle=lambda m: m.group(1),    # default
                 ),
             Grep(regex=r'foo\s+=\s+(' + float_re + ')',
                 ),
             Grep(regex=r'foofoo\s*=\s*(' + float_re + ').*',
                  ),
            ]
    for key, grep in enumerate(glist):
        ret = grep.grepfile(filename)
        ##print("%i: %s; rex=%s" %(key, ret, grep.regex))
        assert ret == '1.23'


    # scalar value, grepdir()
    grep = Grep(regex=r'foo\s+=\s+(' + float_re + ')',
                basename=fn,
                )
    assert grep.grepdir(dir) == '1.23'


    # array values
    tgt = map(str, [1,2,3])
    rex = re.compile(r'bar\s*=\s*(' + float_re + ')')
    assert re.findall(rex, open(filename).read()) == tgt
    assert Grep(regex=rex, 
                func=re.findall, 
                handle=lambda x: x).grepfile(filename) == tgt


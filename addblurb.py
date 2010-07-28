#!/usr/bin/env python

# Copyright (c) 2008, Steve Schmerler <mefx@gmx.net>.
#   
# Add a license blurb (or on fact any kind of text) to the top of each file on
# the cmd line. Leave vim modelines and bang stuff (i.e. #!/bin/sh) alone,
# i.e. try to add the text *after* these. We also try to fiddle out the comment
# char and prefix the blurb with it.
#
# notes:
#   We use optparse. With Python 2.3, the special "%default" variable is not
#   printed in the help, but the parsing works.
# 
# tested: Python 2.5.2


import re
import sys
import shutil
import os.path as osp
import optparse

# regex matching a vim modeline or stuff like '#!/bin/sh'
#
# '(?P<com>.){1}' tries to match the comment char. Currently, we only match
# *one* char, i.e. C++ style '//' doesn't work. Use `-c '//'` in this case.
# '?P<com>' is a Python regex extension: it adds the name 'com' to the matched
# group. Very cool that.
REX = re.compile(r'^[ ]*(?P<com>.){1}.*[ ]*(((vi|vim|ex):)|(!/))')

# Text to add. Should start with a newline.
TEXT = \
"""

Copyright (c) 2008, Steve Schmerler <mefx@gmx.net>.
The pydos package. 
"""

def add_blurb(lines, nlines, rex=REX, text=TEXT, com=None, fcom=None):
    maxidx = 0
    comment = None
    for i in xrange(min(nlines, len(lines))):
        m = rex.match(lines[i])
        if m is not None:
            print "match on line: '%s'" %(lines[i].strip())
            comment = m.group('com')
            maxidx = i
    # override any possibly found comment
    if fcom is not None:            
        comment = fcom
        print("using forced comment char: '%s'" %comment)
    # comment found in file        
    elif comment is not None:        
        print("using found comment char: '%s'" %comment)
    # nothing found in file        
    else:                
        if com is not None:
            comment = com
            print("using suggested comment char: '%s'" %comment)
        else:
            raise StandardError("no comment sign available")
    # Split file at the last possible position and insert `text` prefixed by
    # `comment`.           
    new_lines = lines[:(maxidx+1)] + [text.replace('\n', '\n%s ' %comment) + '\n'] + \
        lines[(maxidx+1):]
    return new_lines

if __name__ == '__main__':
    
    usage = 'usage: %prog [options] <filenames>'
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-s', '--simulate', 
                      action='store_true', 
                      default=False,
                      help="don't do anything, just print top NLINES with added text")
    parser.add_option('-c', '--comment', 
                      default='#',
                      help="""suggest comment chars, if another one if found in 
                      the file, that one is used instead [%default]""")
    parser.add_option('-f', '--fcomment', 
                      default=None,
                      help="""force comment chars [%default]""")
    parser.add_option('-n', '--nlines', 
                      default=10,
                      type='int',
                      help="scan NLINES lines in each file from the top [%default]")

    (opts, args) = parser.parse_args()

    for fn in args:
        print("file: %s" %fn)
        dst = osp.join('/tmp', fn + '.bak')
        print("writing: '%s'" %dst)
        shutil.copy(fn, dst)
        new_lines = add_blurb(open(fn).readlines(), opts.nlines,
            com=opts.comment, fcom=opts.fcomment)
        if opts.simulate:
            ss = '----------------'
            print(ss)
            sys.stdout.writelines(new_lines[:(opts.nlines+1)])
            print(ss)
        else:            
            fh = open(fn, 'w')
            fh.writelines(new_lines)
            fh.close()

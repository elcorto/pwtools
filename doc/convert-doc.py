# Run this from iside pwtools/doc/. This transforms all doc strings to numpy
# format. Only need to be node once, but we keed it b/c the regular expressions
# are so nice :)

from pwtools import common
import re

files = common.backtick('find ../ -name "*.py"').strip().split()
print files

dct = {'args':      'Parameters',
       'example':   'Examples',
       'examples':  'Examples',
       'note':      'Notes',
       'notes':     'Notes',
       'see also':  'See Also',
       'methods':   'Methods',
       'returns':   'Returns',
       'ref':       'References',
       'refs':      'References',
       }

for fn in files:
    print "="*10 + fn + "="*10
    txt = common.file_read(fn)
    for key, val in dct.iteritems():
        rex = r'^([#!\s]*)%s:\s*\n([#!\s]*)[-]+' %key 
        repl = r'\1%s\n\2%s' %(val, '-'*len(val))
        txt = re.sub(rex, repl, txt, flags=re.M)
    common.backup(fn, prefix='.bak')
    common.file_write(fn, txt)

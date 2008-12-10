#-----------------------------------------------------------------------------
# Some timing:
# 
# ---------------------------------------
from debug import Debug
import re
DBG = Debug()
DBG.t('parse')
atomic_species = ['Si', 'O', 'Al', 'N']
fh = open('../datafiles/SiAlON.example_md.out.small')
outfh = open('out.python', 'w')
# '(Si|Al|O|N)' + ...
pat = r'(%s)' %r'|'.join(atomic_species) + r'(([ ]+-*[0-9]+\.*[0-9]*){3})'
rex = re.compile(pat)
for line in fh:
    line = line.strip()
    match = rex.search(line)
    if match is not None:
        print >>outfh, match.group(1) + match.group(2)
fh.close()     
outfh.close()     
DBG.pt('parse')
# ---------------------------------------
#
#
# grep:
#   $ grep ... <file> | grep -o ...
#   Don't use grep. Slow.  
# 
# perl:
#   $ time perl -ne 'print 
#   "$1 $2\n" if /(Al|O|Si|N)(([ ]+-*[0-9]+\.*[0-9]*){3})/' <
#   SiAlON.example_md.out.small > out.perl
#
#   real    0m2.381s
#   user    0m2.288s
#   sys     0m0.024s
#
# awk:
#   $ time awk '/^(Al|O|Si|N)/ 
#   {printf "%3s %10f %10f %10f\n", $1, $2, $3, $4
#   > "out.awk"}' SiAlON.example_md.out.small
#
#   real    0m1.369s
#   user    0m1.196s
#   sys     0m0.020s
#
# Python:
#   --DEBUG--: parse:  time: 0.457535982132
#
# Wohoo! Python rocks!!

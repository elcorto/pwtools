# This here is experimental stuff that't supposed
# to live in pydos.py (or wherever the parsing functions atomic_positions()
# etc. are defined.
# The idea is to have a class (PwInputFile), that automatically parses in a
# whole pw.x input file. For that to work, parsing functions for all cards (See
# INPUT_PW_CARDS_DCT) need to be written. This is already done for the most
# important ones. The namelist stuff is also already handled by
# pydos.conf_namelists() . 
#
# Note: All this is more or less a proof of convept and probably not worth the
# trouble. ATM we don't really need a full-fledged parser. We may switch to
# Abinit anyway and won't need it at all :) In the long run we'll be better off
# using template files for scripted input file generation.

#==============================================================================

# All card names and associated parsing functions that may follow the namelist 
# section in a pw.x input file. 
#
# The order the the same as in the QE docs
# (http://www.quantum-espresso.org/input-syntax/INPUT_PW.html).  Tests have
# shown that the the cards may appear in "almost" arbitrary order in an input
# file. Almost: e.g. atomic_species must appear before atomic_positions. So the
# safest policy for parsing is to assume that they can appear in arbitrary
# order.
INPUT_PW_CARDS_DCT = {\
    'atomic_species':   'atomic_species',
    'atomic_positions': 'atomic_positions',
    'k_points':         None,
    'cell_parameters':  'cell_parameters',
    'climbing_images':  None,
    'constraints':      None,
    'collective_vars':  None,
    'occupations':      None,
    }


# Quick hack. Use module level functions. Can we expose class methods as module
# level functions later?
class PwInputFile(object):
    def __init__(self, card_dct=INPUT_PW_CARDS_DCT):
        self.card_dct = card_dct 
        self.card_names = self.card_dct.keys()
    def parse(self, fn):
        fh = open(fn, 'r')
        self.namelists = conf_namelists(fh)
        
        # This doesn't work:
        # In each function `eval(self.card_dct[name])`, scan_until_pat* is
        # called but the current line of `fh` is already the "header" of the
        # table (e.g. "ATOMIC_POSITIONS"). 
        #
        # The problem is that scan_until_pat*() is not able to return fh at the
        # same position as it was when fh was passed in. They operate with the
        # assumption that fh's position is at least one line above the pattern
        # to search for.
        #
        # That's b/c in the loop over fh, fh.next() is called. That places us
        # immediately on the NEXT line. Additionally, there is no way to check
        # if we're already on the pattern before the loop b/c `fh.readline()`
        # and then `for line in fh` doesn't. work.
        # http://docs.python.org/library/stdtypes.html#file.next
        #
        # This is solved by fh.seek(0), i.e. we go thru the whole file each
        # time. This is ugly but fast enough plus we can re-use all parsing
        # functions.

##        for line in fh:
##            print "line:", line.strip()
##            for name, func in self.card_dct.iteritems():
##                if line.strip().lower().startswith(name) and func is not None:
##                    print "found name:", name
##                    self.__dict__[name] = eval(self.card_dct[name])(fh)

        # This works. We use the already-written parsing funcs which expect a
        # file object. We go thru the whole file every time. Since we know that
        # the input files are small, we could also just read them in
        # (fh.readlines()) and use this. We could then go up and down in the
        # list and use the above approach. Since the funcs already
        # accept arbitrary iterators (they use fh.next() internally but don't
        # actually know that fh is a file object), we would just have to
        # decorate each function so that they execute fh = iter(list), if
        # they are passed a list.
        
        for name, func in self.card_dct.iteritems():
            print "func:", func
            if func is not None:
                print "    func:", func
                fh.seek(0)
                # name : "header" or name of a card
                # self.card_dct[name] : string of function name to parse this card
                print "name:", name
                print "self.card_dct[name]:", self.card_dct[name]
                self.__dict__[name] = eval(self.card_dct[name])(fh)
                print self.__dict__[name]
        fh.close()



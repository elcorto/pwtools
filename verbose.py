# verbose.py
#
# Defines message printing stuff. Used in all other modules. Use the global var
# VERBOSE to turn chatty functions on/off.
#
# from pwtools import verbose, parse
# verbose.VERBOSE = True
# pp = parse.PwSCFOutputFile(...)

VERBOSE = False

def verbose(msg):
    if VERBOSE:
        print(msg)


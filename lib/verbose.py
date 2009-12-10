# verbose.py
#
# Defines message printing stuff. Used in all other modules. Use the global var
# VERBOSE to turn chatty functions on/off.

VERBOSE=True

def verbose(msg):
    if VERBOSE:
        print(msg)


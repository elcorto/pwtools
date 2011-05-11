import itertools
from numpy.random import randint
from pwtools import comb

def test():
    lo,hi = 3,7
    nn = randint(lo, hi)
    seq = randint(0, 10, nn)
    num_perms = comb.factorial(nn)
    perms1 = [list(x) for x in itertools.permutations(seq)]
    perms2 = comb.permute(seq)
    assert len(perms1) == num_perms
    assert len(perms2) == num_perms

    for xx in perms2:
        assert xx in perms1
    for xx in perms1:
        assert xx in perms2       

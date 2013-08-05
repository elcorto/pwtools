from pwtools import num

def test_round_mult():
    assert num.round_up_next_multiple(144,8) == 144
    assert num.round_up_next_multiple(145,8) == 152

from pwtools.batch import Case

def test_case():

    c = Case(a=1, b='b')
    assert c.a == 1
    assert c.b == 'b'

    class MyCase(Case):
        pass

    c = MyCase(a=1, b='b')
    assert c.a == 1
    assert c.b == 'b'

    class MyCase(Case):
        def init(self):
            self.a += 1
            self.b += 'x'

    c = MyCase(a=1, b='b')
    assert c.a == 2
    assert c.b == 'bx'


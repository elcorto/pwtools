"""Test lazy evaluation of properties."""

from pwtools.decorators import lazyprop

class Foo(object):
    def __init__(self):
        self.lazy_called = False

    @lazyprop
    def prop(self):
        self.lazy_called = True
        print("Hi there, I'm the lazy prop.")
        return 123

def test_lazy():
    foo = Foo()
    assert not foo.lazy_called

    # calling hasattr(foo, 'prop') would already define foo.prop, so we need to
    # inspect __dict__ directly
    assert 'prop' not in foo.__dict__

    # The first "foo.prop" defines foo.prop by calling the getter foo.prop =
    # foo.prop() [actually something like setattr(foo, 'prop', foo.prop())].
    # The method prop() gets overwritten by the return value 123, i.e. from now
    # on foo.prop == 123.
    assert foo.prop == 123
    assert foo.lazy_called

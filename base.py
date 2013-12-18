import os.path
from pwtools import common, num
import cPickle

# XXX Can all this be done using @property? If so, send me a patch!
class FlexibleGetters(object):
    """The most basic base class -- the mothership! 
    
    Implements a mechanism which allows to call getters in
    arbitrary order, even if they depend on each other. The mechanism also
    assured that the code in each getter is only executed once (by using checks
    with self.is_set_attr()).
    
    For each attr, there must exist a getter. We define the convention::
      
      self.foo  -> self.get_foo() 
      self.bar  -> self.get_bar()  
      self._baz -> self._get_baz() # note the underscores
      ... 
    
    self.attr_lst is an *optional* list of strings, each is the name of a data
    attribute, e.g. ['foo', 'bar', '_baz', ...].       
    Derived classes can override self.attr_lst by using self.set_attr_lst().
    
    This model is extremely powerful and allows the construction of very
    general code (see parse.py). One drawback: Beware of cyclic dependencies
    (i.e. get_bar -> get_foo -> get_bar -> ..., max. recursion limit error).
    Always test! This corner case danger is outweight by usefullness.
   
    Examples
    --------
    ::

        class MySuperParsingClass(FlexibleGetters):
            def __init__(self):
                self.set_attr_lst(['foo', 'bar', '_baz'])
                self.set_all()
            
            def set_all(self):
                "Sets self.foo, self.bar and self._baz by calling their
                getters"
                for attr in self.attr_lst:
                    self.try_set_attr(attr)
            
            # Getters call each other
            def _get_baz(self):
                return self.calc_baz()
            
            def get_bar(self):
                if self.check_set_attr('_baz'):
                    return self.calc_stuff(self._baz)**2.0
                else:
                    return None

            def get_foo(self):
                required = ['bar', '_baz']
                if self.check_set_attr_lst(required):
                    return do_stuff(self._baz, self.bar)
                else:
                    return None
    
    Setting self.attr_lst is optional. It is supposed to be used only in
    set_all(). The try_set_attr() - method works without it, too. 
    """ 
    # Notes for derived classes (long explaination):
    #
    # In this class we define a number of members (self.foo, self.bar,
    # ...) which shall all be set by the set_all() method.
    #
    # There are 3 ways of doing it:
    #
    # 1) Put all code in set_all(). 
    #    Con: One might forget to implement the setting of a member.
    # 
    # 2) Implement set_all() so that for each data member of the API, we have
    #       self.foo = self.get_foo()
    #       self.bar = self.get_bar()
    #       ...
    #    and put the code for each member in a separate getter. This is good
    #    coding style, but often data needs to be shared between getters (e.g.
    #    get_foo() needs bar, which is the result of self.bar =
    #    self.get_bar(). This means that in general the calling order
    #    of the getters is important and is different in each set_all() of each
    #    derived class.
    #    Con: One might forget to call a getter in set_all() and/or in the wrong 
    #         order.
    # 
    # 3) Implement all getters such that they can be called in arbitrary order.
    #    Then in each set_all(), one does exactly the same:
    #
    #        attr_lst = ['foo', 'bar', ...]
    #        for attr in attr_lst:
    #            self.try_set_attr(attr)
    #    
    #    This code (the "getting" of all API members) can then be moved to the
    #    *base* class's set_all() and thereby forcing all derived classes to
    #    conform to the API. 
    #
    #    If again one getter needs a return value of another getter, one has to
    #    transform
    #    
    #       def get_foo(self):
    #           return do_stuff(self.bar)
    #    to 
    #       
    #       def get_foo(self):
    #           if self.check_set_attr('bar'):
    #               return do_stuff(self.bar)
    #           else:               
    #               return None
    #
    #    If one does
    #        self.foo = self.get_foo()
    #        self.bar = self.get_bar()
    #        ....
    #    then some calls may in fact be redundant b/c e.g. get_foo() has
    #    already been called inside get_bar(). There is NO big overhead in
    #    this approach b/c in each getter we test with try_set_attr() if a
    #    needed other member is already set.
    #    
    #    This way we get a flexible and easily extensible framework to
    #    implement new parsers and modify existing ones (just implement another
    #    getter get_newmember() in each class and extend the list of API
    #    members by 'newmember').
    #
    
    def __init__(self):
        self.set_attr_lst([])
    
    def _debug_attrs(self):
        for attr in self.attr_lst:
            if getattr(self, attr) is None:
                print "%s: None" %attr
            else:                
                print "%s: ok" %attr

    def set_all(self, attr_lst=None):
        """Call getter for each attr name in `attr_lst`."""
        attr_lst = self.attr_lst if attr_lst is None else attr_lst
        for attr in attr_lst:
            self.try_set_attr(attr)
    
    def set_attr_lst(self, attr_lst):
        """Set self.attr_lst and init each attr to None."""
        self.attr_lst = attr_lst
        self.init_attr_lst()

    def init_attr_lst(self, attr_lst=None):
        """Set each self.<attr> in `attr_lst` to None."""
        lst = self.attr_lst if attr_lst is None else attr_lst
        for attr in lst:
            setattr(self, attr, None)

    def dump(self, dump_filename, mkdir=True):
        """Pickle (write to binary file) the whole object."""
        # Dumping with protocol "2" is supposed to be the fastest binary format
        # writing method. Probably, this is platform-specific.
        if mkdir:
            dr = os.path.dirname(dump_filename)
            if dr != '':
                common.makedirs(dr)
        cPickle.dump(self, open(dump_filename, 'wb'), 2)

    def load(self, dump_filename):
        """Load pickled object.
        
        Examples
        --------
        >>> # save
        >>> x = FileParser('foo.txt')
        >>> x.parse()
        >>> x.dump('foo.pk')
        >>> # load: method 1 - recommended
        >>> xx = common.cpickle_load('foo.pk')
        >>> # or 
        >>> xx = cPickle.load(open('foo.pk'))
        >>> # load: method 2, not used / tested much
        >>> xx = FileParser()
        >>> xx.load('foo.pk')
        """
        # this does not work:
        #   self = cPickle.load(...)
        self.__dict__.update(cPickle.load(open(dump_filename, 'rb')).__dict__)
    
    def is_set_attr(self, attr):
        """Check if self has the attribute self.<attr> and if it is _not_ None.

        Parameters
        ----------
        attr : str
            Attrubiute name, e.g. 'foo' for self.foo
        
        Returns
        -------
        True : `attr` is defined and not None
        False : not defined or None
        """
        if hasattr(self, attr): 
            return (getattr(self, attr) is not None)
        else:
            return False
    
    def is_set_attr_lst(self, attr_lst):
        assert common.is_seq(attr_lst), "attr_lst must be a sequence"
        for attr in attr_lst:
            if not self.is_set_attr(attr):
                return False
        return True                

    def try_set_attr(self, attr):
        """If self.<attr> does not exist or is None, then invoke an
        appropirately named getter as if this command would be executed::
        
            self.foo = self.get_foo()
            self._foo = self._get_foo()

        Parameters
        ----------
        attr : string
        """
        if not self.is_set_attr(attr):
            if attr.startswith('_'):
                get = '_get'
            else:
                get = 'get_'
            setattr(self, attr, eval('self.%s%s()' %(get, attr))) 
    
    def try_set_attr_lst(self, attr_lst):
        for attr in attr_lst:
            self.try_set_attr(attr)
    
    def check_set_attr(self, attr):
        """Run try_set_attr() and return the result of is_set_attr(), i.e. True
        or False. Most important shortcut method.
        
        Examples
        --------
        ::
            
            def get_foo(self):
                if self.check_set_attr('bar'):
                    return self.bar * 2
                else:
                    return None
        
        which is the same as ::

            def get_foo(self):
                self.try_set_attr('bar):
                if self.is_set_attr('bar'):
                    return self.bar * 2
                else:
                    return None
        
        """
        self.try_set_attr(attr)
        return self.is_set_attr(attr) 

    def check_set_attr_lst(self, attr_lst):
        for attr in attr_lst:
            self.try_set_attr(attr)
        return self.is_set_attr_lst(attr_lst)

    def assert_attr(self, attr):
        """Raise AssertionError if self.<attr> is not set (is_set_attr()
        returns False."""
        if not self.is_set_attr(attr):
            raise AssertionError("attr '%s' is not set" %attr)
    
    def assert_attr_lst(self, attr_lst):
        for attr in attr_lst:
            self.assert_attr(attr)
    
    def assert_set_attr(self, attr):
        """Same as assert_attr(), but run try_set_attr() first."""
        self.try_set_attr(attr)
        self.assert_attr(attr)

    def assert_set_attr_lst(self, attr_lst):
        for attr in attr_lst:
            self.assert_set_attr(attr)
    
    def raw_slice_get(self, attr_name, sl, axis):
        """Shortcut method:

        * call ``try_set_attr(_<attr_name>_raw)`` -> set
          ``self._<attr_name>_raw`` to None or smth else
        * if set, return ``self._<attr_name>_raw`` sliced by `sl` along `axis`,
          else return None
        """
        raw_attr_name = '_%s_raw' %attr_name
        self.try_set_attr(raw_attr_name)
        if self.is_set_attr(raw_attr_name):
            arr = getattr(self, raw_attr_name)
            ret = num.slicetake(arr, sl, axis) 
            # slicetake always returns an array, return scalar if ret =
            # array([10]) etc
            if (ret.ndim == 1) and (len(ret) == 1):
                return ret[0]
            else:
                return ret
        else:
            return None
    
    def raw_return(self, attr_name):
        """Call ``try_set_attr(_<attr_name>_raw)`` and return it if set, else
        None. This is faster but the same the same as
        ``raw_slice_get(<attr_name>, sl=slice(None), axis=0)`` or axis=1 or any
        other valid axis.
        """
        raw_attr_name = '_%s_raw' %attr_name
        self.try_set_attr(raw_attr_name)
        if self.is_set_attr(raw_attr_name):
            return getattr(self, raw_attr_name)
        else:
            return None
   
    def get_return_attr(self, attr_name):
        """Call try_set_attr() are return self.<attr_name> if set."""
        self.try_set_attr(attr_name)
        if self.is_set_attr(attr_name):
            return getattr(self, attr_name)
        else:
            return None


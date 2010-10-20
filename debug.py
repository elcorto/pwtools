from time import time

class Debug(object):
    """
    Helper Class for timimg and debugging. It's meant to be used for manually
    inspecting code. For permanent verbose/debug stuff, use the logging and
    warnings modules.

    Usage:
        DBG = Debug()
        
        # print some message 
        DBG.p('start profiling part 1')
        # set start time for tag 'outer-loop'
        DBG.t('outer-loop')
        for i ...
            <code>
            # set start time for tag 'inner-loop'
            DBG.t('inner-loop')
            for j ...
                <code>
            # use case 1: get stop time and print timing (stop - start) for tag 
            # 'inner-loop' immediately           
            DBG.pt('inner-loop')
        # use case 2: get stop time and store it
        DBG.t('outer-loop')
        <some_more_code>
        # print timing (stop - start) for tag 'outer-loop' later (maybe in some
        # summary statistic or so)
        DBG.pt('outer-loop')
        
        # it's possible to re-use tags
        DBG.p('start profiling part 2')
        DBG.t('outer-loop')
        for i ...
            <code>
            DBG.t('inner-loop')
        ....
    """
    def __init__(self, silence=False):
        self.none_ar = [None, None]
        # {'tag0': array([val0, val1]), 'tag1': array([val2, val3]), ...}
        # Every `val` can be None or a float (= a time value). `tag` is a tag
        # string like 'outer-loop'.
        self.time_ar_dict = dict()
        self.silence = silence
    
    def t(self, tag):
        """
        Assign and save a numeric value (a time value) in a storage array
        associated with `tag`.
        
        input:
            tag -- a tag (string) associated with a storage array
                                       
        Notes:
            After initialization, self.time_ar_dict[tag] == [None, None].
            
            The 1st call assings self.time_ar_dict[tag][0] = <time>. 
            The 2nd call assings self.time_ar_dict[tag][1] = <time>. 
            The 3rd call resets self.time_ar_dict[tag][1] = [None, None]
            and recursively calls t(), which then does the the same as the 1st.
            ...
        """
        # Init a new array for a new tag.
        if tag not in self.time_ar_dict.keys():
            # numpy arrays:
            #   Use array method copy(), otherwise, we would use the exact same
            #   array everytime, since 'a = numpy.array(...); b = a' only creates
            #   a *view* of `a` (like a pointer).
            # lists:
            #   Behave like numpy arrays (b = a is view of a). Must also copy:
            #   b = a[:] (use slicing).
            self.time_ar_dict[tag] = self.none_ar[:]
        
        # array is [None, None], assign the 1st time value.
        if self.time_ar_dict[tag][0] is None:
            self.time_ar_dict[tag][0] = time()
        # The second time value.            
        elif self.time_ar_dict[tag][1] is None:
            self.time_ar_dict[tag][1] = time()
        # array is [<val>, <val>], so reset to [None, None] and 
        # assign the 1st time value.
        else:            
            self.time_ar_dict[tag] = self.none_ar[:]
            self.t(tag)
 
    def pt(self, tag, msg=''):
        """
        Print self.time_ar_dict[tag][1] - self.time_ar_dict[tag][0].
        """
        if tag not in self.time_ar_dict.keys():
            raise ValueError("array for tag '%s' not jet initialized; " %tag\
                + "you have to call t() first.")
        # hmm ... array is [None, None] .. shouldn't be
        # list test: [a, a] == [a, a] -> True
        if self.none_ar == self.time_ar_dict[tag]:
            raise ValueError("time array for tag '%s' or none_ar is wrong:\n" 
                "time array: %s\n"
                "none_ar: %s\n"
                %(tag, str(self.time_ar_dict[tag]), str(self.none_ar)))
        # array is [<val>, None] (use case 1) or [<val>, <val>] (use case 2) 
        if self.time_ar_dict[tag][0] is not None:
            # [<val>, None], assign second time                
            if self.time_ar_dict[tag][1] is None:
                self.t(tag)
            self.p("%s: %s time: %s" %(tag, msg,\
                self.time_ar_dict[tag][1] - self.time_ar_dict[tag][0]))
        else:                
            raise ValueError("illegal array content for tag '%s' " %tag)
    
    def p(self, msg):
        """ Simply print `msg`. """
        if not self.silence:
            print("--DEBUG--: %s" %msg)


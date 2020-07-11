from time import time

class TagTimer:
    """
    Helper class for timimg. It's meant to be used for manually inspecting
    code.

    Examples
    --------
    .. code-block:: python

        tt = TagTimer()

        # print some message
        tt.p('start profiling part 1')
        # set start time for tag 'outer-loop'
        tt.t('outer-loop')
        for i ...
            <code>
            # set start time for tag 'inner-loop'
            tt.t('inner-loop')
            for j ...
                <code>
            # use case 1: get stop time and print timing (stop - start) for tag
            # 'inner-loop' immediately
            tt.pt('inner-loop')
        # use case 2: get stop time and store it
        tt.t('outer-loop')
        <some_more_code>
        # print timing (stop - start) for tag 'outer-loop' later (maybe in some
        # summary statistic or so)
        tt.pt('outer-loop')

        # it's possible to re-use tags
        tt.p('start profiling part 2')
        tt.t('outer-loop')
        for i ...
            <code>
            tt.t('inner-loop')
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

        Parameters
        ----------
        tag : anything hashable
            a tag (most likely a string) associated with a storage array

        Notes
        -----
        After initialization, self.time_ar_dict[tag] == [None, None].

        | The 1st call assings self.time_ar_dict[tag][0] = <time>.
        | The 2nd call assings self.time_ar_dict[tag][1] = <time>.
        | The 3rd call resets self.time_ar_dict[tag] = [None, None]
        | and recursively calls t(), which then does the the same as the 1st.
        | ...
        """
        # Init a new array for a new tag.
        if tag not in self.time_ar_dict:
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
        Print time difference since last ``t(tag)`` call for `tag`, which is
        ``self.time_ar_dict[tag][1] - self.time_ar_dict[tag][0]``.

        Parameters
        ----------
        tag : anything hashable
            a tag (most likely a string) associated with a storage array
        msg : string
            Extra string to be printed along with the time difference.
        """
        if tag not in self.time_ar_dict:
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
            print("--TagTimer--: {}".format(msg))


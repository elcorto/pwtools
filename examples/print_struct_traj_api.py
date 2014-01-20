import numpy as np
from pwtools import crys
from pwtools.test.test_trajectory import get_rand_traj, get_rand_struct
rand = np.random.rand

tr = get_rand_traj()
st = get_rand_struct()
onlytr = set.difference(set(tr.attr_lst), set(st.attr_lst))
onlyst = set.difference(set(st.attr_lst), set(tr.attr_lst))
print """
API (possible attributes in attr_lst):

Structure:
{st}

only in Trajectory:
{onlytr}

only in Structure:
{onlyst}

Attributes which are None w.r.t. the Trajectory API after the following
operation, starting with a fully populated struct or traj (all attrs not None):
""".format(st=st.attr_lst, tr=tr.attr_lst, onlytr=list(onlytr), 
           onlyst=list(onlyst))

items = [\
    ('tr', tr),
    ('tr.copy', tr.copy()),
    ('tr[0:5]', tr[0:5]),
    ('st', st),
    ('st.copy', st.copy()),
    ('tr[0]', tr[0]),
    ('mean(tr)', crys.mean(tr)),
    ('concatenate([st,st])', crys.concatenate([st,st])),
    ('concatenate([st,tr])', crys.concatenate([st,tr])),
    ('concatenate([tr,tr])', crys.concatenate([tr,tr])),
    ]
for name,obj in items:
    none_attrs = set.difference(set(tr.attr_lst),
                                crys.populated_attrs([obj]))
    typ = 'traj' if obj.is_traj else 'struct'                                          
    print "{:25} {:7} {}".format(name, typ, list(none_attrs))

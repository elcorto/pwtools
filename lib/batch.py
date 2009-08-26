import types
import re
import tempfile

class ClassBase(object):
    """Abstract base class defining some general routines for convenient access
    to public data members. Most member functions implement an interface to
    extract the names and values of all data members (NOT methods) in certain
    formats in an automatic fashion. This is planned to be used to
    automatically fill a database with informations about jobs."""
    
    def __init__(self):
        
        # List of regexes. See _get_keys().
        self._skip = [r'^_.*']

        # Mapping of python types to SQL (sqlite3) types. We shall use this for
        # putting all data in the classes in a database later ... Stay tuned.
        self._sql_types = [\
            (types.NoneType, 'NULL'),
            (types.IntType, 'INTEGER'),
            (types.LongType, 'INTEGER'),
            (types.FloatType, 'REAL'),
            (types.StringType, 'TEXT'),
            (types.UnicodeType, 'TEXT'),
            (types.BufferType, 'BLOB'), # e.g. numpy array raw data
            ]
        
        # see _get_keys()
        self._keys = None

    def _repr(self, val):
        # Avoid 
        # >>> repr('lala')
        # "'lala'"
        if isinstance(val, types.StringType):
            return val
        else:
            return repr(val)

    def _get_sql_type(self, val):
        for py_type, sql_type in self._sql_types:
            if isinstance(val, py_type):
                return sql_type
        raise StandardError("no matching type for: %s" %repr(val)) 
    
    def _match(self, key):
        """Helper for _get_keys()."""
        for pat in self._skip:
            if re.match(pat, key) is not None:
                return True
        return False
    
    def _get_keys(self):
        """Current list with names of public members.
        """
        # Skip key names matching any regex in `self._skip`. 

##        # Also skip possible derived classes.
##        return [key for key in self.__dict__.keys() if not _match(key) \
##            and not isinstance(getattr(self, key), self.__class__)]
        return [key for key in self.__dict__.iterkeys() if not self._match(key)]
    
    # get_keys() and get_values() are supposed to return key-value pairs of the
    # same order. This is accieved by acessing keys *only* thru get_keys() .
    def get_keys(self):
        """Return "static" key list if self._keys is set, or dynamic (current
        state) if not set.
        """
        if self._keys is None:
            return self._get_keys()
        else:            
            return self._keys
    
    def get_values(self):
        """List of values of public members.""" 
        return [self.__dict__[key] for key in self.get_keys()]
    
    def get_keys_str(self, delim=' '):
        """Retrun string with keys of all data members."""
        return delim.join(self.get_keys())

    def get_values_str(self, delim=' '):
        """Retrun string with all data members."""
        return delim.join(map(repr, self.get_values()))

    def get_sql_types(self):
        return map(self._get_sql_type, self.get_values())
    
    def set_internal_keys(self):
        """Set private self._keys. 
        
        If set (i.e. not None), self._keys is used to extract values out of
        self.__dict__ .
   
        If at any point in time you want to "fix" the list of keys and thus
        the output of get_keys(), get_values() etc, call this function. After
        that, self._keys is set and fixed.
        
        If 
            * self._keys is set
            * one adds new data menbers
            * one wants them to show up in get_keys(), get_values() etc,
        then use this function to update self._keys.
        """
        self._keys = self._get_keys()
    
    def set_skip(self, skip):   
        self._skip += skip
    
    
    def get_template_txt(self, txt, fn):
        """Return a text string (usually a read text file). If `fn` is a
        string, then it is assumed to be a filename and the content of the file
        is returned. Otherwise, `txt` is returned.
        
        args:
        -----
        txt : {None, str}
        fn : {None, str}
        """
        if isinstance(txt, types.StringType): 
            return txt
        elif isinstance(fn, types.StringType): 
            return com.file_read(fn)
        else:
            raise StandardError("neither 'txt' nor 'fn' are strings")
    
    def merge(self, obj):
        """Merge namespace of an arbitrary classinstance (obj.__dict__) into
        our namespace (self.__dict__)."""
        #FIXME: what hapens if we have self.foo and obj.foo? Does self.foo get
        # overwritten?
        self.__dict__.update(obj.__dict__)

#-----------------------------------------------------------------------------

class MachineBase(ClassBase):
    def __init__(self, 
                 ncpu_max=None, 
                 machine_name=None, 
                 mpirun=None,
                 jobfile_name=None, 
                 outdir=None, 
                 queue=None, 
                 queue_com=None,
                 jobfile_templ_txt=None,
                 jobfile_templ_fn=None,
                 ):
        """Base class for machine-specific stuff. 
            for jobfile : queueing system etc.
            for input file : outdir
        """
        
        #--- required ------------------------------------
        
        # int 
        self.ncpu_max = ncpu_max
        # str
        self.machine_name = machine_name
        # str
        self.mpirun = mpirun
        # str
        self.jobfile_name = jobfile_name
        # str
        self.outdir = outdir
        # str
        self.queue = queue
        # str
        self.queue_com = queue_com
        # str
        self.jobfile_templ_fn = jobfile_templ_fn

        #--- optional w/ default value -------------------

        # int (FG)
        self.nhosts = ''
        # str (FG)
        self.nodes_base = ''
        
        if (jobfile_templ_txt is not None) or (jobfile_templ_fn is not None):
            self.jobfile_templ_txt = self.get_template_txt(jobfile_templ_txt,
                                                           jobfile_templ_fn)
        
    def init_queue(self, ncpu):
        """Select queue on machine based on the number of requested cpus."""
        # str
        self.queue = None
   
    
    
#-----------------------------------------------------------------------------
    
class MachineMars(MachineBase):
    def __init__(self, **kwargs):
        MachineBase.__init__(self,
            machine_name = 'mars',
            ncpu_max = 1866,
            mpirun = 'pamrun',
            jobfile_name = 'job.lsf',
            outdir='/fastfs/schmerle',
            queue_com = 'bsub <',
            **kwargs)
    
    def init_queue(self, ncpu):        
        if ncpu <= 64:
            self.queue = 'small'
        elif ncpu <= 256:
            self.queue = 'small_long'
        elif ncpu <= 1024:            
            self.queue = 'intermediate'
        elif ncpu <= self.ncpu_max:            
            self.queue = 'large'
        else:            
            raise ValueError("no queue on %s for ncpu: %i" 
                %(self.machine_name, ncpu))

#-----------------------------------------------------------------------------
        
class MachineDeimos(MachineBase):
    def __init__(self, **kwargs):
        MachineBase.__init__(self,
            machine_name = 'deimos',
            ncpu_max = 256,
            mpirun = 'mpirun.lsf',
            jobfile_name = 'job.lsf',
            outdir='/fastfs/schmerle',
            queue_com = 'bsub -a openmpi <',
            **kwargs)
    
    def init_queue(self, ncpu):
        if ncpu <= 4:
            self.queue = 'small'
        elif ncpu <= 128:
            self.queue = 'intermediate'
        elif ncpu <= 256:
            self.queue = 'large'
        else:
            raise ValueError("no queue on %s for ncpu: %i" 
                %(self.machine_name, ncpu))

#-----------------------------------------------------------------------------

class MachineFG(MachineBase):
    def __init__(self, **kwargs):
        raise NotImplementedError('fixme: fg machine settings')
        MachineBase.__init__(self,
            machine_name = 'fg',
            queue = 'crunch',
            nodes = self.nodes_base + ",ncpus=%i" %self.ncpu, #!!!
            jobfile_name = 'job.pbs',
            outdir='/scratch/schmerler',
            queue_com = 'qsub',
            **kwargs)
        
        # Attempt to automatically select nodes on the FG cluster. This will
        # probably never work :) Either we know the load of the cluster BEFORE
        # and request only free nodes (well .. that's the job of the batch
        # system) OR we request ALL nodes and leave the selection to the batch
        # system.
        nwood = self.nodes.count(':wood')
        nquad = self.nodes.count(':quad')
        nfquad = self.nodes.count(':fquad')
        
        if self.nhosts is None:
            nh_quad = 1
            while nh*8 < self.ncpu and nquad > 0:
                nh += 1
                nquad -= 1
            while nh*8 < self.ncpu and nquad > 0:
                nh += 1
                nquad -= 1
            self.nhosts = nh                    
                    

#-----------------------------------------------------------------------------

class Calc(ClassBase):
    """Class that holds all informations about a job. 
    
    TODO:
    The most elegant way to write job- and input files is to define (as we do
    already) template files. But to get fully automatic, we should set up a
    naming convention for the placeholders, e.g. XXXNAME, and `name` must be
    the name of a data member of this class. Then, go thru an arbitrary
    template file and replace all placeholders matching a data member name.
    This avoids writing a "replacement rules dictionary" in each input file
    generation script. What's left to do is to write loops over the parameters
    to vary and create a list of instances of this class, each one representing
    a caclulation with all information.
    """
    def __init__(self,
                 ncpu = 1,
                 npool = 1,
                 dir='.',
                 jobname_base='job', 
                 machine=None,
                 infile_templ_txt=None,
                 infile_templ_fn=None,
                 # XXX
##                 value_to_vary= value_to_vary
                 ):
        
        ClassBase.__init__(self)

        # --- members from input ----------------
        self.ncpu = ncpu
        self.npool = npool
        self.dir = dir
        self.jobname_base = jobname_base
        self.machine = machine
        self.infile_templ_fn = infile_templ_fn

        if (infile_templ_txt is not None) or (infile_templ_fn is not None):
            self.infile_templ_txt = self.get_template_txt(infile_templ_txt,
                                                          infile_templ_fn)
        
        rand_suffix = tempfile.mktemp(dir='')
        if self.jobname_base != '':
            self.prefix = self.jobname_base + '-' + rand_suffix
        else:
            self.prefix = rand_suffix
        
        # Merge information about the job: calculation-specific (this class) +
        # machine-specific.
        if self.machine is not None:
            self.machine.init_queue(self.ncpu)
            self.merge(self.machine)
        
        self._skip += [r'.*_templ_txt$', r'^machine$']
        self.set_internal_keys()

#-----------------------------------------------------------------------------

class Entry(object):
    def __init__(self, name, val):
        """Class representing a key = value and the associated placeholder in
        a file (pw.x input file).
        
        args:
        -----
        name : name of a valid input file identifier, used to construct the
            placeholder
        val : it's value

        example
        -------
        >>> entry = Entry('ecutwfc', 35)
        >>> entry.ph
        XXXECUTWFC
        >>> entry.val
        35
        >>> entry.str_val
        '35'
        """
        # 'ecutwfc'
        self.name = name
        # 35
        self.val = val
        # XXXECUTWFC
        self.ph = self.get_ph()
        self.str_val = self.get_str_val()

    def get_str_val(self):
        """Return a string representation of self.val as it should appear in
        the input file when self.ph is replaced."""
        return str(self.val)

    def get_ph(self):
        return 'XXX' + self.name.upper()

#-----------------------------------------------------------------------------

class Kpoints(Entry):
    def __init__(self, name, val):
        Entry.__init__(self, name, val)
    
    def get_str_val(self):
        """
        example:
        --------
        >>> k=Kpoints('k_points', 2)
        >>> k.ph
        XXXK_POINTS
        >>> k.str_val
        '2 2 2 0 0 0'
        """
        return ' '.join([str(self.val)]*3) + ' 0 0 0'


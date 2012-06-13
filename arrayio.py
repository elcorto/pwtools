# arrayio.py
#
# File IO stuff (text and binary files).

from cStringIO import StringIO
import numpy as np

import common
from decorators import open_and_close
from common import PydosConfigParser
from verbose import verbose

# globals
HEADER_MAXLINES = 20
HEADER_COMMENT = '#'
TXT_MAXDIM = 3


@open_and_close
def _read_header_config(fh, header_maxlines=HEADER_MAXLINES, 
                        header_comment=HEADER_COMMENT):
    """Read a ini-style file from the header of a text file. Return a
    PydosConfigParser object.

    args:
    -----
    fh : file handle, readable
    header_maxlines : max lines to read down from top of the file
    header_comment : comment sign w/ which the header must be prefixed
    
    returns:
    --------
    PydosConfigParser object

    example:
    --------
    >>> !cat foo.txt
    # [array]
    # shape = 3
    # axis = -1
    1
    2
    3
    >>> _get_header_config('foo.txt')
    <pwtools.common.PydosConfigParser instance at 0x2c52320>
    """
    fn = common.get_filename(fh)
    verbose("[_read_header_config]: reading header from '%s'" %fn)
    header = ''
    for i in range(header_maxlines):
        try:
            line = fh.next().strip()
        except StopIteration:
            break
        if line.startswith(header_comment):
            header += line.replace(header_comment, '').strip() + '\n'
    # Read one more line to see if the header is bigger than header_maxlines.
    try:
        if fh.next().strip().startswith(header_comment):
            raise StandardError("header seems to be > header_maxlines (%i)"
                %header_maxlines)
    except StopIteration:
        pass
    c = PydosConfigParser()
    c.readfp(StringIO(header))
    # If header_maxlines > header size, we read beyond the header into the data. That
    # causes havoc for all functions that read fh afterwards.
    fh.seek(0)
    return c


# the open_and_close decorator cannot be used here b/c it only opens
# files in read mode, not for writing
def _write_header_config(fh, config, header_comment=HEADER_COMMENT,
                         header_maxlines=HEADER_MAXLINES):
    """Write ini-style config file from `config` prefixed with `header_comment` to
    file handle `fh`."""
    fn = common.get_filename(fh)
    verbose("[_write_header_config]: writing header to '%s'" %fn)
    # write config to dummy file
    ftmp = StringIO()
    config.write(ftmp)
    # write with comment signs to actual file
    ftmp.seek(0)
    lines = ftmp.readlines()
    common.assert_cond(len(lines) <= header_maxlines, 
                "header has more then header_maxlines (%i) lines" \
                %header_maxlines)
    for line in lines:
        fh.write(header_comment + ' ' + line)
    ftmp.close()

def writetxt(fn, arr, axis=-1, maxdim=TXT_MAXDIM):
    """Write 1d, 2d or 3d arrays to txt file. 
    
    If 3d, write as 2d chunks. Take the 2d chunks along `axis`. Write a
    commented out ini-style header in the file with infos needed by readtxt()
    to restore the right shape.
    
    args:
    -----
    fn : filename
    arr : array (max 3d)
    axis : axis along which 2d chunks are written
    maxdim : highest number of dims that `arr` is allowed to have
    """
    verbose("[writetxt] writing: %s" %fn)
    common.assert_cond(arr.ndim <= maxdim, 'no rank > %i arrays supported' %maxdim)
    fh = open(fn, 'w+')
    c = PydosConfigParser()
    sec = 'array'
    c.add_section(sec)
    c.set(sec, 'shape', common.seq2str(arr.shape))
    c.set(sec, 'axis', axis)
    _write_header_config(fh, c)
    # 1d and 2d case
    if arr.ndim < maxdim:
        np.savetxt(fh, arr)
    # 3d        
    else:
        # TODO get rid of loop?                                                 
        # write 2d arrays, one by one
        sl = [slice(None)]*arr.ndim
        for ind in range(arr.shape[axis]):
            sl[axis] = ind
            np.savetxt(fh, arr[sl])
    fh.close()


@open_and_close
def readtxt(fh, axis=None, shape=None, header_maxlines=HEADER_MAXLINES,
            header_comment=HEADER_COMMENT, maxdim=TXT_MAXDIM, **kwargs):
    """Read arrays from .txt file using np.loadtxt(). 
    
    If the file stores a 3d array as consecutive 2d arrays (e.g. output from
    molecular dynamics code) the file header (see writetxt()) is used to
    determine the shape of the original 3d array and the array is reshaped
    accordingly.
    
    If `axis` or `shape` is not None, then these are used instead and 
    the header, if existing, will be ignored. This has the potential to shoot
    yourself in the foot. Use with care.
    
    If `axis` and `shape` are None, then this function does not work with
    normal text files which have no special header. Use np.loadtxt() in this
    case.

    args:
    -----
    fh : file_like
    axis : int
    shape : tuple
    **kwargs : keyword args passed to numpy.loadtxt(), e.g. comments='@@' to
        ignore weird lines etc.

    returns:
    --------
    nd array
    """
    fn = common.get_filename(fh)
    verbose("[readtxt] reading: %s" %fn)
    verbose("[readtxt]    axis: %s" %str(axis))
    verbose("[readtxt]    shape: %s" %str(shape))
    if shape is None or axis is None:
        c = _read_header_config(fh)
        sec = 'array'
        if shape is None:
            shape = common.str2tup(c.get(sec, 'shape'))
        if axis is None:            
            axis = int(c.get(sec, 'axis'))
    ndim = len(shape)
    common.assert_cond(ndim <= maxdim, 'no rank > %i arrays supported' %maxdim)
    # axis = -1 means the last dim
    if axis == -1:
        axis = ndim - 1
    
    # handle empty files (no data, only special header or nothing at all)
    header_lines = []
    for i in range(header_maxlines):
        try:
            line = fh.next().strip()
            if not line.startswith(header_comment) and line != '':
                header_lines.append(line)
        except StopIteration:
            break
    fh.seek(0)
    if header_lines == []:
        verbose("[readtxt] WARNING: empty file: %s" %fn)
        return np.array([])
    else:
        fh.seek(0)
        read_arr = np.loadtxt(fh, **kwargs)
    
    # 1d and 2d
    if ndim <= 2:
        arr = read_arr
    # 3d        
    else:
        # example:
        #   axis = 1
        #   shape = (50, 1000, 3)
        #   shape_2d_chunk =  (50, 3)
        shape_2d_chunk = shape[:axis] + shape[(axis+1):]
        # (50, 1000, 3) : natoms = 50, nstep = 1000, 3 = x,y,z
        arr = np.empty(shape, dtype=read_arr.dtype)
        # read_arr: (50*1000, 3)
        expect_shape = (shape_2d_chunk[0]*shape[axis],) + (shape_2d_chunk[1],)
        common.assert_cond(read_arr.shape == expect_shape, 
                    "read 2d array from '%s' has not the correct "
                    "shape, got %s, expect %s" %(fn, 
                                                 str(read_arr.shape),
                                                 str(expect_shape)))
        # TODO get rid of loop?                                                 
        sl = [slice(None)]*ndim
        for ind in range(shape[axis]):
            sl[axis] = ind
            arr[sl] = read_arr[ind*shape_2d_chunk[0]:(ind+1)*shape_2d_chunk[0], :]
    verbose("[readtxt]    returning shape: %s" %str(arr.shape))
    return arr


def readarr(fn, type='bin'):
    """Read bin or txt array from file `fn`."""
    verbose("[readarr] reading: %s" %fn)
    common.assert_cond(type in ['bin', 'txt'], "`type` must be 'bin' or 'txt'")
    if type == 'bin':
        return np.load(fn)
    elif type == 'txt':
        return readtxt(fn)


def writearr(fn, arr, comment=None, info=None,
             type='bin', axis=-1):
    """Write `arr` to binary (*.npy) or text file (*.txt) `fn`. Optionally,
    write a file `fn`.info with some misc information from `comment` and
    `info`. 
    
    args:
    -----
    arr : numpy ndarrray
    fn : str, filename
    comment : string
        a comment which will be written to the .info file (must start with '#')
    info : dict of dicts 
        sections for the .info file        
        example:
            {'foo': {'rofl1': 1, 'rofl2': 2},
             'bar': {'lol1': 5, 'lol2': 7}
            }
            
            will be converted to
            [foo]
            rofl1 = 1
            rofl2 = 2
            [bar]
            lol1 = 5
            lol2 = 7
    type : str, {bin,txt)
    only type == 'txt'
        axis : axis kwarg for writetxt()
    """
    common.assert_cond(type in ['bin', 'txt'], "`type` must be 'bin' or 'txt'")
    verbose("[writearr] writing: %s" %fn)
    verbose("[writearr]     shape: %s" %repr(arr.shape))
    if type == 'txt':
        verbose("[writearr]     axis: %i" %axis)
    if type == 'bin':
        np.save(fn, arr)
    else:
        writetxt(fn, arr, axis=axis)
    # .info file
    if comment is not None or info is not None:
        f = open(fn + '.info', 'w')
        if comment is not None:
            f.write(comment + '\n')
        if info is not None:
            c = PydosConfigParser()
            c = common.add_to_config(c, info) 
            c.write(f)
        f.close()


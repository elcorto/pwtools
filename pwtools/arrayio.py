"""Array text file IO. Some tools to write and read MD-like 3D arrays."""

from io import StringIO
from configparser import ConfigParser
import numpy as np

from pwtools import common
from pwtools.decorators import open_and_close
from pwtools.verbose import verbose

# globals
HEADER_MAXLINES = 20
HEADER_COMMENT = '#'
TXT_MAXDIM = 3


@open_and_close
def _read_header_config(fh, header_maxlines=HEADER_MAXLINES, 
                        header_comment=HEADER_COMMENT):
    """Read a ini-style file from the header of a text file. Return a
    ConfigParser object.

    Parameters
    ----------
    fh : file handle, readable
    header_maxlines : max lines to read down from top of the file
    header_comment : comment sign w/ which the header must be prefixed
    
    Returns
    -------
    ConfigParser object

    Examples
    --------
    >>> !cat foo.txt
    # [array]
    # shape = 3
    # axis = -1
    1
    2
    3
    >>> _get_header_config('foo.txt')
    <pwtools.common.ConfigParser instance at 0x2c52320>
    """
    fn = common.get_filename(fh)
    verbose("[_read_header_config]: reading header from '%s'" %fn)
    header = ''
    for i in range(header_maxlines):
        try:
            line = next(fh).strip()
        except StopIteration:
            break
        if line.startswith(header_comment):
            header += line.replace(header_comment, '').strip() + '\n'
    # Read one more line to see if the header is bigger than header_maxlines.
    try:
        if next(fh).strip().startswith(header_comment):
            raise Exception("header seems to be > header_maxlines (%i)"
                %header_maxlines)
    except StopIteration:
        pass
    c = ConfigParser()
    c.read_file(StringIO(header))
    # If header_maxlines > header size, we read beyond the header into the data. That
    # causes havoc for all functions that read fh afterwards.
    fh.seek(0)
    return c


# the open_and_close decorator cannot be used here b/c it only opens
# files in read mode, not for writing
# XXX really? can't decorator take arguments as well
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
        fh.write((header_comment + ' ' + line).encode())
    ftmp.close()


# XXX same here, what about an argument for the decorator
def writetxt(fn, arr, axis=-1, maxdim=TXT_MAXDIM, header=True):
    """Write 1d, 2d or 3d arrays to txt file. 
    
    If 3d, write as 2d chunks. Take the 2d chunks along `axis`. Write a
    commented out ini-style header in the file with infos needed by readtxt()
    to restore the right shape.
    
    Parameters
    ----------
    fn : filename
    arr : nd array
    axis : axis along which 2d chunks are written
    maxdim : highest number of dims that `arr` is allowed to have
    header : bool
        Write ini style header. Can be used by readtxt().
    """
    verbose("[writetxt] writing: %s" %fn)
    common.assert_cond(arr.ndim <= maxdim, 'no rank > %i arrays supported' %maxdim)
    fh = open(fn, 'wb+')
    if header:
        c = ConfigParser()
        sec = 'array'
        c.add_section(sec)
        c.set(sec, 'shape', common.seq2str(arr.shape))
        c.set(sec, 'axis', str(axis))
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

    Parameters
    ----------
    fh : file_like
    axis : int
    shape : tuple
    **kwargs : keyword args passed to numpy.loadtxt(), e.g. comments='@@' to
        ignore weird lines etc.

    Returns
    -------
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
            line = next(fh).strip()
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
        arr = arr2d_to_3d(read_arr, shape=shape, axis=axis)
    verbose("[readtxt]    returning shape: %s" %str(arr.shape))
    return arr


def arr2d_to_3d(arr, shape, axis=-1):
    """Reshape 2d array `arr` to 3d array of `shape`, with 2d chunks aligned
    along `axis`.
    
    Parameters
    ----------
    arr : 2d array
    shape : tuple
        Target shape of 3d array
    axis : int
        Axis of 3d arr along which 2d chunks are placed.
    
    Returns
    -------
    arr3d : 3d array 

    Examples
    --------
    >>> axis = 1
    >>> shape = (50, 1000, 3)
    >>> shape_2d_chunk =  (50, 3)
    >>> arr.shape = (1000*50,3)
    """    
    assert arr.ndim == 2, "input must be 2d array"
    assert len(shape) == TXT_MAXDIM
    shape_2d_chunk = shape[:axis] + shape[(axis+1):]
    # arr:   (50*1000, 3)
    # arr3d: (50, 1000, 3) : natoms = 50, nstep = 1000, 3 = x,y,z
    expect_shape = (shape_2d_chunk[0]*shape[axis],) + (shape_2d_chunk[1],)
    common.assert_cond(arr.shape == expect_shape, 
                "input 2d array has not the correct "
                "shape, got %s, expect %s" %(str(arr.shape),
                                             str(expect_shape)))
    arr3d = np.empty(shape, dtype=arr.dtype)
    sl = [slice(None)]*3
    for ind in range(shape[axis]):
        sl[axis] = ind
        arr3d[sl] = arr[ind*shape_2d_chunk[0]:(ind+1)*shape_2d_chunk[0], :]
    return arr3d


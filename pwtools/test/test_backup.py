import os, tempfile, uuid
from pwtools.common import backup, file_write, file_read
from .testenv import testdir
pj = os.path.join

def create_full_dir(dn):
    os.makedirs(dn)
    for name in ['a', 'b', 'c']:
        file_write(pj(dn, name), 'foo')

def test_backup():
    # file
    name = tempfile.mktemp(prefix='testfile', dir=testdir)
    file_write(name, 'foo')
    backup(name)
    assert os.path.exists(name + '.0')
    backup(name)
    assert os.path.exists(name + '.1')
    backup(name)
    assert os.path.exists(name + '.2')
    
    # dir
    name = tempfile.mktemp(prefix='testdir', dir=testdir)
    create_full_dir(name)
    backup(name)
    assert os.path.exists(name + '.0')
    backup(name)
    assert os.path.exists(name + '.1')
    backup(name)
    assert os.path.exists(name + '.2')

    # link to file
    filename = tempfile.mktemp(prefix='testfile', dir=testdir)
    linkname = tempfile.mktemp(prefix='testlink', dir=testdir)
    file_write(filename, 'foo')
    os.symlink(filename, linkname)
    backup(linkname)
    assert os.path.exists(linkname + '.0')
    assert os.path.isfile(linkname + '.0')
    assert file_read(linkname + '.0') == file_read(filename)

    # link to dir
    dirname = tempfile.mktemp(prefix='testdir', dir=testdir)
    linkname = tempfile.mktemp(prefix='testlink', dir=testdir)
    create_full_dir(dirname)
    os.symlink(dirname, linkname)
    backup(linkname)
    assert os.path.exists(linkname + '.0')
    assert os.path.isdir(linkname + '.0')
    for name in ['a', 'b', 'c']:
        assert file_read(pj(dirname, name)) == \
               file_read(pj(linkname + '.0', name))
    
    # prefix
    name = tempfile.mktemp(prefix='testfile', dir=testdir)
    file_write(name, 'foo')
    backup(name, prefix="-bak")
    assert os.path.exists(name + '-bak0')

    # nonexisting src, should silently pass
    filename = str(uuid.uuid4())
    assert not os.path.exists(filename)
    backup(filename)

from time import sleep
from pwtools.timer import TagTimer

tt = TagTimer()

def test_timer():
    tt.t('outer loop')
    for ii in range(10):
        sleep(0.01)
        tt.t('inner loop')
        for jj in range(2):
            sleep(0.01)
        tt.pt('inner loop')
    tt.pt('outer loop')


import numpy as np
from nose.tools import assert_equal, nottest

from cili.extract import *

@nottest
def sc_samp_test(ds, events, duration, offset=0):
    ext, num = extract_events(ds, events, duration=duration, units=SAMP_UNITS, return_count=True)
    assert_equal(num * duration, np.shape(ext)[0])

@nottest
def sc_time_test(ds, events, sampfreq, duration, offset=0):
    ext, num = extract_events(ds, events, duration=duration, units=TIME_UNITS, return_count=True)
    assert_equal(int(sampfreq * (duration / 1000.)
                     * num), np.shape(ext)[0])

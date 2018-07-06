import numpy as np
from nose.tools import assert_equal, nottest

from cili.extract import *

@nottest
def sc_samp_test(ds, events, duration, offset=0):
    ext = extract_events(ds, events, duration=duration, units=SAMP_UNITS)
    ev_count = np.shape(events)[0]
    assert_equal(ev_count * duration, np.shape(ext)[0])

@nottest
def sc_time_test(ds, events, sampfreq, duration, offset=0):
    ext = extract_events(ds, events, duration=duration, units=TIME_UNITS)
    ev_count = np.shape(events)[0]
    assert_equal(int(sampfreq * (duration / 1000.)
                     * ev_count), np.shape(ext)[0])

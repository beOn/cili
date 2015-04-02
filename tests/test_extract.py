import os
import numpy as np
import pandas as pd
from nose.tools import assert_equal, assert_true, assert_raises, raises

from config import *
from cili.extract import *
from cili.util import load_eyelink_dataset, pandas_dfs_from_asc


# Unit tests for the extract module. Paths to data files assume tests
# are run from the /cili/ directory.

DATA_DIR = os.path.join(os.getcwd(), 'tests', 'data')

""" shape of extracted data """

def test_timeunit_extract_samplecount_250():
    ds, es = pandas_dfs_from_asc(paths['mono250'])
    fixations = es.EFIX[:-1]
    sc_time_test(ds, fixations, 250, 500)
    sc_time_test(ds, fixations, 250, 1000)

def test_timeunit_extract_samplecount_500():
    ds, es = pandas_dfs_from_asc(paths['mono500'])
    fixations = es.EFIX[:-1]
    sc_time_test(ds, fixations, 500, 500)
    sc_time_test(ds, fixations, 500, 1000)

def test_timeunit_extract_samplecount_1000():
    ds, es = pandas_dfs_from_asc(paths['mono1000'])
    fixations = es.EFIX[:-1]
    sc_time_test(ds, fixations, 1000, 500)
    sc_time_test(ds, fixations, 1000, 1000)

def test_timeunit_extract_samplecount_2000():
    ds, es = pandas_dfs_from_asc(paths['mono2000'])
    fixations = es.EFIX[:-1]
    sc_time_test(ds, fixations, 2000, 500)
    sc_time_test(ds, fixations, 2000, 1000)

def test_sampleunit_extract_samplecount_250():
    ds, es = pandas_dfs_from_asc(paths['mono250'])
    fixations = es.EFIX[:-1]
    sc_samp_test(ds, fixations, 10)
    sc_samp_test(ds, fixations, 130)

def test_sampleunit_extract_samplecount_500():
    ds, es = pandas_dfs_from_asc(paths['mono500'])
    fixations = es.EFIX[:-1]
    sc_samp_test(ds, fixations, 50)
    sc_samp_test(ds, fixations, 350)

def test_sampleunit_extract_samplecount_1000():
    ds, es = pandas_dfs_from_asc(paths['mono1000'])
    fixations = es.EFIX[:-1]
    sc_samp_test(ds, fixations, 500)
    sc_samp_test(ds, fixations, 750)

def test_sampleunit_extract_samplecount_2000():
    ds, es = pandas_dfs_from_asc(paths['mono2000'])
    fixations = es.EFIX[:-1]
    sc_samp_test(ds, fixations, 350)
    sc_samp_test(ds, fixations, 700)

def sc_samp_test(ds, events, duration, offset=0):
    ext = extract_events(ds, events, duration=duration, units=SAMP_UNITS)
    ev_count = np.shape(events)[0]
    assert_equal(ev_count * duration, np.shape(ext)[0])

def sc_time_test(ds, events, sampfreq, duration, offset=0):
    ext = extract_events(ds, events, duration=duration, units=TIME_UNITS)
    ev_count = np.shape(events)[0]
    assert_equal(int(sampfreq * (duration/1000.) * ev_count), np.shape(ext)[0])

""" exceptions """

@raises(IndexError)
def test_timeunit_extract_underbounds():
    ds, es = pandas_dfs_from_asc(paths['mono1000'])
    ext = extract_events(ds, es.EFIX, offset=-10000, duration=3000, units=TIME_UNITS)

@raises(IndexError)
def test_timeunit_extract_overbounds():
    ds, es = pandas_dfs_from_asc(paths['mono1000'])
    ext = extract_events(ds, es.EFIX, duration=3000, units=TIME_UNITS)

""" bounds """

@raises(ValueError)
def test_negative_duration():
    ds, es = pandas_dfs_from_asc(paths['mono250'])
    fixations = es.EFIX[:-1]
    ext = extract_events(ds, fixations, duration=-1000, units=TIME_UNITS)

def test_posneg_offset():
    ds, es = pandas_dfs_from_asc(paths['mono250'])
    fixations = es.EFIX[3:-1]
    ext = extract_events(ds, fixations, duration=500, offset=-250, units=TIME_UNITS)
    ext = extract_events(ds, fixations, duration=500, offset=250, units=TIME_UNITS)
    assert(True)

# TODO: test fields of returned data for all data types

# TODO: test that you can borrow attributes

import os
import numpy as np
import pandas as pd
from nose.tools import assert_equal, assert_true, assert_raises, raises
from numpy.testing import assert_array_equal

from tests.config import *
from tests.helpers import *
from cili.extract import *
from cili.util import load_eyelink_dataset, pandas_dfs_from_asc


# Unit tests for the extract module. Paths to data files assume tests
# are run from the /cili/ directory.

DATA_DIR = os.path.join(os.getcwd(), 'tests', 'data')

""" shape of extracted data """

def test_timeunit_extract_samplecount_250():
    ds, es = pandas_dfs_from_asc(paths['mono250'])
    fixations = es.EFIX[:-2]
    sc_time_test(ds, fixations, 250, 400)

def test_timeunit_extract_samplecount_500():
    ds, es = pandas_dfs_from_asc(paths['mono500'])
    fixations = es.EFIX[:-2]
    sc_time_test(ds, fixations, 500, 20)
    sc_time_test(ds, fixations, 500, 100)


def test_timeunit_extract_samplecount_1000():
    ds, es = pandas_dfs_from_asc(paths['mono1000'])
    fixations = es.EFIX[:-2]
    sc_time_test(ds, fixations, 1000, 20)
    sc_time_test(ds, fixations, 1000, 100)


def test_timeunit_extract_samplecount_2000():
    ds, es = pandas_dfs_from_asc(paths['mono2000'])
    fixations = es.EFIX[:-2]
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


""" exceptions """


@raises(ValueError)
def test_timeunit_extract_underbounds():
    ds, es = pandas_dfs_from_asc(paths['mono1000'])
    ext = extract_events(ds, es.EFIX, offset=-5000,
                         duration=3000, units=TIME_UNITS)


@raises(ValueError)
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
    # make sure that you can use a positive or a negative offset
    ds, es = pandas_dfs_from_asc(paths['mono250'])
    fixations = es.EFIX[3:-1]
    ext = extract_events(ds, fixations, duration=40,
                         offset=-250, units=TIME_UNITS)
    ext = extract_events(ds, fixations, duration=40,
                         offset=250, units=TIME_UNITS)
    assert(True)


""" fields """


def test_extracted__bino250_cols():
    ds, es = pandas_dfs_from_asc(paths['bino250'])
    rs = es.ESACC[:-3]
    ext = extract_events(ds, rs, duration=40, offset=0, units=TIME_UNITS)
    test_cols = ['x_l', 'y_l', 'pup_l', 'x_r',
                 'y_r', 'pup_r', 'samp_warns', 'orig_idx']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_extracted_bino500_cols():
    ds, es = pandas_dfs_from_asc(paths['bino500'])
    rs = es.ESACC[:-3]
    ext = extract_events(ds, rs, duration=500, offset=0, units=TIME_UNITS)
    test_cols = ['x_l', 'y_l', 'pup_l', 'x_r',
                 'y_r', 'pup_r', 'samp_warns', 'orig_idx']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_extracted_bino1000_cols():
    ds, es = pandas_dfs_from_asc(paths['bino1000'])
    rs = es.ESACC[:-3]
    ext = extract_events(ds, rs, duration=500, offset=0, units=TIME_UNITS)
    test_cols = ['x_l', 'y_l', 'pup_l', 'x_r',
                 'y_r', 'pup_r', 'samp_warns', 'orig_idx']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_extracted_binoRemote250_cols():
    ds, es = pandas_dfs_from_asc(paths['binoRemote250'])
    rs = es.EFIX[:-3]
    ext = extract_events(ds, rs, duration=40, offset=0, units=TIME_UNITS)
    test_cols = ['x_l', 'y_l', 'pup_l', 'x_r', 'y_r', 'pup_r', 'samp_warns', 'targ_x',
                 'targ_y', 'targ_dist', 'remote_warns', 'orig_idx']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_extracted_binoRemote500_cols():
    ds, es = pandas_dfs_from_asc(paths['binoRemote500'])
    rs = es.ESACC[:-3]
    ext = extract_events(ds, rs, duration=500, offset=0, units=TIME_UNITS)
    test_cols = ['x_l', 'y_l', 'pup_l', 'x_r', 'y_r', 'pup_r', 'samp_warns', 'targ_x',
                 'targ_y', 'targ_dist', 'remote_warns', 'orig_idx']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_extracted_mono250_cols():
    ds, es = pandas_dfs_from_asc(paths['mono250'])
    rs = es.ESACC[:-3]
    ext = extract_events(ds, rs, duration=40, offset=0, units=TIME_UNITS)
    test_cols = ['x_l', 'y_l', 'pup_l', 'samp_warns', 'orig_idx']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_extracted_mono500_cols():
    ds, es = pandas_dfs_from_asc(paths['mono500'])
    rs = es.ESACC[:-3]
    ext = extract_events(ds, rs, duration=500, offset=0, units=TIME_UNITS)
    test_cols = ['x_l', 'y_l', 'pup_l', 'samp_warns', 'orig_idx']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_extracted_mono1000_cols():
    ds, es = pandas_dfs_from_asc(paths['mono1000'])
    rs = es.ESACC[:-3]
    ext = extract_events(ds, rs, duration=50, offset=0, units=TIME_UNITS)
    test_cols = ['x_r', 'y_r', 'pup_r', 'samp_warns', 'orig_idx']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_extracted_mono2000_cols():
    ds, es = pandas_dfs_from_asc(paths['mono2000'])
    rs = es.ESACC[:-3]
    ext = extract_events(ds, rs, duration=500, offset=0, units=TIME_UNITS)
    test_cols = ['x_r', 'y_r', 'pup_r', 'samp_warns', 'orig_idx']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_extracted_monoRemote250_cols():
    ds, es = pandas_dfs_from_asc(paths['monoRemote250'])
    rs = es.EFIX[:-3]
    ext = extract_events(ds, rs, duration=500, offset=0, units=TIME_UNITS)
    test_cols = ['x_l', 'y_l', 'pup_l', 'samp_warns', 'targ_x', 'targ_y', 'targ_dist',
                 'remote_warns', 'orig_idx']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_extracted_monoRemote500_cols():
    ds, es = pandas_dfs_from_asc(paths['monoRemote500'])
    rs = es.ESACC[:-3]
    ext = extract_events(ds, rs, duration=500, offset=0, units=TIME_UNITS)
    test_cols = ['x_l', 'y_l', 'pup_l', 'samp_warns', 'targ_x', 'targ_y', 'targ_dist',
                 'remote_warns', 'orig_idx']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_borrowed_fields():
    ds, es = pandas_dfs_from_asc(paths['binoRemote500'])
    rs = es.ESACC[:-3]
    ext = extract_events(ds, rs, duration=500, offset=0, units=TIME_UNITS,
                         borrow_attributes=['peak_velocity'])
    assert_true('peak_velocity' in ext.columns)

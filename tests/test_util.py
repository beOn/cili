import os
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.util.testing import assert_frame_equal
from nose.tools import assert_equal, assert_true, assert_raises, raises

from tests.config import *
from cili.util import load_eyelink_dataset, pandas_dfs_from_asc

# Unit tests for the util module. Paths to data files assume tests
# are run from the /cili/ directory.
#
# To run tests, type `nosetests -v` with `nose` installed.

""" load_eyelink_dataset """


def test_load_asc_convenience():
    # TODO: make sure you can load an asc without an exception
    load_eyelink_dataset(paths['bino250'])


# def test_load_txt_convenience():
#     # TODO: make sure you can load a text file
#     raise NotImplementedError()


""" pandas_df_from_txt """
# TODO: we need test data for this...

""" pandas_dfs_from_asc """
# test that the function can be run on each data type


def test_load_asc_bino250():
    pandas_dfs_from_asc(paths['bino250'])


def test_load_asc_bino500():
    pandas_dfs_from_asc(paths['bino500'])


def test_load_asc_bino1000():
    pandas_dfs_from_asc(paths['bino1000'])


def test_load_asc_binoRemote250():
    pandas_dfs_from_asc(paths['binoRemote250'])


def test_load_asc_binoRemote500():
    pandas_dfs_from_asc(paths['binoRemote500'])


def test_load_asc_mono250():
    pandas_dfs_from_asc(paths['mono250'])


def test_load_asc_mono500():
    pandas_dfs_from_asc(paths['mono500'])


def test_load_asc_mono1000():
    pandas_dfs_from_asc(paths['mono1000'])


def test_load_asc_mono2000():
    pandas_dfs_from_asc(paths['mono2000'])


def test_load_asc_monoRemote250():
    pandas_dfs_from_asc(paths['monoRemote250'])


def test_load_asc_monoRemote500():
    pandas_dfs_from_asc(paths['monoRemote500'])

# test that for each type of data, you get the expected headers


def test_bino250_cols():
    ds, es = pandas_dfs_from_asc(paths['bino250'])
    test_cols = ['x_l', 'y_l', 'pup_l', 'x_r', 'y_r', 'pup_r', 'samp_warns']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_bino500_cols():
    ds, es = pandas_dfs_from_asc(paths['bino500'])
    test_cols = ['x_l', 'y_l', 'pup_l', 'x_r', 'y_r', 'pup_r', 'samp_warns']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_bino1000_cols():
    ds, es = pandas_dfs_from_asc(paths['bino1000'])
    test_cols = ['x_l', 'y_l', 'pup_l', 'x_r', 'y_r', 'pup_r', 'samp_warns']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_binoRemote250_cols():
    ds, es = pandas_dfs_from_asc(paths['binoRemote250'])
    test_cols = ['x_l', 'y_l', 'pup_l', 'x_r', 'y_r', 'pup_r',
                 'samp_warns', 'targ_x', 'targ_y', 'targ_dist', 'remote_warns']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_binoRemote500_cols():
    ds, es = pandas_dfs_from_asc(paths['binoRemote500'])
    test_cols = ['x_l', 'y_l', 'pup_l', 'x_r', 'y_r', 'pup_r',
                 'samp_warns', 'targ_x', 'targ_y', 'targ_dist', 'remote_warns']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_mono250_cols():
    ds, es = pandas_dfs_from_asc(paths['mono250'])
    test_cols = ['x_l', 'y_l', 'pup_l', 'samp_warns']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_mono500_cols():
    ds, es = pandas_dfs_from_asc(paths['mono500'])
    test_cols = ['x_l', 'y_l', 'pup_l', 'samp_warns']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_mono1000_cols():
    ds, es = pandas_dfs_from_asc(paths['mono1000'])
    test_cols = ['x_r', 'y_r', 'pup_r', 'samp_warns']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_mono2000_cols():
    ds, es = pandas_dfs_from_asc(paths['mono2000'])
    test_cols = ['x_r', 'y_r', 'pup_r', 'samp_warns']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_monoRemote250_cols():
    ds, es = pandas_dfs_from_asc(paths['monoRemote250'])
    test_cols = ['x_l', 'y_l', 'pup_l', 'samp_warns',
                 'targ_x', 'targ_y', 'targ_dist', 'remote_warns']
    assert_array_equal(test_cols, ds.columns.tolist())


def test_monoRemote500_cols():
    ds, es = pandas_dfs_from_asc(paths['monoRemote500'])
    test_cols = ['x_l', 'y_l', 'pup_l', 'samp_warns',
                 'targ_x', 'targ_y', 'targ_dist', 'remote_warns']
    assert_array_equal(test_cols, ds.columns.tolist())

# test that for each frequency, the sample rate is what you expect


def test_mono250_idx():
    ds, es = pandas_dfs_from_asc(paths['mono250'])
    diffs = np.diff(ds.index)
    diffs = np.unique(diffs[diffs < 100])
    assert_equal(diffs, 4)


def test_mono500_idx():
    ds, es = pandas_dfs_from_asc(paths['mono500'])
    diffs = np.diff(ds.index)
    diffs = np.unique(diffs[diffs < 100])
    assert_equal(diffs, 2)


def test_mono1000_idx():
    ds, es = pandas_dfs_from_asc(paths['mono1000'])
    diffs = np.diff(ds.index)
    diffs = np.unique(diffs[diffs < 100])
    assert_equal(diffs, 1)


def test_mono2000_idx():
    ds, es = pandas_dfs_from_asc(paths['mono2000'])
    diffs = np.diff(ds.index)
    diffs = sorted(np.unique(diffs[diffs < 100]))
    assert_equal(diffs, [0,1])

# check extracted event types and fields against the expected values


def test_event_types():
    # make sure the full list of events is there
    ds, es = pandas_dfs_from_asc(paths['binoRemote500'])
    test_evs = sorted(['END', 'EFIX', 'EBLINK', 'START', 'ESACC', 'MSG'])
    assert_array_equal(sorted(list(es.dframes.keys())), test_evs)


def test_end_fields():
    ds, es = pandas_dfs_from_asc(paths['binoRemote500'])
    test_fields = ['name', 'types', 'x_res', 'y_res']
    assert_array_equal(es.END.columns.values, test_fields)


def test_efix_fields():
    ds, es = pandas_dfs_from_asc(paths['binoRemote500'])
    test_fields = ['name', 'eye', 'last_onset', 'duration',
                   'x_pos', 'y_pos', 'p_size']
    assert_array_equal(es.EFIX.columns.values, test_fields)


def test_eblink_fields():
    ds, es = pandas_dfs_from_asc(paths['binoRemote500'])
    test_fields = ['name', 'eye', 'last_onset', 'duration']
    assert_array_equal(es.EBLINK.columns.values, test_fields)


def test_start_fields():
    ds, es = pandas_dfs_from_asc(paths['binoRemote500'])
    test_fields = ['name', 'eye', 'types']
    assert_array_equal(es.START.columns.values, test_fields)


def test_esacc_fields():
    ds, es = pandas_dfs_from_asc(paths['binoRemote500'])
    test_fields = ['name', 'eye', 'last_onset', 'duration',
                   'x_start', 'y_start', 'x_end', 'y_end',
                   'vis_angle', 'peak_velocity']
    assert_array_equal(es.ESACC.columns.values, test_fields)


def test_msg_fields():
    ds, es = pandas_dfs_from_asc(paths['binoRemote500'])
    test_fields = ['name', 'label', 'content']
    assert_array_equal(es.MSG.columns.values, test_fields)

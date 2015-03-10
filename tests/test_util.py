# import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
# from numpy.testing import assert_allclose
# from nose.tools import assert_equal, assert_true, assert_raises, raises

from cili.util import load_eyelink_dataset
import os

# Unit tests for the util module. Paths to data files assume tests
# are run from the /cili/ directory.
#
# To run tests, type `nosetests -v` with `nose` installed.


def test_load_eyelink_dataset_1():
    """Test loading samples..."""
    fname = os.path.join(os.getcwd(), 'tests', 'data_1.asc')
    samps, events = load_eyelink_dataset(fname)

    fname = os.path.join(os.getcwd(), 'tests', 'samps_1.csv')
    true = pd.read_csv(fname, index_col='onset')
    assert_frame_equal(samps, true)


def test_load_eyelink_dataset_2():
    """Test loading events..."""
    fname = os.path.join(os.getcwd(), 'tests', 'data_1.asc')
    samps, events = load_eyelink_dataset(fname)

    fname = os.path.join(os.getcwd(), 'tests', 'events_1.csv')
    true = pd.read_csv(fname, index_col=0)
    assert_frame_equal(events.EFIX, true)


def test_load_eyelink_dataset_3():
    """Test loading samples..."""
    fname = os.path.join(os.getcwd(), 'tests', 'data_2.asc')
    samps, events = load_eyelink_dataset(fname)

    fname = os.path.join(os.getcwd(), 'tests', 'samps_2.csv')
    true = pd.read_csv(fname, index_col='onset')
    assert_frame_equal(samps, true)


def test_load_eyelink_dataset_4():
    """Test loading events..."""
    fname = os.path.join(os.getcwd(), 'tests', 'data_2.asc')
    samps, events = load_eyelink_dataset(fname)

    fname = os.path.join(os.getcwd(), 'tests', 'events_2.csv')
    true = pd.read_csv(fname, index_col=0)
    assert_frame_equal(events.EFIX, true)

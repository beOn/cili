import pandas as pd
from pandas.util.testing import assert_frame_equal
# from numpy.testing import assert_allclose
# from nose.tools import assert_equal, assert_true, assert_raises, raises

from cili.extract import extract_events
from cili.util import load_eyelink_dataset
import os

# Unit tests for the extract module. Paths to data files assume tests
# are run from the /cili/ directory.


def test_extract_events_1():
    """Test event extraction for dataset 2, "duration" of
    5 samples"""
    fname = os.path.join(os.getcwd(), 'tests', 'data_2.asc')
    samps, events = load_eyelink_dataset(fname)
    extracted = extract_events(samps, events.EFIX, duration=5)

    fname = os.path.join(os.getcwd(), 'tests', 'extracted_1.csv')
    true = pd.read_csv(fname, index_col=[0, 1])
    assert_frame_equal(extracted, true)

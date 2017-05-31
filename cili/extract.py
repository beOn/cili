from .models import *
import pandas as pd
import numpy as np
from copy import deepcopy

TIME_UNITS = 'time'
SAMP_UNITS = 'samples'

def extract_event_ranges(samples, events_dataframe, start_offset=0,
                         end_offset=0, round_indices=True, borrow_attributes=[]):
    """ Extracts ranges from samples based on event timing.

    This method works, but has been replaced by extract_events (below).

    Parameters
    ----------
    samples (Samples object)
        The Samples object from which you'd like to extract ranges.
    events_dataframe (DataFrame object containing event timing info)
        Indices should be onset times, duration should be in a column named
        'duration'. Note that if you have an Events object evs, and it has,
        say, a set of events named "EBLINK", then you can pass evs.EBLINK
        for this argument.
    start_offset (number - same type as your samples index, probably ms)
        Each index of the events_dataframe is an event onset time, and we add
        the start_offset to each of these times to find the beginnings of our
        target ranges, then search the sample times to find the sample indices
        of these range onset times. If there isn't an exact match, we pick the
        last sample time before the range onset time.
    end_offset (number - same type as your samples index, probably ms)
        Like start_offset, but for the offsets of target ranges instead of the
        onsets. Note, the sample containing the range offset time will be
        *included* in the extracted range.
    borrow_attributes (list of strings)
        A list of column names in the events_dataframe whose values you would
        like to copy to the respective ranges. For each item in the list, a
        column will be created in the ranges dataframe - if the column does
        not exist in the events dataframe, the values in the each
        corrisponding range will be set to float('nan').
    round_indices (bool)
        Deprecated.
    """
    from warnings import warn
    warn("extract_event_ranges is deprecated, use extract_events instead.")
    if start_offset >= end_offset:
        raise ValueError("start_offset must be < end_offset")
    # get the list of start and stop times - note that we no longer pay
    # attention to the stop times (see below)
    e_starts = events_dataframe.index.to_series()
    r_times = pd.DataFrame(e_starts + end_offset)
    r_times.index += start_offset
    r_times.columns = ['last_onset']
    # sanity check - make sure no events start before the data, or end afterwards
    if any(r_times.index < samples.index[0]):
        raise ValueError("at least one event range starts before the first sample")
    if any(r_times.index > samples.index[-1]):
        raise ValueError("at least one event range ends after the last sample")

    # get the indices for the first event (minus the first index)
    ev_idxs = np.logical_and(samples.index <= r_times.last_onset.iloc[0],
                             samples.index > r_times.index[0])
    # this method just uses the length of the first event as a template for
    # all future events
    r_len = len(np.where(ev_idxs)[0]) + 1
    # we're going to make a df with a hierarchical index.
    samples['orig_idx'] = samples.index
    midx = pd.MultiIndex.from_product([list(range(len(e_starts))), list(range(r_len))],
        names=['event', 'onset'])
    # get all of the samples!
    # idxs = []
    df = pd.DataFrame()
    idx = 0
    for stime, etime in r_times.itertuples():
        # get the start time... add the number of indices that you want...
        s_idx = np.where(samples.index > stime)[0][0]-1
        e_idx = s_idx + r_len - 1
        stime = samples.index[s_idx]
        etime = samples.index[e_idx]
        new_df = samples.loc[stime:etime]
        if borrow_attributes:
            for ba in borrow_attributes:
                new_df[ba] = events_dataframe.iloc[idx].get(ba, float('nan'))
        df = pd.concat([df, new_df])
        idx += 1
    df.index = midx
    return df

def extract_events(samples, events, offset=0, duration=0,
                   units='samples', borrow_attributes=[]):
    """ Extracts ranges from samples based on event timing and sample count.

    Parameters
    ==========
    samples (Samples object)
        The Samples object from which you'd like to extract ranges.
    events (DataFrame object containing event timing info)
        Indices should be onset times, duration should be in a column named
        'duration'. Note that if you have an Events object evs, and it has,
        say, a set of events named "EBLINK", then you can pass evs.EBLINK
        for this argument.
    offset (number)
        How to position extraction range start relative to event start.
        Interpretation depends upon 'units'. Default 0.
    duration (number)
        How long a range to extract. Interpretation depends upon 'units'.
        Default 0. Note that if this and offset are both 0, you'll get None in
        return.
    units (string constant)
        Can be 'time' or 'samples'. Default is 'samples'. Determines which index
        will be used to interpret the offset and duration parameters. If units
        is 'time', then we will extract ranges offset from each event's start
        time by 'offset' ms, and 'duration' ms long (or as close as we can get
        given your sampling frequency). Actually, we use the sample count of
        the first event as a template for all events, so this method can be a
        little slippery. For finer control over the size of the returned
        dataset, you can set 'units' to 'samples'. Then, we will extract ranges
        offset from each event's start time by 'offset' *samples*, and
        'duration' samples long. It's then up to you to calculate how long the
        sample is in time, based on your sampling rate.
    borrow_attributes (list of strings)
        A list of column names in the 'events' whose values you would like to
        copy to the respective ranges. For each item in the list, a column
        will be created in the ranges dataframe - if the column does not exist
        in the events dataframe, the values in the each corrisponding range
        will be set to float('nan').
    """
    # dummy check
    if offset == 0 and duration == 0:
        return None
    # negative duration should raise an exception
    if duration <= 0:
        raise ValueError("Duration must be >0")
    # get the list of start and stop sample indices
    e_starts = events.index.to_series()

    if units == TIME_UNITS:
        # get the indices for the first event (minus the first index), then use
        # the length of the first event as a template for all events
        r_times = e_starts+offset
        ev_idxs = np.logical_and(samples.index <= r_times.iloc[0] + duration,
                                 samples.index > r_times.iloc[0])
        r_dur = len(np.where(ev_idxs)[0]) + 1
        r_idxs = [np.where(samples.index > rt)[0][0]-1 for rt in r_times]
        # sanity check - make sure no events start before the data, or end afterwards
        if any(r_times < samples.index[0]):
            raise ValueError("at least one event range starts before the first sample")
        if any(r_times > samples.index[-1]):
            raise ValueError("at least one event range ends after the last sample")
    elif units == SAMP_UNITS:
        # just find the indexes of the event starts, and offset by sample count
        r_idxs = np.array([np.where(samples.index > et)[0][0]-1+offset for et in e_starts])
        r_dur = duration
        if any(r_idxs < 0):
            raise ValueError("at least one event range starts before the first sample")
        if any(r_idxs >= len(samples)):
            raise ValueError("at least one event range ends after the last sample")
    else:
        raise ValueError("Not a valid unit!")

    # make a hierarchical index
    samples['orig_idx'] = samples.index
    midx = pd.MultiIndex.from_product([list(range(len(e_starts))), list(range(r_dur))],
        names=['event', 'onset'])
    # get the samples
    df = pd.DataFrame()
    idx = 0
    for s_idx in r_idxs:
        # get the start time... add the number of indices that you want...
        e_idx = s_idx + r_dur-1 # pandas.loc indexing is inclusive
        # this deepcopy is heavy handed... but gets around some early pandas bugs
        new_df = deepcopy(samples.loc[samples.index[s_idx] : samples.index[e_idx]])
        for ba in borrow_attributes:
            new_df[ba] = events.iloc[idx].get(ba, float('nan'))
        df = pd.concat([df, new_df])
        idx += 1
    df.index = midx
    return df

from .models import *
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.stats import mode

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
        raise ValueError(
            "at least one event range starts before the first sample")
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
        s_idx = np.where(samples.index > stime)[0][0] - 1
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
                   units='samples', borrow_attributes=[], return_count=False):
    """ Extracts ranges from samples based on event timing and sample count.

    Note that we will exclude any ranges which would cross discontinuities in
    the dataset. If there are no events to return, we will return None

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
    return_count (bool)
        If true, will return the number of events extracted
    """
    # dummy check
    if offset == 0 and duration == 0:
        if return_count:
            return None, 0
        return None
    # negative duration should raise an exception
    if duration <= 0:
        raise ValueError("Duration must be >0")
    # get the list of start time indices
    e_starts = events.index.to_series()
    # find the indices of discontinuities
    idiff = np.diff(samples.index)
    diffmode = mode(idiff[np.where(idiff > 0)])[0][0]
    disc_idxs = np.where(idiff > diffmode)[0] + 1

    if units == TIME_UNITS:
        # we want all of the extracted chunks to be the same length. but we're
        # dealing with time, so who knows if time and samples are well aligned
        # in all cases. so, we're going to get the sample index bounds for the
        # first event, then re-use the length of the first event (# samples) for
        # all other events.

        # first, find the first samples of all of the events (taking the offset
        # into account). searchsorted returns the insertion point needed to
        # maintain sort order, so the first time index of an event is the
        # leftmost insertion point for each event's start time.
        r_times = e_starts + offset
        r_idxs = np.searchsorted(samples.index, r_times.iloc[:], 'left')

        if any(r_times < samples.index[0]):
            raise ValueError(
                "at least one event range starts before the first sample")

        # exclude events that cross discontinuities
        e_idxs = np.searchsorted(samples.index, r_times.iloc[:] + duration, 'left')
        ok_idxs = [i for i in range(len(r_idxs)) if not
            any([all((r_idxs[i]<=d, e_idxs[i]>=d)) for d in disc_idxs])]
        if (len(r_idxs) - len(ok_idxs)) == 0:
            print("excluding %d events for crossing discontinuities" %  (len(r_idxs) - len(ok_idxs)))
        # return None if there's nothing to do
        if len(ok_idxs) == 0:
            if return_count:
                return None, 0
            return None
        # trim the events data
        events = events.iloc[ok_idxs]
        e_starts = e_starts.iloc[ok_idxs]
        r_idxs = r_idxs[ok_idxs]
        e_idxs = e_idxs[ok_idxs]
        
        # find the duration of the first event.
        r_dur = e_idxs[0] - r_idxs[0]
    elif units == SAMP_UNITS:
        # just find the indexes of the event starts, and offset by sample count
        r_idxs = np.searchsorted(samples.index, e_starts.iloc[:], 'left') + offset
        r_dur = duration

        # exclude events that cross discontinuities
        e_idxs = r_idxs + duration
        ok_idxs = [i for i in range(len(r_idxs)) if not
            any([all((r_idxs[i]<=d, e_idxs[i]>=d)) for d in disc_idxs])]
        if (len(r_idxs) - len(ok_idxs)) == 0:
            print("excluding %d events for crossing discontinuities" %  (len(r_idxs) - len(ok_idxs)))
        # return None if there's nothing to do
        if len(ok_idxs) == 0:
            if return_count:
                return None, 0
            return None
        # trim the events data
        events = events.iloc[ok_idxs]
        e_starts = e_starts.iloc[ok_idxs]
        r_idxs = r_idxs[ok_idxs]
        e_idxs = e_idxs[ok_idxs]
    else:
        raise ValueError("'%s' is not a valid unit!" % units)

    # sanity check - make sure no events start before the data, or end afterwards
    if any(r_idxs < 0):
        raise ValueError(
            "at least one event range starts before the first sample")
    if any(e_idxs >= len(samples)):
        raise ValueError(
            "at least one event range ends after the last sample")

    # make a hierarchical index
    samples['orig_idx'] = samples.index
    midx = pd.MultiIndex.from_product([list(range(len(e_starts))), list(range(r_dur))],
                                      names=['event', 'onset'])
    # get the samples
    df = pd.DataFrame()
    idx = 0
    for s_idx in r_idxs:
        # get the start time... add the number of indices that you want...
        e_idx = s_idx + r_dur - 1  # pandas.loc indexing is inclusive
        # this deepcopy is heavy handed... but gets around some early pandas bugs
        new_df = deepcopy(
            samples.loc[samples.index[s_idx]: samples.index[e_idx]])
        for ba in borrow_attributes:
            new_df[ba] = events.iloc[idx].get(ba, float('nan'))
        df = pd.concat([df, new_df])
        idx += 1
    df.index = midx
    if return_count:
        return df, len(r_idxs)
    return df

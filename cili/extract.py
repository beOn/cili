from models import *
import pandas as pd
import numpy as np

def extract_event_ranges(samples, events_dataframe, start_offset=0,
                         end_offset=0, round_indices=True, borrow_attributes=[]):
    """ Extracts ranges from samples based on event timing.

    See note at bottom - this method works, but should be replaced.

    Parameters
    ----------
    samples (Samples object)
        The Samples object from which you'd like to extract ranges.
    events_dataframe (DataFrame object containing event timing info)
        Indices should be onset times, duration should be in a column named
        'duration'. Note that if you have an Events object evs, and it has,
        say, a set of events named "EBLINK", then you can pass Events.EBLINK
        for this argument.
    start_offset (number - same type as your samples index, probably ms)
        Each index of the events_dataframe is an event onset time, and we add
        the start_offset to each of these times to find the beginnings our or
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
        Depricated.

    # TODO: this really should be replaced with a method that just takes a
    # start offset and a sample count. It's cleaner than calculating the
    # sample count from the first event, which makes it hard to control the
    # shape of the data the function returns.

    """
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
    if any(r_times.index < samples.index[0]):
        raise ValueError("at least one event range ends after the last sample")

    # get the indices for the first event (minus the first index)
    ev_idxs = np.logical_and(samples.index <= r_times.last_onset.iloc[0],
                             samples.index > r_times.index[0])
    # this method just uses the length of the first event as a template for
    # all future events
    r_len = len(np.where(ev_idxs)[0]) + 1
    # we're going to make a df with a hierarchical index.
    samples['orig_idx'] = samples.index
    midx = pd.MultiIndex.from_product([range(len(e_starts)), range(r_len)],
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

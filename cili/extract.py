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
    round_indices (bool)
        Default is True. If True, we'll use samples.index.asof on each of the
        start/end times to make sure we ask for valid indices. If false, you
        may have issues using events defined outside of your eyetracking
        software. Take care to ensure that your onsets/durations align with
        your samples. The downside of using the default setting is that event
        sample onsets may no longer align with the events_dataframe onsets.
    borrow_attributes (list of strings)
        A list of column names in the events_dataframe whose values you would
        like to copy to the respective ranges. For each item in the list, a
        column will be created in the ranges dataframe - if the column does
        not exist in the events dataframe, the values in the each
        corrisponding range will be set to float('nan').

    # TODO: this really should be replaced with a method that just takes a
    # start offset and a sample count. It's cleaner than calculating the
    # sample count from the first event, which makes it hard to control the
    # shape of the data the function returns.

    """
    if start_offset >= end_offset:
        raise ValueError("start_offset must be < end_offset")
    # get the list of start and stop times
    e_starts = events_dataframe.index.to_series()
    r_times = pd.DataFrame(e_starts + end_offset)
    r_times.index += start_offset
    r_times.columns = ['last_onset']
    # find the number of samples per range... right now, we assume they're all equal
    r_len = samples.index.get_loc(r_times.last_onset.iloc[0]) - samples.index.get_loc(r_times.index[0]) + 1
    # we're going to make a df with a hierarchical index.
    # There's an annoying assumption hidden here!
    samples['orig_idx'] = samples.index
    midx = pd.MultiIndex.from_product([range(len(e_starts)), range(r_len)],
        names=['event', 'onset'])
    # get all of the samples!
    # idxs = []
    df = pd.DataFrame()
    idx = 0
    for stime, etime in r_times.itertuples():
        # lr_len = samples.index.get_loc(etime) - samples.index.get_loc(stime) + 1
        if round_indices:
            stime = samples.index.asof(stime)
            etime = samples.index.asof(etime)
            stime = getattr(stime,'value',stime)
            etime = getattr(etime,'value',etime)
        new_df = samples.loc[stime:etime]
        if borrow_attributes:
            for ba in borrow_attributes:
                new_df[ba] = events_dataframe.iloc[idx].get(ba, float('nan'))
        df = pd.concat([df, new_df])
        idx += 1
        # idxs.extend(samples.loc[stime:etime].index.tolist())
    # news = samples.loc[idxs]
    df.index = midx
    return df

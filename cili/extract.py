from models import *
import pandas as pd
import numpy as np

def extract_event_ranges(samples, events_dataframe, start_offset=0,
                         end_offset=0, round_indices=True, borrow_attributes=[]):
    """ Extracts ranges from samples based on event timing.

    Parameters
    ----------
    samples (Samples object)
        The Samples object from which you'd like to extract ranges.
    events_dataframe (DataFrame object containing event timing info)
        Indices should be onset times, duration should be in a column named
        'duration'. Note that if you have an Events object evs, and it has,
        say, a set of events named "EBLINK", then you can pass Events.EBLINK
        for this argument.
    start_offset (numer - same type as your samples index)
        Added to each event onset to find the range start indices. Note, the
        start index is *included* in the returned samples.
    end_offset (numer - same type as your samples index)
        Added to each event onset to find the range end indices. Note, the end
        index is *included* from the subsample. See documentation on pandas
        dataset .loc method for more info. So if your data is 1KHz, your
        start_offset is 0, and you want 1000 ms worth of data your end_offset
        should be 999. Got that? It's the offset between the onset of the
        event, and the onset of the last sample you want to *include* in the
        range.
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
    """
    if start_offset == end_offset or start_offset > end_offset:
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

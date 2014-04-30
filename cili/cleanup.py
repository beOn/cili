from models import *
import pandas as pd

#-------------------------------------------------------------
# Masking

def find_nested_events(samples, outer, inner):
    """ returns indices in outer that contain events in inner """
    # looking for inner events whose onset is at or before outer offset,
    # and whose offset is at or after inner onset
    # get list of onsets of first samples *after* our events
    onsets = inner.index.to_series()
    post_onsets = onsets + inner.duration
    # convert to list of positional indices
    max_onset = samples.index[-1]
    last_idxs = post_onsets.apply(lambda x: max(0, samples.index.searchsorted(x, side="right")-1))
    # last_idxs = post_onsets.apply(lambda x: samples.index.get_loc(min(x,max_onset)))
    # step back by one positional index to get pos. index of last samples of our events.
    # stupid fix - don't nudge the index back for events whose duration went beyond the samples
    end_safe_evs = post_onsets <= max_onset
    last_idxs[end_safe_evs] = last_idxs[end_safe_evs] - 1
    # get the time indices of the last samples of our events
    last_onsets = last_idxs.apply(lambda x: samples.index[x])
    idxs = outer.apply(has_overlapping_events, axis=1, args=[onsets, last_onsets])
    if len(idxs) == 0:
        return pd.DataFrame()
    return outer[idxs]

def has_overlapping_events(row, onsets, last_onsets):
    """ searches series last_onsets for rows with onset <= row offset, and offset >= row onset. """
    matches = last_onsets[(onsets <= row.name+row.duration) & (last_onsets >= row.name)]
    return len(matches) > 0

def get_eyelink_mask_events(samples, events, find_recovery=True, recovery_field="pup_l"):
    be = events.EBLINK.duration.to_frame()
    be = pd.concat([be, find_nested_events(samples, events.ESACC.duration.to_frame(), be)])
    if find_recovery:
        adjust_eyelink_recov_idxs(samples, be, field=recovery_field)
    return be

def get_eyelink_mask_idxs(samples, events, find_recovery=True, recovery_field="pup_l"):
    be = get_eyelink_mask_events(samples, events, find_recovery=find_recovery, recovery_field=recovery_field)
    bi = ev_row_idxs(samples, be)
    return bi

def mask_eyelink_blinks(samples, events, mask_fields=["pup_l"], find_recovery=True, recovery_field="pup_l"):
    samps = samples.copy(deep=True)
    indices = get_eyelink_mask_idxs(samps, events, find_recovery=find_recovery, recovery_field=recovery_field)
    samps.loc[indices, mask_fields] = float('nan')
    return samps

def mask_zeros(samples, mask_fields=["pup_l"]):
    samps = samples.copy(deep=True)
    for f in mask_fields:
        samps[samps[f] == 0] = float("nan")
    return samps

def interp_zeros(samples, interp_fields=["pup_l"]):
    samps = mask_zeros(samples, mask_fields=interp_fields)
    samps = samps.interpolate(method="linear", axis=0, inplace=False)
    # since interpolate doesn't handle the start/finish, bfill the ffill to
    # take care of NaN's at the start/finish samps.
    samps.fillna(method="bfill", inplace=True)
    samps.fillna(method="ffill", inplace=True)
    return samps

def interp_eyelink_blinks(samples, events, find_recovery=True, interp_fields=["pup_l"], recovery_field="pup_l"):
    samps = mask_eyelink_blinks(samples, events, mask_fields=interp_fields, find_recovery=find_recovery, recovery_field=recovery_field)
    # inplace=True causes a crash, so for now...
    # fixed by #6284 ; will be in 0.14 release of pandas
    samps = samps.interpolate(method="linear", axis=0, inplace=False)
    return samps

def ev_row_idxs(samples, events):
    """ we expect a series of durations, with time indexes in the same unit... """
    import numpy as np
    idxs = []
    for idx, dur in events.duration.iteritems():
        idxs.extend(range(idx, int(idx+dur)))
    idxs = np.unique(idxs)
    idxs = np.intersect1d(idxs, samples.index.tolist())
    return idxs

def adjust_eyelink_recov_idxs(samples, events, z_thresh=.1, field="pup_l", window=1000, kernel_size=100):
    """ extends event endpoint until the z-scored derivative of 'field's timecourse drops below thresh

    We will try to extend *every* event passed in.

    Parameters
    ----------
    samples (list of dicts)
        A Samples object
    events (list of dicts)
        An Events object
    z_thresh (float)
        The threshold below which the z-score of the timecourse's gradient
        must fall before we'll consider the event over.
    field (string)
        The field to use.
    window (int)
        The number of indices you'll search through for z-threshold
    kernel_size (int)
        The number of indices we'll average together at each measurement. So
        what you really get with this method is the index of the first voxel
        whose gradient value, when averaged together with that of the
        n='kernel' indices after it, then z-scored, is below the given z
        threshold.
    """
    import numpy as np
    # use pandas to take rolling mean. pandas' kernel looks backwards, so we need to pull a reverse...
    dfs = np.gradient(samples[field].values)
    reversed_dfs = dfs[::-1]
    reversed_dfs_ravg = np.array(pd.rolling_mean(pd.Series(reversed_dfs),window=kernel_size, min_periods=1))
    dfs_ravg = reversed_dfs_ravg[::-1]
    dfs_ravg = np.abs((dfs_ravg-np.mean(dfs_ravg))/np.std(dfs_ravg))
    samp_count = len(samples)
    # search for drop beneath z_thresh after end index
    new_durs = []
    for idx, dur in events.duration.iteritems():
        try:
            s_pos = samples.index.get_loc(idx + dur)  - 1
            e_pos = samples.index[min(s_pos+window, samp_count-1)]
        except Exception, e:
            # can't do much about that
            s_pos = e_pos = 0
        if s_pos == e_pos:
            new_durs.append(dur)
            continue
        e_dpos = np.argmax(dfs_ravg[s_pos:e_pos] < z_thresh) # 0 if not found
        new_end = samples.index[min(s_pos + e_dpos, samp_count-1)]
        new_durs.append(new_end - idx)
    events.duration = new_durs

#-------------------------------------------------------------
# Filters

def butterworth_series(samples, fields=["pup_l"], filt_order=5, cutoff_freq=.01, inplace=False):
    """ Applies a butterworth filter to the given fields

    See documentation on scipy's butter method FMI.
    """
    # TODO: This is pretty limited right now - you'll have to tune filt_order
    # and cutoff_freq manually. In the future, it would be nice to use
    # signal.buttord (heh) to let people adjust in terms of dB loss and
    # attenuation.
    import scipy.signal as signal
    from numpy import array
    samps = samples if inplace else samples.copy(deep=True)
    B, A = signal.butter(filt_order, cutoff_freq, output="BA")
    samps[fields] = samps[fields].apply(lambda x: signal.filtfilt(B,A,x), axis=0)
    return samps


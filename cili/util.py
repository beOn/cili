import os
from multiprocessing import Pool, cpu_count
from time import sleep
import numpy as np

from models import *

# TODO: get the junk columns out of the pandas parsed datasets

ASC_SFIELDS_EYE = {
    'l':[('onset', np.int64),
         ('x_l', np.float64),
         ('y_l', np.float64),
         ('pup_l', np.float64),],
    'r':[('onset', np.int64),
         ('x_r', np.float64),
         ('y_r', np.float64),
         ('pup_r', np.float64)],
    'b':[('onset', np.int64),
         ('x_l', np.float64),
         ('y_l', np.float64),
         ('pup_l', np.float64),
         ('x_r', np.float64),
         ('y_r', np.float64),
         ('pup_r', np.float64)],}
ASC_SFIELDS_VEL = {
    'l':[('vel_x_l', np.float64),
         ('vel_y_l', np.float64),],
    'r':[('vel_x_r', np.float64),
         ('vel_y_r', np.float64),],
    'b':[('vel_x_l', np.float64),
         ('vel_y_l', np.float64),
         ('vel_x_r', np.float64),
         ('vel_y_r', np.float64)],}
ASC_SFIELDS_REZ = [
    ('res_x', np.float64),
    ('res_y', np.float64)]
ASC_SFIELDS_EXTRA = [('junk', object)]
ASC_SFIELDS_IGNORE = ['junk',]
TXT_FIELDS = {
    'LEFT_ACCELLERATION_X': np.float64,
    'LEFT_ACCELLERATION_Y': np.float64,
    'LEFT_GAZE_X': np.float64,
    'LEFT_GAZE_Y': np.float64,
    'LEFT_IN_BLINK': np.int64,
    'LEFT_IN_SACCADE': np.int64,
    'LEFT_PUPIL_SIZE': np.float64,
    'LEFT_VELOCITY_X': np.float64,
    'LEFT_VELOCITY_Y': np.float64,
    'RECORDING_SESSION_LABEL': object,
    'RIGHT_ACCELLERATION_X': np.float64,
    'RIGHT_ACCELLERATION_Y': np.float64,
    'RIGHT_GAZE_X': np.float64,
    'RIGHT_GAZE_Y': np.float64,
    'RIGHT_IN_BLINK': np.int64,
    'RIGHT_IN_SACCADE': np.int64,
    'RIGHT_PUPIL_SIZE': np.float64,
    'RIGHT_VELOCITY_X': np.float64,
    'RIGHT_VELOCITY_Y': np.float64,
    'TIMESTAMP': np.int64,}
TXT_INT_TYPES = [np.int64]
TXT_NAME_MAP = [np.int64]
ASC_EV_LINE_STARTS = [
    'MSG',
    'START',
    'END',
    'EBLINK',
    'ESACC',
    'EFIX',
    'BUTTON',
    'SAMPLES',]
ASC_EFIELDS_EVENT = {
    'MSG':[('name', object),
           ('onset', np.int64),
           ('label', object),
           ('content', object)],
    'START':[('name', object),
             ('onset', np.int64),
             ('eye', object),
             ('types', object)],
    'END':[('name', object),
           ('onset', np.int64),
           ('types', object),
           ('x_res', np.float64),
           ('y_res', np.float64)],
    'EBLINK':[('name', object),
              ('eye', object),
              ('onset', np.int64),
              ('last_onset', np.int64),
              ('duration', np.int64)],
    'ESACC':[('name', object),
             ('eye', object),
             ('onset', np.int64),
             ('last_onset', np.int64),
             ('duration', np.int64),
             ('x_start', np.float64),
             ('y_start', np.float64),
             ('x_end', np.float64),
             ('y_end', np.float64),
             ('vis_angle', np.float64),
             ('peak_velocity', np.int64)],
    'EFIX':[('name', object),
            ('eye', object),
            ('onset', np.int64),
            ('last_onset', np.int64),
            ('duration', np.int64),
            ('x_pos', np.float64),
            ('y_pos', np.float64),
            ('p_size', np.int64)],
    'BUTTON':[('name', object),
              ('onset', np.int64),
              ('b_num', np.int64),
              ('state', np.int64),],}
ASC_EFIELDS_RES = {
    'MSG':[],
    'START':[],
    'END':[],
    'EBLINK':[],
    'ESACC':[('x_res', np.float64),
             ('y_res', np.float64)],
    'EFIX':[('x_res', np.float64),
             ('y_res', np.float64)],
    'BUTTON':[],}
ASC_EV_IGNORE_COLUMNS = {
    'MSG':[],
    'START':[],
    'END':[],
    'EBLINK':[],
    'ESACC':[],
    'EFIX':[],
    'BUTTON':[],}
ASC_IRREG_EVENTS = ['MSG','START','END']
ASC_INT_TYPES = [np.int64]

def load_eyetracking_dataset(file_name):
    """ returns a dataset, and an event set"""
    root, ext = os.path.splitext(file_name)
    if ext == '.asc':
        return load_asc(file_name)
    elif ext == '.pkl':
        return load_pkl(file_name)
    elif ext in ['.txt','.tdf']:
        return load_tdf(file_name)
    else:
        raise ValueError("only .asc and .pkl files supported at the moment...")

def load_pkl(file_name):
    # if eventset doesn't load, try dataset
    event_set = CEEventSet.load_saved(file_name)
    data_set = None
    if not event_set:
        data_set = CEDataSet.load_saved(file_name)
    return data_set, event_set

def load_tdf(file_path):
    ds = read_df(file_path, delimiter='\t')
    for d in ds:
        if 'TIMESTAMP' in d.keys():
            d['onset'] = d['TIMESTAMP']
            del d['TIMESTAMP']
        for key, val in d.iteritems():
            if isinstance(val, str) and val == '.':
                # replace missing data with 0's
                d[key] = 0.0
    return cedataset_with_data(ds, None), None

def read_df(file_path, delimiter='\t',):
    # maybe use sniffer to unify this and csv reader... would be better
    import io
    enc = detect_encoding(file_path)
    f = io.open(file_path, encoding=enc)
    lines = [line.split(delimiter) for line in f]
    f.close()
    lines = [line for line in lines if len(line) != 0]
    length = len(lines[0])
    if not all([len(line) == length for line in lines]):
        raise ValueError("tdf error, not all rows had the same length in %s" % file_path)
    fieldnames = [c.encode('utf-8').strip() for c in lines[0]]
    return [dict(zip(fieldnames, [numberfy(w.encode('utf-8').strip()) for w in l])) for l in lines[1:]]

def load_asc(file_name):
    import numpy as np
    # step through the lines, accumulating samples, events and metadata
    samples = []
    sample_d_types = []
    sample_side = 'l'
    events = []
    meta = []
    start_hit = False
    record_hit = False
    with open(file_name) as f:
        lines = list(f)
        s_check = np.repeat(None,len(lines))
        sc_count = 0;
        e_check = np.repeat(None,len(lines))
        ec_count = 0;
        for line in lines:
            words = line.split()
            wCount = len(words)
            if wCount == 0:
                continue
            if not start_hit:
                if words[0] == "START":
                    start_hit = True
                    e = _event_for_asc_line(line)
                    if e is not None:
                        events.append(e)
                        continue
                meta.append(line)
                continue
            elif not record_hit:
                if wCount > 1 and words[0] == "SAMPLES":
                    # this line contains information on what the sample lines contain
                    sample_d_types = []
                    sample_side = 'l'
                    if 'VEL' in words:
                        sample_d_types.append('VEL')
                    if 'RES' in words:
                        sample_d_types.append('RES')
                    if 'LEFT' in words and not 'RIGHT' in words:
                        sample_side = 'l'
                    elif 'RIGHT' in words and not 'LEFT' in words:
                        sample_side = 'r'
                    else:
                        sample_side = 'b'
                elif ("!MODE" in words and "RECORD" in words):
                    record_hit = True
                meta.append(line)
                continue
            if words[0] == "END":
                e = _event_for_asc_line(line)
                if e is not None:
                    events.append(e)
                # there can be >1 start/end pair in a file! reset, look for another start
                start_hit = False
                record_hit = False
                continue
            # place into either sample or event bucket based on starting character
            if line[0].isdigit() and line[-4:-1] == "...":
                # s_check[sc_count] = map(lambda x: 0 if x=='.' else numberfy(x), words[:-1])
                s_check[sc_count] = tuple(words[:-1])
                sc_count += 1
            else:
                e_check[ec_count] = line.split()
                ec_count += 1
    # s_check = np.array(s_check[:sc_count].tolist()) # trim, make it re-find its shape
    s_check = s_check[:sc_count]
    e_check = e_check[:ec_count]
    # let's try some faster parsing
    samps = asc_get_samps(s_check, extra_fields=sample_d_types, side=sample_side)
    events = _asc_get_events(e_check)
    d_set = cedataset_with_data(samps, meta)
    e_set = ceeventset_with_data(events)
    return d_set, e_set

def _sample_for_asc_line(a_line, dtypes=[], side='l'):
    # return none if first word is not a number
    if isinstance(a_line, (list, tuple)):
        words = a_line
    else:
        words = a_line.split("\t")
    if (not words[0].isdigit() or len(words) < 4):
        return None
    if len(words) > 8 or (len(dtypes) == 0 and len(words) >= 7):
        side = 'b'
    values = [numberfy(w) for w in words]
    keys = []
    if side != 'b':
        keys = ['onset', 'x_'+side, 'y_'+side, 'pup_'+side]
        if 'VEL' in dtypes:
            keys.extend(['vel_x_'+side, 'vel_y_'+side])
    else:
        keys = ['onset', 'x_l', 'y_l', 'pup_l', 'x_r', 'y_r', 'pup_r']
        if 'VEL' in dtypes:
            keys.extend(['vel_x_l', 'vel_y_l', 'vel_x_r', 'vel_y_r'])
    if 'RES' in dtypes:
            keys.extend(['res_x', 'res_y'])
    return dict(zip(keys, values))

def fast_samples_for_asc_lines(lines, dtypes=[]):
    """ hopefully a better way to get samples from asc lines...
    Parameters
    ----------
    lines (iterable of tuples of strings)
        lines contianing samples
    dtypes (iterable of tuples containing)
        ('name', 'type') pairs for the sample columns
    """
    line_count = len(lines)
    field_count = len(dtypes)
    samps = np.fromiter(lines, dtype=dtypes, count=line_count)    
    # samps = np.zeros((line_count,key_count), dtype=np_dtype)
    # for i in xrange(key_count):
    #     samps[keys[i]] = [line[i] for line in lines]
    # samps = np.array(samps.tolist()) # this is slow, and stinks... how to regain numpy dimensions back?
    # samps[np.char.equal(samps, ".")] = "0" # get the damned periods out of there
    # samps = samps.astype(np.float)
    # samps = [a[0] for a in samps.tolist()]
    return samps

def _event_for_asc_line(a_line):
    import re
    # todo: parse the line into an event entry
    words = a_line.split("\t")
    if len(words) == 0:
        return None
    cmd = words[0].split(' ')[0]
    event = None
    
    # I know regex is a little ugly, but the output isn't as regular as could be... sometimes there
    # are tabs where you wouldn't want them, and sometimes where you would, they aren't.

    rg = None # regex string
    other = {} # dict of ('name',lambda), name optional
    if cmd == "MSG":
        # MSG <time> <message>
        rg = "MSG\s+(?P<onset>\S+)\s+(?P<content>(?P<name>\S+).*)"
    elif cmd == "START":
        # START <time> <eye> <types>
        rg = "(?P<name>START)\s+(?P<onset>\S+)\s+(?P<eye>\S+)(?:\s+(?P<types>.*))?"
    elif cmd == "END":
        # END <time> <types>
        rg = "(?P<name>END)\s+(?P<onset>\S+)\s+(?P<types>\S+)(?:\s+(?P<res>.*))?"
    elif cmd == "SBLINK":
        # SBLINK <eye> <stime>
        pass
    elif cmd == "EBLINK":
        # EBLINK <eye> <stime> <etime> <dur>
        rg = "E(?P<name>BLINK)\s+(?P<eye>\S+)\s+(?P<onset>\S+)\s+(?P<last_onset>\S+)\s+(?P<duration>\S+)"
        other = {'last_onset': numberfy}
    elif cmd == "SSACC":
        # SSACC <eye> <stime>
        pass
    elif cmd == "ESACC":
        # ESACC <eye> <onset> <last_onset> <duration> <x_start> <y_start> <x_end> <y_end> <vis_angle> <peak_velocity>
        rg = "E(?P<name>SACC)\s+(?P<eye>\S+)\s+(?P<onset>\S+)\s+(?P<last_onset>\S+)\s+(?P<duration>\S+)"+\
             "\s+(?P<x_start>\S+)\s+(?P<y_start>\S+)\s+(?P<x_end>\S+)\s+(?P<y_end>\S+)\s+(?P<vis_angle>\S+)"+\
             "\s+(?P<peak_velocity>\S+)"
        other = {
                'last_onset': numberfy,
                'x_start': numberfy,
                'y_start': numberfy,
                'x_end': numberfy,
                'y_end': numberfy,
                'vis_angle': numberfy,
                'peak_velocity': numberfy,
             }
    elif cmd == "SFIX":
        # SFIX <eye> <stime>
        pass
    elif cmd == "EFIX":
        # EFIX <eye> <stime> <etime> <dur> <axp> <ayp> <aps> <xr> <yr>
        # \s+(?P<>\S+)
        rg = "E(?P<name>FIX)\s+(?P<eye>\S+)\s+(?P<onset>\S+)\s+(?P<last_onset>\S+)\s+(?P<duration>\S+)"+\
             "\s+(?P<x_pos>\S+)\s+(?P<y_pos>\S+)\s+(?P<p_size>\S+)(?:\s+(?P<x_res>\S+)\s+(?P<y_res>\S+))?"
        other = {
        'last_onset': numberfy,
        'x_pos': numberfy,
        'y_pos': numberfy,
        'p_size': numberfy,
        'x_res': numberfy,
        'y_res': numberfy,
        }
    elif cmd == "BUTTON":
        # BUTTON <time> <button #> <state> 1 == pressed, 0 == released
        rg = "(?P<name>BUTTON+)\s+(?P<onset>\S+)\s+(?P<b_num>\S+)\s+(?P<state>\S+)"
        other = {'state': numberfy}

    if rg is None:
        return None

    m = re.match(rg, a_line)
    d = m.groupdict()

    # make sure we can get name, onset
    name = d.get('name', None)
    onset = numberfy(d.get('onset', None))
    if name is None or onset is None:
        return None

    # start the event dict
    event = {'name':name, 'onset':onset}

    # get duration if you can
    dur = numberfy(d.get('duration',None))
    if dur is not None:
        event.update({'duration':dur})

    # todo: add the extra values to the dict (not name,onset,duration)
    oks = other.keys()
    for k, v in d.iteritems():
        if k in ['name','onset','duration']:
            continue
        if k in oks:
            val = other[k](v)
        else:
            val = v.strip()
        event.update({k:val})

    return event

def _event_for_asc_line_new(es):
    """ takes a list of strings, tries to turn it into an event, or None"""
    if len(es) == 0:
        return None
    elif es[0] == "MSG":
        return {"onset":numberfy(es[1]), "name":es[2], "content":" ".join(es[2:])}
    elif es[0] == "START":
        e = {"onset":numberfy(es[1]), "name":"START", "eye":es[2]}
        if len(es) > 3:
            e["types"] = join(es[3:])
        return e
    elif es[0] == "END":
        e = {"onset":numberfy(es[1]), "name":"END", "types":es[2]}
        if len(es) > 3:
            e["res"] = numberfy(es[3:])
        return e
    elif es[0] == "EBLINK":

        return {"name":"BLINK", "eye":es[1], "onset":numberfy(es[2]), "last_onset":numberfy(es[3]), "duration":numberfy(es[4])}
    elif es[0] == "ESACC":
        return {"name":"SACC", "eye":es[1], "onset":numberfy(es[2]), "last_onset":numberfy(es[3]), "duration":numberfy(es[4]),
                "x_start":numberfy(es[5]), "y_start":numberfy(es[6]), "x_end":numberfy(es[7]), "y_end":numberfy(es[8]), "vis_angle":numberfy(es[9]), "peak_velocity":numberfy(es[10])}
    elif es[0] == "EFIX":
        e = {"name":"FIX", "eye":es[1], "onset":numberfy(es[2]), "last_onset":numberfy(es[3]), "duration":numberfy(es[4]),
             "x_pos":numberfy(es[5]), "y_pos":numberfy(es[6]), "p_size":numberfy(es[7])}
        if len(es) >= 10:
            e.update({"x_res":numberfy(es[8]), "y_res":numberfy(es[9])})
        return e
    elif es[0] == "BUTTON":

        return {"name":"BUTTON", "onset":numberfy(es[1]), "b_num":es[2], "state":es[3]}
    return None

def cedataset_with_data(samples, meta):
    dset = CEDataSet()
    dset.samples = samples
    dset.meta = meta
    return dset

def ceeventset_with_data(events):
    eset = CEEventSet()
    eset.events = events
    return eset

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def numberfy(s):
    n = s
    try:
        n = float(n)
        return n
    except Exception:
        return s

def detect_encoding(file_path):
    from chardet.universaldetector import UniversalDetector
    u = UniversalDetector()
    with open(file_path, 'rb') as f:
        for line in f:
            u.feed(line)
        u.close()
    result = u.result
    if result['encoding']:
        return result['encoding']
    return None

def _asc_get_events(lines):
    """ takes a list of lists, tries to turn each into an event"""
    pool = Pool()
    result = pool.map_async(_event_for_asc_line_new, lines)
    result.wait()
    es = np.array(result.get())
    es = es[np.argwhere(es)] # take out None
    es = [a[0] for a in es.tolist()]
    return es

class LineRunner(object):
    def __init__(self, side=None, extra_fields=None, dtypes=None, go_fast=True):
        self.dtypes = dtypes
        self.side = side
        self.go_fast = go_fast
        self.extra_fields = extra_fields
    def __call__(self, lines):
        """ expects lines to be an numpy array"""
        if self.go_fast:
            return fast_samples_for_asc_lines(lines, dtypes=self.dtypes)
        else:
            return [_sample_for_asc_line(line, dtypes=self.extra_fields, side=self.side) for line in lines]

def asc_get_samps(lines, extra_fields=[], side='l'):
    """ takes a list of lists, tries to turn each into a sample"""
    from math import ceil
    if len(lines) == 0:
        return []
    # okay, let's try avoiding loops...
    # ss = np.array(lines)
    ss = lines
    ls = np.unique([len(a) for a in ss])
    pool = Pool()
    chunk_size = int(ceil(len(ss)/min(cpu_count(),1)))
    line_chunks = [ss[i:min(i+chunk_size,len(ss))] for i in xrange(0, len(ss), chunk_size)]
    if len(ls) == 1:
        # pre-select the dtypes
        dtypes = ASC_SFIELDS_EYE[side]
        if 'VEl' in extra_fields:
            dtypes.extend(ASC_SFIELDS_VEL[side])
        if 'RES' in extra_fields:
            dtypes.extend(ASC_SFIELDS_REZ)
        ss = fast_samples_for_asc_lines(ss, dtypes=dtypes)
        # result = pool.map_async(LineRunner(dtypes=dtypes, go_fast=True), line_chunks)
        # result.wait()
        # ss = result.get()
        np.hstack(ss)
    else:
        # ugh we suck! this is still slow...
        result = pool.map_async(LineRunner(side=side, extra_fields=extra_fields, go_fast=False), line_chunks)
        result.wait()
        ss = np.hstack(result.get())
        ss = ss[np.argwhere(ss)].flatten().tolist() # take out None
    return ss

def pandas_df_from_txt(file_path):
    import pandas as pd
    import io
    # first we'll just grab everything as objects...
    # then we'll get the fields, figure out the dtypes, and do conversions
    # accordingly. It would be nice if the dtypes would work in read_csv, but
    # so far no luck...
    df = pd.read_csv(file_path, sep="\t", index_col="TIMESTAMP", low_memory=False, na_values=["."],)
    fields = [str(x) for x in df.dtypes.keys()]
    dtypes = dict([(d, object) for d in fields if not d in TXT_FIELDS.keys()])
    dtypes.update(dict([(k, v) for k, v in TXT_FIELDS.iteritems() if k in fields]))
    nums = [k for k, v in dtypes.iteritems() if v not in [object]]
    ints = [k for k in nums if dtypes[k] in TXT_INT_TYPES]
    df[nums] = df[nums].convert_objects(convert_numeric=True)
    df[ints] = df[ints].astype(np.int64)
    # rename TIMESTAMP to "onset" for consistency, and make all columns lower
    fields = [f.lower() for f in fields]
    df.columns = fields
    df.index.name = "onset"
    return Samples.from_pd_obj(df)

def pandas_dfs_from_asc(file_path):
    # collect lines for each event type (including samples)
    e_lines = dict([(k,[]) for k in ASC_EV_LINE_STARTS])
    s_lines = []
    with open(file_path, "r") as f:
        for line in f:
            if len(line) == 0:
                continue
            if line[0].isdigit():
                s_lines.append(line)
                continue
            for k in ASC_EV_LINE_STARTS:
                if line.startswith(k):
                    e_lines[k].append(line)
                    break
    # determine column names, dtypes
    if not len(e_lines["SAMPLES"]) > 0:
        raise ValueError("Could not find samples line in .asc file.")
    side, has_vel, has_res = info_from_asc_samples_line(e_lines["SAMPLES"][0])
    samp_dtypes = build_asc_samp_dtypes(side, has_vel, has_res)
    ev_names = [k for k in ASC_EV_LINE_STARTS if not k in ["SAMPLES"]]
    ev_dtypes = dict([(ev_name, build_asc_ev_dtypes(ev_name, side, has_vel, has_res)) for ev_name in ev_names])
    # get a df for the samples
    samp_df = pandas_df_from_lines(s_lines, samp_dtypes, ASC_SFIELDS_IGNORE)
    # handle event types that need to have their lines preprocessed...
    for ev_name in ASC_IRREG_EVENTS:
        if not ev_name in e_lines:
            continue
        e_lines[ev_name] = prep_irreg_asc_event_lines(e_lines[ev_name], ev_name)
    # get a df for each event type
    ev_dfs = dict([(ev_name,
                    pandas_df_from_lines(e_lines[ev_name],
                                         ev_dtypes[ev_name],
                                         ASC_EV_IGNORE_COLUMNS[ev_name]))
                    for ev_name in ev_names if len(e_lines[ev_name]) > 0])

    # TODO add omitting ASC_EV_IGNORE_COLUMNS[ev_name]
    return Samples.from_pd_obj(samp_df), Events.from_dict(ev_dfs)

def pandas_df_from_lines(csv_lines, dtypes, ignore):
    import pandas as pd
    import cStringIO
    c = cStringIO.StringIO("".join(csv_lines))
    fields, dts = zip(*dtypes)
    # use_names = [n for n in fields if not n in ignore]
    df = pd.read_csv(c,
                     delim_whitespace=True,
                     index_col=["onset"],
                     low_memory=False,
                     na_values=["."],
                     names=fields,
                     header=0,
                     error_bad_lines=False,
                     # usecols=use_names,
                     warn_bad_lines=False,)
    nums = [d[0] for d in dtypes if d[1] not in [object] and d[0] not in ['onset']]
    ints = [d[0] for d in dtypes if d[1] in ASC_INT_TYPES and d[0] not in ['onset']]
    df[nums] = df[nums].convert_objects(convert_numeric=True)
    df[ints] = df[ints].astype(np.int64)
    for ig in ignore:
        del df[ig]
    return df

def prep_irreg_asc_event_lines(lines, ev_name):
    """ uses quotes to force annoying events into usable chunks
    use sparingly - not super fast right now
    """
    new_lines = []
    if ev_name == 'MSG':
        # name, onset, label, content
        # easy - just break content into a third, quoted column
        for line in lines:
            l = line.split()
            lab = l[2] if len(l) > 2 else '.'
            cont = ' '.join(l[3:]) if len(l) > 3 else '.'
            nl = '%s\t%s\t"%s"\t"%s"\n' % (l[0], l[1], lab, cont)
            new_lines.append(nl)
    elif ev_name == 'START':
        # name, onset, eye, then one or two types
        for line in lines:
            l = line.split()
            new_lines.append('%s\t%s\t%s\t"%s"\n' % (l[0],l[1],l[2],', '.join(l[3:])))
    elif ev_name == 'END':
        # name, onset, maybe a list of types, 'RES', x_res, y_res
        # we'll take out the "RES" here
        for line in lines:
            l = line.split()
            types = ' '.join(l[2:-3]) if len(l) > 5 else '.'
            x_res = l[-2]
            y_res = l[-1]
            new_lines.append('%s\t%s\t"%s"\t%s\t%s\n' % (l[0], l[1], types, l[-2], l[-1]))
    else:
        new_lines = lines
    return new_lines

def build_asc_samp_dtypes(side, has_vel, has_res):
    dtypes = list(ASC_SFIELDS_EYE[side])
    if has_vel:
        dtypes.extend(ASC_SFIELDS_VEL[side])
    if has_res:
        dtypes.extend(ASC_SFIELDS_REZ)
    dtypes.extend(ASC_SFIELDS_EXTRA)
    return dtypes

def build_asc_ev_dtypes(ev_name, side, has_vel, has_res):
    dtypes = list(ASC_EFIELDS_EVENT.get(ev_name,[]))
    if has_res:
        dtypes.extend(ASC_EFIELDS_RES.get(ev_name,[]))
    return dtypes if dtypes else None

def info_from_asc_samples_line(line_txt):
    """ gets sample info from asc SAMPLE lines

    returns
    -------
    side (str)
        'l', 'r', or 'b'
    has_velocity (bool)
        True if velocity information is included in samples
    has_resolution (bool)
        True is resolution information is included in samples
    """
    words = line_txt.split()
    # this line contains information on what the sample lines contain
    has_velocity = "VEL" in words
    has_resolution = "RES" in words
    sample_side = 'b'
    if 'LEFT' in words and not 'RIGHT' in words:
        sample_side = 'l'
    elif 'RIGHT' in words and not 'LEFT' in words:
        sample_side = 'r'
    return sample_side, has_velocity, has_resolution

def percentile_bucket(vals, bucket_size=10, scale=1.0, shift=0.0):
    """ returns percentile scores for each value
    Parameters
    ----------
    bucket_size (float)
        The size of each bucket, in percentile points 0-100. Actual bucket
        cutoffs are calculated with numpy.arange(), so if 100 isn't divisible
        by bucket_size, your top bucket will be small.
    scale (float)
        All values will be multiplied by this number after bucketing.
    shift (float)
        All values will have this added to them after scaling.
    """
    from scipy.stats import scoreatpercentile as sp
    import numpy as np
    from bisect import bisect_left
    percs = np.concatenate([np.arange(bucket_size,100,bucket_size), [100]]) # arange to get the percentiles
    cuts = [sp(vals, p) for p in percs] # to get the cutoff score for each percentile
    new_list = np.array([bisect_left(cuts, val)+1 for val in vals]) * scale + shift # turn values into bucket numbers... +1 since we want 1-indexed buckets
    return new_list

def ensure_dir(dir_path, overwrite=False):
    from shutil import rmtree
    from os.path import isdir, exists
    from os import makedirs
    if exists(dir_path):
        if not isdir(dir_path):
            raise ValueError("%s is a file..." % dir_path)
        if overwrite:
            rmtree(dir_path)
    if not exists(dir_path):
        makedirs(dir_path)

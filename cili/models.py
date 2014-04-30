import pandas as pd
import pandas.io.pytables as pt
from pandas.compat import u_safe as u, string_types, isidentifier

class SaveMixin(object):
    """ Bakes in some save settings for NDFrame subclasses
    
    You can still use the pandas methods, but for quick saving and loading
    this mixin provides some setting you might want to reuse.
    """
    def __init__(self, *args, **kwargs):
        super(SaveMixin, self).__init__(*args, **kwargs)

    def save(self, save_path):
        # explore slower but more flexible "table" later
        initialize_hdf5()
        self.to_hdf(save_path, "obj", format="fixed")

    @classmethod
    def load_saved(cls, save_path):
        from pandas import read_hdf
        initialize_hdf5()
        obj = read_hdf(save_path, "obj")
        obj2 = cls.from_pd_obj(obj)
        return obj2

    @classmethod
    def from_pd_obj(cls, pd_obj):
        return cls(pd_obj._data.copy()).__finalize__(pd_obj)

class Samples(SaveMixin, pd.DataFrame):
    """Pandas DataFrame subclas for representing eye tracking timeseries data.

    Indexes may be hierarchical.
    """
    def __init__(self, *args, **kwargs):
        super(Samples, self).__init__(*args, **kwargs)
        
class Events(object):
    """Pandas Panel-like object that gives you access to DataFrames via standard accessors.

    Also implements save and load_saved. And init form dict. Otherwise, it's
    not much like a Panel where it counts.

    One advantage to avoiding Panel is that this lets us bundle all of our
    event DataFrames into one object without loding the column dtypes during
    merge.

    Right now, the best way way to make one of these is to use Events.from_dict().
    """
    def __init__(self, *args, **kwargs):
        super(Events, self).__init__(*args, **kwargs)
        self.dframes = {}

    def save(self, save_path):
        s = pt.HDFStore(save_path)
        for k in self.dframes.keys():
            s[k] = self.dframes[k]
        s.close()

    @classmethod
    def load_saved(cls, save_path):
        # TODO: read in all of the dataframes stored on the object
        obj = cls()
        s = pt.HDFStore(save_path)
        obj.dframes = dict([(k[1:],s[k]) for k in s.keys()])
        s.close()
        return obj

    @classmethod
    def from_dict(cls, the_d):
        """ Returns an Events instance containing the given DataFrames
        
        Parameters
        ----------
        the_d (dict)
            A dictionary with event names for keys, and DataFrames for values.
        """
        obj = cls()
        obj.dframes = the_d
        return obj

    @classmethod
    def from_list_of_dicts(cls, events_list):
        """ Returns an Events instance built using the given dicts.

        Parameters
        ----------
        events_list (list of dictionaries, or path to pickled file)
            This can either be an actual list, or the location of a pickled
            list. We'll group the events by 'name' into DataFrames, then pass
            those to from_dict. Note that only field that appear in every
            dictionary with a given name will make it through to the
            corresponding DataFrame.
        """
        import os
        from cPickle import load as pload
        if isinstance(events_list, str):
            if not os.path.isfile(events_list):
                raise ValueError("Could not find file %s" % events_list)
            evl = pload(events_list)
        else:
            evl = events_list
        evd = {}
        for ev in evl:
            if not "name" in ev:
                continue
            if not ev["name"] in evd:
                evd[ev["name"]] = [ev]
            else:
                evd[ev["name"]].append(ev)
        for k in evd.keys():
            evd[k] = pd.DataFrame(evd[k])
        return cls.from_dict(evd)

    def _local_dir(self):
        """ add the string-like attributes from the info_axis """
        return [c for c in self.dframes.keys()
            if isinstance(c, string_types) and isidentifier(c)]

    def __dir__(self):
        """
        Provide method name lookup and completion
        Only provide 'public' methods
        """
        return list(sorted(list(set(dir(type(self)) + self._local_dir()))))

    def __getattr__(self, name):
        """Similar to Panel behavior, but with filtered accessors.

        Rows will be filtered by any(notnull(columns)), then columns filtered
        by any(notnull(rows)). This way, you should quickly be able to get
        ahold of and inspect the set of events and values you're interested
        in.
        """
        if name in self.dframes:
            return self.dframes[name]
        raise AttributeError("'%s' object has no attribute '%s'" %
                             (type(self).__name__, name))

def initialize_hdf5():
    pt._TYPE_MAP.update({Events:u('wide'), Samples:u('frame'),})
    pt._AXES_MAP.update({Events:[1, 2], Samples:[0],})

# PANDAS objects:
# Samples - CEDataSet - DataFrame
# Events - CEEventSet - Panel
# Redundant of Samples: CETimeseriesCollection - Hierarchically Indexed DataFrame
# CEAbstractObj -


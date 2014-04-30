warning
=======
This documentation is under development.

examples
========

TODO:
-----

* update to include the following
** hdf5 must now be installed!

first exploration
-----------------

Let's walk through generating a timeseries collection given some samples and
some events. Imagine that you have two kinds of events that you're interested
in collecting samples for, called 'incent_precue' and 'non_incent_precue.' And
let's say that for each event, you want to collect the two samples prior to
event onset for use as a baseline, and the five after event onset as the main
timeseries. Then, just for fun, let's say you want to map each sample in the
main timeseries to its percent deviation from the mean of the baseline
timeseries. (note... usually you would collect many more timepoints, but for
the sake of this demo we'll keep the numbers low.)

First things first - let's pretend you load up an asc file:

```python
data_set, event_set = load_eyetracking_dataset('/path/to/the/file.asc')
```

Before we do anything with the data, let's do some quick blink correction.
This involves using a CEDataMapper, like so:

```python
dm = CEInterpolator()
dm.interp_type = CE_INTERPOLATION_LINEAR_EYELINK_BLINK
dm.interp_event_names = ['BLINK'] # the name of the blink event
dm.interp_sample_fields = ['pup_l'] # apply only to the left pupil val
data_set = dm.apply_to(data_set, event_set=event_set)[0]
```

Amazing. Now, let's specify a couple of rules for extracting the timeseries
that we're looking for.

```python
ranges = [{'offset_to_first':-2, 'offset_to_last':-1},
          {'offset_to_first':0, 'offset_to_last':5-1}]
range_defs = {'incent_precue': {'incentive': ranges},
              'non_incent_precue': {'non-incentive': ranges}}
```

In the future, we might elaborate the extraction rules, but for now this is
how it works: in the dictionary range_defs, the keys are event names that we
expect to find in the event set we're using. In this case, I expect to find
precue events for incentive and non-incentive trials. Those events will be
called incent_precue, and non_incent_precue respectively. For every event
named incent_precue, we'll extract the samples specified by 'ranges.' The fact
that there are two entries in ranges meand that for each event, we will
extract two timeseries. Each pair of timeseries will be added to a collection
of pairs under the name 'incentive'. Let's see this in action:

```python
ts = CETimeseriesCollection()
ts.extract_from_dataset(data_set, event_set, range_defs)
```

Note: to explore the event names in your own data, open python, load one of
your asc files using load_eyetracking_dataset, and check out the returned
event_set's .events property. It'll be a list of parsed events.

Let's just take a quick look at a very small bit of the data...
```
In [81]: ts.series['incentive'][2:4]
Out[83]: 
[[[{'onset': 4312240, 'pup_l': 6699.0, 'x_l': 491.7, 'y_l': 370.3},
   {'onset': 4312241, 'pup_l': 6709.0, 'x_l': 492.3, 'y_l': 368.9}],
  [{'onset': 4312242, 'pup_l': 6710.0, 'x_l': 492.2, 'y_l': 369.1},
   {'onset': 4312243, 'pup_l': 6711.0, 'x_l': 492.2, 'y_l': 370.5},
   {'onset': 4312244, 'pup_l': 6708.0, 'x_l': 492.2, 'y_l': 372.0},
   {'onset': 4312245, 'pup_l': 6704.0, 'x_l': 491.6, 'y_l': 371.1},
   {'onset': 4312246, 'pup_l': 6701.0, 'x_l': 491.1, 'y_l': 370.2}]],
 [[{'onset': 4324821, 'pup_l': 6848.0, 'x_l': 495.4, 'y_l': 405.6},
   {'onset': 4324822, 'pup_l': 6838.0, 'x_l': 495.8, 'y_l': 405.4}],
  [{'onset': 4324823, 'pup_l': 6835.0, 'x_l': 496.3, 'y_l': 405.4},
   {'onset': 4324824, 'pup_l': 6843.0, 'x_l': 496.8, 'y_l': 405.4},
   {'onset': 4324825, 'pup_l': 6853.0, 'x_l': 496.6, 'y_l': 405.2},
   {'onset': 4324826, 'pup_l': 6864.0, 'x_l': 496.2, 'y_l': 405.0},
   {'onset': 4324827, 'pup_l': 6873.0, 'x_l': 495.7, 'y_l': 404.7}]]]
```

So, we can see here that in the 'incentive' set, for each entry, we have two
arrays. Each of those arrays contains samples - the first one has the two
prior to event onset, and the second one has the five starting with event
onset.

Now, let's recalculate the pup_l field in each of the entries in each of the
second arrays as a percentage of deviation from the average of all pup_l
values from the first array (the baseline timeseries).

```python
ts.ts_map(map_within_samples,
          map_subfunc=perc_dev_from_avg,
          map_fields=['pup_l'],
          series_names=['incentive', 'non-incentive'])
```

Cool, let's take a look at the values:

```
    In [90]: ts.series['incentive'][2:4]
    Out[90]: 
    [[[{'pup_l': 0.0008949880668257757},
       {'pup_l': 0.0010441527446300716},
       {'pup_l': 0.0005966587112171838},
       {'pup_l': 0.0},
       {'pup_l': -0.00044749403341288785}]],
     [[{'pup_l': -0.0011690778898144089},
       {'pup_l': 0.0},
       {'pup_l': 0.0014613473622680112},
       {'pup_l': 0.0030688294607628232},
       {'pup_l': 0.004384042086804033}]]]
```

Please note that while in this example we've lost values from all fields
except thsoe passed into 'map_fields,' since the creation of this document
that issue has been addressed. Now, if the map_subfunc does not change the
shape of your data too dramatically, ccp_eye will preserve the un-mapped
fields, so in this case the samples would still have values for 'onset,'
'pup_l,' 'x_l,' and 'y_l.'

Now if we wanted to extract an average from each of these arrays, we
could use the average_range subfunc. This should look kindof familiar after
the last thing we did.

```python
ts.ts_map(map_within_samples,
          map_subfunc=average_range,
          map_fields=['pup_l'],
          series_names=['incentive', 'non-incentive'],
          ranges={'incentive':[0, 4], 'non-incentive':[0, 4]})
```

Now we have the following:

```
In [28]: ts.series['incentive'][:10]
Out[28]: 
[[[{'pup_l': [0.0]}]],
 [[{'pup_l': [0.0001463378941977025]}]],
 [[{'pup_l': [0.00063394988066825769]}]],
 [[{'pup_l': [0.00084027473330410636]}]],
 [[{'pup_l': [-0.00072400810889081948]}]],
 [[{'pup_l': [0.0013949049262168708]}]],
 [[{'pup_l': [0.00029587987277165473]}]],
 [[{'pup_l': [-0.0010700527892709374]}]],
 [[{'pup_l': [0.0012714327230456263]}]],
 [[{'pup_l': [0.00041635124905374716]}]]]
```

It looks like that first 0 is due to the combination of a blink at the
beginning of the first trial, and a very short timeseries extraction range.
The blink interpolation will have extended the first available value all the
way to the 0'th timepoint, so we can assume that all of the timepoints we've
sampled here have exactly the same value.

Anyway, if we just want a straight list of the values above, we can use a
numpy array to make things a little faster:

```python
import numpy as np
samps = np.array(ts.series['incentive']).flatten()
avgs = [samp['pup_l'][0] for samp in samps]
```

Now we should just have a normal list:

```
In [45]: avgs[:10]
Out[45]: 
[0.0,
 0.0001463378941977025,
 0.00063394988066825769,
 0.00084027473330410636,
 -0.00072400810889081948,
 0.0013949049262168708,
 0.00029587987277165473,
 -0.0010700527892709374,
 0.0012714327230456263,
 0.00041635124905374716]
```

exporting timeseries collections to txt
---------------------------------------

Let's say you have four events, ev_a, ev_b, ev_c and ev_d, and you'd like to
load up a saved data set and a saved event set containing the event
definitions, then extract the time series data for those events, then export
those data points to a tab delimited file with a column indicating which event
the samples came from. Let's also go ahead and say that these four event types
are the only kinds of events in your events file.

Well, first I'd say, "You should be using HDF5." And we should - for larger
data sets - but since that isn't supported yet, let's just trudge on.

Here's what it looks like:

```python

from ccp_eye.CEDataSet import *
from ccp_eye.CETimeseriesCollection import *
from ccp_eye.CEEventSet import *
ds = CEDataSet.load_saved('/path/to/dataset/data_1.pkl')
es = CEEventSet.load_saved('/path/to/eventset/event_1.pkl')
ts = CETimeseriesCollection()
ts.extract_from_dataset(ds, es, None)
ts.export_to_tdf('/path/to/desired/out_dir/output.txt', include_collection_names=True)

```

That's it - we should now be able to read output.txt into anything that can
parse a tab-delimited file. The 'include_collection_names' argument is what
adds the column indicating which event each sample came from. Also, notice
that the third argument to extract_from_dataset is None. This causes the
extraction method to automatically extract every event in the event set, using
the event's onset and duration to determine which samples to grab. This is the
equivalent of specifying a range_defs dict with a key for every event name,
and with each entry being a dict consisting of one key (the event name) and
the value None.

This example didn't include any blink fixing or filtering, as those steps
would usually be done prior to timeseries extraction, but if you combine this
with the example above it should be clear how you'd grab some data from an
eyetracking file, do some blink correction, select samples for some events,
then export those to a TDF.


merging event sets
------------------

Damn. We messed up - we have good metadata about our events in the edat file,
but those files don't have accurate timing info. In the asc file, we have
great timing data but little else.

Fear not - the illusion of competence shall not be undone. As long as we can
extract event sets of equal length and order from the two files (using the
varys module, perhaps?), we can merge them together like so:

```python
from ccp_eye.CEEventSet import *
es = CEEventSet.merge('/path/to/file1.pkl', '/path/to/file2')
# or...
es = CEEventSet.merge(event_set_1, event_set_2, event_set_3)
# or...
es = CEEventSet.merge([{},{},{}], [{},{},{}])
# or mix and match. then save!
es.save('/path/to/new_file.pkl')
```

You can merge any number of event sets as long as they have the same length
and order. You can pass the method a heterogeneous mix of CEEventSets, arrays
of dictionaries, and paths to any file the CEEventSet.load_saved() can read.


passing values from events to samples
-------------------------------------

If you're exporting your extracted timeseries data for analysis in other
languages, you might want to label each sample with specific properties from
the event that selected it. For example, if you wanted to be able to tell by
looking at a single sample whether it came from a trial that was answered
correctly or incorrectly, you would typically have that information encoded in
your event definition - wouldn't it be great if you could add an extra
property, 'acc,' to all of the samples collected from that event? Yes, it is
great. Let's look at how we'd do it.

For starters, we need an event that contains a value for 'acc,' then we'll use
CETimeseriesCollection's inherit_event_fields argument to pass that value from
the event to the extracted samples. Let's assume we already have a dataset
loaded up, called 'ds.'

```python
events = [{'name':'some_event', 'onset':47000, 'duration':5000, 'acc':1}]
ts = CETimeseriesCollection()
ts.extract_from_dataset(ds, es, None, inherit_event_fields=['acc'])
```

That's it - all of the samples in the timeseries collection will now have an
'acc' field.

exporting datasets to tdf
-------------------------

Let's say you just want to load a dataset, then save it to TDF.

```python
data_set, event_set = load_eyetracking_dataset('/path/to/the/file.asc')
data_set.export_to_tdf('/path/to/save/file.txt')
```

And you're done.

maybe later...
--------------
- [ ] example of averaging across samples
- [ ] example of graphing
- [ ] duck type all lists and tuples
- [ ] formalize secs/ms and frequency in dataset

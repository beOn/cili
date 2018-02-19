[![DOI](https://zenodo.org/badge/19322912.svg)](https://zenodo.org/badge/latestdoi/19322912)

[![Build Status](https://travis-ci.org/beOn/cili.svg?branch=master)](https://travis-ci.org/beOn/cili)

cili
====

Cili is meant to reduce the overhead of basic eyetracking data processing.
While it sets the stage for more advanced work by providing data in the form
of pandas DataFrames, manipulating the contents of those DataFrames works like
manipulating any other pandas DataFrame, so that's where cili stops - we
leave it to the user to learn to work with pandas. If you're going to be
dealing  with eyetracking data in python, you'll be glad you did.

At the moment, we support EyeLink data only. We'd be happy to support other
manufacturers, but don't have the data on hand to do so. If you have the data,
and would like us to support something in particular, please drop us a message
at https://github.com/beOn/cili/issues, or add it yourself and submit a pull
request.

Please note that this is an alpha release.

Installation
============

Before installing cili, you'll need to install numpy. The rest of the
dependencies should install when you install cili, but numpy is special. Then
you can install cili using pip or easy_install:

```
pip install numpy
pip install cili
```

Or grab the latest development version from https://github.com/beOn/cili, and
install using setup.py.

Examples
========

Parsing EyeLink Data
--------------------

Cili can parse samples and events from EyeLink .asc files, and samples from
EyeLink Experiment Builder .txt files. You can use the same method for both,
but need to ignore the second returned value if you're using a .txt.

```python
from cili.util import *
# using .asc
samps, events = load_eyelink_dataset("/some/file.asc")
# using .txt
samps, _ = load_eyelink_dataset("/some/file.txt")
```

Handling Blinks and Missing Data
--------------------------------

Try as we might to recruit cooperative subjects, chances are your data is full
of blinks; try as we might to set up the eyetracker correctly, chances are
there's some signal dropout. Cili provides a couple of convenient methods for
deriving values missing for either of these reasons using linear
interpolation.

Handling blinks in EyeLink data doesn't seem to tricky at first blush, but
there are a couple of subtleties and related options to be aware of.

First, EyeLink embeds every blink event within a saccade, and recommends that
if you plan to scrub out blinks, you should also scrub out saccades containing
blinks. Cili does this automatically when you call mask_eyelink_blinks().

Second, EyeLink's blink marking algorithm is a little too aggressive for my
taste when it comes to declaring a blink's end time. Even when we interpolate
over the containing saccade, the reported pupil size for several dozen
milliseconds after the interpolated range often contains absurdly low values,
with an absurdly high slope. This can add a lot of noise to your data, and I
can't imagine anyone arguing that these values have any bearing on the eye's
real state. So cili will optionally creep the blink recovery time forward by
looking for the first point within 1000ms where the 100-sample rolling average
of the z-scored derivative of the pupil timecourse drops to within .1 of the
entire timecourse's average. This method has worked pretty well for us, but
there's still room for improvement.

All non-blink signal dropout will be recorded as 0s. We don't make any
adjustments to the dropout onset/offset times, but otherwise these values get
handled in much the same way as blinks. But you should always clean dropout
after cleaning blinks, otherwise the blink recovery index method described
above won't work.

So, let's roll up our sleeves and clean some data!

```python
from cili.util import *
from cili.cleanup import *
samps, events = load_eyelink_dataset("/some/file.asc")
samps = interp_eyelink_blinks(samps, events, interp_fields=["pup_l"])
samps = interp_zeros(samps, interp_fields=["pup_l"])
```

Well that was kindof anticlimactic.

Note that if you collect the right pupil, "pup_l" should be changed to
"pup_r", and if you collect both eyes, you'll want to include both pup_l and
pup_r in interp_fields. EyeLink's Experiment Builder calls the same values
"right_pupil_size" and "left_pupil_size" in the .txt files it generates, so if
your samples came from a Experiment Builder .txt, use those names instead.

Check the documentation on these methods FMI.

Smoothing Data
--------------

If you look closely at EyeLink data, you'll probably notice a little high
frequency noise. This can be a little problematic in several circumstances. To
deal with it, cili provides a butterworth filter function with default
settings based on previously published pupillometry studies. You can modify
the order and cutoff frequency of the filter if you like, but the basic usage
looks like this:

```python
samps = butterworth_series(samps, fields=["pup_l"])
```

FMI, check out the documentation on butterworth_series.

Events from a List of Dicts
---------------------------

Sometimes you are interested in events recorded using something other than
EyeLink software. For those crazy times, if you can turn that data into a list
of dicts, each containing a name, onset and duration, then it's pretty easy to
create a cili Events object. Assuming you already have your list of dicts, and
that it's called ```list_o_dicts```, then all you do is:

```python
from cili.models import Events
events = Events.from_list_of_dicts(list_o_dicts)
```

Extracting Event-based Ranges
-----------------------------

To my mind, this is where things start to get interesting. In many eye
tracking and pupillometry studies, the goal is to examine a collection of
sample ranges surrounding certain events. So cili provides a method for
extracting sample ranges based on event timing, returning a DataFrame with a
MultiIndex (event #, sample #).

Suppose you were interested in the 10 seconds following every event in some
Events object, called "events," and you have a 1kHz Samples object, samps. To
extract this range for every event in events, you would:

```python
from cili.extract import extract_event_ranges
ranges = extract_event_ranges(samps, events, end_offset=10000)
```

Often, pupillometric sample ranges will be transformed into a % deviation from
baseline measure, where the baseline is an average of some small range
immediately preceding the range of interest. Continuing the example above,
let's extract baseline measures for each of the events, then divide the ranges
of interest by the baselines for the field "pup_r":

```python
baselines = extract_event_ranges(samps, events, start_offset=-100, end_offset=-1).mean(level=0)
ranges.pup_r = (ranges.pup_r / baselines.pup_r - 1).values
```

Not so painful! For more info on range extraction, check out the documentation
on extract_event_ranges. To work with the returned data effectively, You'll
probably also want to take a minute to learn about pandas MultiIndex objects.

Borrowing Event Attributes
--------------------------

If your events each have a field, let's say "subject", and you'd like to
insert each event's value for that field into every row of the corresponding
range under a column of the same name, you can "borrow" event attributes using
borrow_attributes, like so:

```python
ranges = extract_event_ranges(samps, events, end_offset=10000, borrow_attributes=["subject"])
```

Saving and Loading
------------------

If you keep reading and writing large .txt files, you'll die young. Or at
least having spent too much of your time waiting for .txt files to be read or
written. So cili uses hdf5 to speed things up. To use this, you'll need to
install h5py and its dependencies, as documented at
http://docs.h5py.org/en/latest/build.html.

Once that's done, saving and loading Samples and Events objects is pretty
easy. It works the same way in both cases, so we'll just work with samples
below:

```python
from cili.models import Samples
samps.save("some_filename.hdf")
samps_2 = Samples.load_saved("some_filename.hdf")
```

Exporting to .txt
-----------------

If you have to export samples or extracted ranges to a .txt file, fine. Ok. We
understand.

Luckily, pandas datasets already include a function for writing csv files, any
several other formats as well (check their documentation for the complete
list: http://pandas.pydata.org/pandas-docs/stable/io.html). For example, to
create a tab delimited .txt file:

```python
samps.to_csv("some_filename.txt", sep="\t")
```

To create a Zamboni delimited .txt file, just set sep to "Zamboni".

Checking For Signal Dropout
---------------------------

Sometimes, for one reason or another, eyetracking sessions can go pretty
poorly. Usually, this means that there's a high level of signal dropout due to
blinks, or the tracker losing track of the eye. One way to check for this is
to see what percentage of the timeline's pupil value(s) were recorded as 0.
Cili's util.py offers a convenient way to check the dropout rate for all of
the .asc files in a directory, like so:

```python
util.py --dropout -d /path/to/dir/containing/ascf_iles/
```

Reporting Bugs, Requesting Features
===================================

Submit all bug reports and feature requests using the github ticketing system:
https://github.com/beOn/cili/issues

Please make an effort to provide high quality bug reports. If we get one that
just says, "sample range extraction is broken," we'll probably trash it
without a second look, because the submitter is probably the kind of person
who saps energy from everything they touch.

A good bug report should include three things:

1. Steps to reproduce the bug
2. Expected result
3. Actual result

The goal is to give the developers the ability to recreate the bug before
their own eyes. If you can give us that, we'll take a very close look.

Citation
========
If you use cili in your work, please reference:

Acland BT, Braver TS (2014). Cili (v0.5.4) [Software] Available from http://doi.org/10.5281/zenodo.48843. doi:10.5281/zenodo.48843


Why Cili?
=========

Because, like the mighty ciliary muscles, it brings your eye data into focus.

Thanks, credit to Dr. Todd Braver and the Cognitive Control and Psychopathology
Lab at Washington University in St Louis, where this code was developed.


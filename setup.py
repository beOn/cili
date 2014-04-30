#!/usr/bin/env python

import os
from setuptools import setup

def read(fname):
    fpath = os.path.join(os.path.dirname(__file__), fname)
    if os.path.exists(fpath):
      return open(fpath).read()
    return ""

setup(name='cili',
      version='0.5.0',
      author='Ben Acland',
      author_email='benacland@gmail.com',
      description='Eyetracking data tools based on pandas',
      license='BSD',
      keywords='eyetracking pupillometry eyelink',
      url='https://github.com/beOn/cili',
      install_requires=['scipy','chardet','numexpr','tables','pandas'],
      packages=['cili'],
      long_description=read('README.md'),
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
      ],
)

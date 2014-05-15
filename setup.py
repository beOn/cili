#!/usr/bin/env python

import os
from setuptools import setup

def get_readme():
    md_path = os.path.join(os.path.dirname(__file__), "README.md")
    txt_path = os.path.join(os.path.dirname(__file__), "README.txt")

    if os.path.exists(txt_path):
        d = open(txt_path).read()
    elif os.path.exists(md_path):
        d = open(md_path).read()
    else:
        d = ""
    return d

setup(name='cili',
      version='0.5.2',
      author='Ben Acland',
      author_email='benacland@gmail.com',
      description='Eyetracking data tools based on pandas',
      license='BSD',
      keywords='eyetracking pupillometry eyelink',
      url='https://github.com/beOn/cili',
      install_requires=['scipy','numexpr','tables','pandas'],
      packages=['cili'],
      long_description=get_readme(),
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
      ],
)

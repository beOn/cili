import os

# test data paths
DATA_DIR = os.path.join(os.getcwd(), 'tests', 'data')
paths = ['bino250', 'bino500', 'bino1000', 'binoRemote250',
         'binoRemote500', 'mono250', 'mono500', 'mono1000',
         'mono2000', 'monoRemote250', 'monoRemote500']
paths = dict([(p, os.path.join(DATA_DIR, p + '.asc')) for p in paths])

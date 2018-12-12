"""
  Common functions and modules used by
  the rest of the python code.
"""

import time
import copyreg
import itertools
import operator

from numpy import *
import cmath

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.colors

import scipy.integrate
import scipy.interpolate
import scipy.special
import scipy.misc
import scipy.linalg

from semiconductor import *
from job_api import JobAPI

initializeMPFR_GSL()


def register_pickle_func(struct, func):
    def pickle_func(obj):
        return struct, (func(obj), )

    copyreg.pickle(struct, pickle_func)


def register_pickle_custom(struct, *params):
    def pickle_func(obj):
        return struct, tuple(getattr(obj, p) for p in params)

    copyreg.pickle(struct, pickle_func)


## Define how to pickle system_data objects

register_pickle_func(Uint32Vector, tuple)
register_pickle_func(DoubleVector, tuple)
register_pickle_custom(system_data, 'dl_m_e', 'dl_m_h', 'eps_r', 'T')

# Generate an iterator that behaves like
# linspace when func == None.
#
# Otherwise func is applied to every element.


class iter_linspace:
    def __init__(self, x0, x1, N, func=None):
        self.x0 = x0
        self.x1 = x1
        self.N = N
        self.h = (x1 - x0) / (N - 1)
        self.i = 0
        self.func = func

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.N:
            raise StopIteration

        if self.func == None:
            r = self.x0 + self.h * self.i
        else:
            r = self.func(self.x0 + self.h * self.i)

        self.i += 1
        return r


def iter_log_func(x):
    return 10**x


def pickle_iter_linspace(iter_x):
    return iter_linspace, (iter_x.x0, iter_x.x1, iter_x.N, iter_x.func)


copyreg.pickle(iter_linspace, pickle_iter_linspace)


def color_map(cx_arr):
    r, ph = abs(cx_arr), angle(cx_arr)

    h = 0.5 + 0.5 * ph / pi
    s = 0.9 * ones_like(r)
    v = r / (1.0 + r)

    return array([h, s, v]).T


def time_func(func, *args, **kwargs):
    t0 = time.time()
    r = func(*args, **kwargs)
    print('%s (dt: %.2fs)' % (func.__name__, time.time() - t0))
    return r

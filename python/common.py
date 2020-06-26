"""
  Common functions and modules used by
  the rest of the python code.
"""

import time
import copyreg
import itertools
import operator
import json
import uuid
import base64
import sys as pysys
import os
from ctypes import *
import multiprocessing

from numpy import *
import cmath

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LogNorm, SymLogNorm, ListedColormap, BoundaryNorm
import matplotlib.colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

from mpl_toolkits.mplot3d import Axes3D

import scipy.integrate
from scipy.integrate import (quad, simps, dblquad)
from scipy.interpolate import interp1d
from scipy import special
import scipy.misc
import scipy.linalg
import scipy.stats as stats
from scipy.optimize import (
    minimize,
    minimize_scalar,
    root,
    root_scalar,
    curve_fit,
)

from semiconductor import *
from job_api import JobAPI

initializeMPFR_GSL()


# https://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting
# Define function for string formatting of scientific notation
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """

    if isinf(num):
        return r'$\infty$'
    elif num == 0:
        return r'$0$'

    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r'${0:.{2}f}\cdot10^{{{1:d}}}$'.format(coeff, exponent, precision)


def save_data(filename, vars_list, extra_data=None):
    export_data = zeros((
        max(*[v.size for v in vars_list])
        if len(vars_list) > 1 else vars_list[0].size,
        len(vars_list),
    ))

    for (i, var) in enumerate(vars_list):
        export_data[:var.size, i] = var

    savetxt('%s.csv.gz' % filename, export_data, delimiter=',')

    print('Saved to %s' % filename)

    if not extra_data is None:
        with open('%s.json' % filename, 'w+') as fp:
            json.dump(extra_data, fp)


def load_data(filename, extra_dict={}):
    try:
        exported_data = loadtxt('%s.csv.gz' % filename, delimiter=',').T
    except:
        exported_data = loadtxt('%s.csv' % filename, delimiter=',').T

    try:
        with open('%s.json' % filename, 'r') as fp:
            data_file = json.load(fp)
            extra_dict.update(data_file)
    except Exception as exc:
        print('load_data: %s' % exc)

    return exported_data


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
register_pickle_custom(
    system_data,
    'dl_m_e',
    'dl_m_h',
    'eps_r',
    'T',
    'size_d',
    'size_Lx',
    'size_Ly',
    'hwhm_x',
    'hwhm_y',
    'eps_mat',
    'ext_dist_l',
)


## Define how to convert a result_s type to a python object
class result_s:
    _ATTR_LIST = ['value', 'error', 'neval']
    _ATTR_TYPES = [c_double, c_double, c_int32]
    _ATTR_PYTHON_TYPE = [list, list, list]

    def __init__(self, c_result):
        self.c_result = c_result
        self.n_int = c_result.n_int

        for key, key_type, key_python_type in zip(self._ATTR_LIST,
                                                  self._ATTR_TYPES,
                                                  self._ATTR_PYTHON_TYPE):
            setattr(
                self, key,
                key_python_type((key_type * self.n_int).from_address(
                    int(getattr(self.c_result, key)))))

    def total_value(self):
        return self.c_result.total_value()

    def total_error(self):
        return self.c_result.total_error()

    def total_abs_error(self):
        return sum((v * err for v, err in zip(self.value, self.error)))


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


def states_sorted(n_states, Lx, Ly, sys, nmax=30):
    """
    Returns the ''n_states'' lowest energy states.
    If ''n_states'' is negative, it is interpreted as the maximum energy.
    """
    states_vec = list(
        itertools.product(
            range(1, nmax + 1, 2),
            range(1, nmax + 1, 2),
        ))

    energy_vec = [exciton_cm_se(Lx, Ly, nx, ny, sys) for nx, ny in states_vec]

    if n_states > 0:
        return [x for _, x in sorted(zip(energy_vec, states_vec))][:n_states]

    return [
        x for e, x in sorted(zip(energy_vec, states_vec)) if e < (-n_states)
    ]


def states_sorted_os(n_states, Lx, Ly, nmax=30):
    """
    Returns the ''n_states'' lowest energy states.
    If ''n_states'' is negative, it is interpreted as the maximum energy.
    """
    states_vec = list(
        itertools.product(
            range(1, nmax + 1, 2),
            range(1, nmax + 1, 2),
        ))

    so_vec = [exciton_os(Lx, Ly, nx, ny) for nx, ny in states_vec]

    if n_states > 0:
        return [x for _, x in sorted(zip(so_vec, states_vec), reverse=True)
                ][:n_states]

    return [
        x for e, x in sorted(zip(so_vec, states_vec), reverse=True)
        if e < (-n_states)
    ]

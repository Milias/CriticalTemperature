import time
import copyreg
import itertools

from numpy import *
import cmath

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

import scipy.special
import scipy.misc

from semiconductor import *
from job_api import JobAPI

initializeMPFR_GSL()

## Define how to pickle system_data objects

def pickle_system_data(sys):
  return system_data, (sys.dl_m_e, sys.dl_m_h, sys.eps_r, sys.T)

copyreg.pickle(system_data, pickle_system_data)

# Generate an iterator that behaves like
# linspace when func == None.
#
# Otherwise func is applied to every element.

class iter_linspace:
  def __init__(self, x0, x1, N, func = None):
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


import os
import sys
import time
import itertools
import json
import uuid
import gzip
import base64

import ctypes

from numpy import *
import numpy.lib.scimath as sm
from scipy.integrate import *
from scipy.optimize import *
from scipy.interpolate import *
from scipy.special import gamma, erfc

import matplotlib.pyplot as plt

import copy
from integrals import *

from multiprocessing import Pool, cpu_count

initializeMPFR_GSL()

def __saveData(func, args, p, bs, dt, N, y):
  data_uuid = uuid.uuid4()
  data_time = time.time()

  args_data = [list(arg) for arg in args]

  data = {
    'uuid' : str(data_uuid),
    'time' : data_time,
    'proc' : p,
    'bsize' : bs,
    'dt' : dt,
    'N' : N,
    'args' : args_data,
    'result' : y
  }

  data_str = json.dumps(data, sort_keys = True)
  filename = 'bin/data/data_%s_%d.json.gz' % (func.__name__, 1e6 * data_time)

  with gzip.open(filename, 'wb') as fp:
    fp.write(data_str.encode())

  print("Saved to %s." % filename)

  return filename

def loadData(filename):
  with gzip.open(filename, 'rb') as fp:
    return json.loads(fp.read().decode())

def parallelTableSync(func, *args, p = None, bs = 16):
  if p == None:
    p = cpu_count()

  print('Starting "%s" with %d processors and block size %d.' % (func.__name__, p, bs))
  args_cpy = copy.deepcopy(args)
  x = map(tuple, zip(*args))

  t0 = time.time()
  with Pool(p) as workers:
    y = workers.starmap(func, x, bs)
  dt = time.time() - t0

  N = len(y)
  print('Finishing "%s": N = %d, t*p/N = %.2f ms, t = %.2f s.' % (func.__name__, N, p * dt * 1e3 / N, dt))

  __saveData(func, args_cpy, p, bs, dt, N, y)

  print('')

  return y

def asyncCallback(result):
  return result

def asyncErrorCallback(error):
  print('Error: %s' % error)

def parallelTable(func, *args, p = None, bs = 16):
  if p == None:
    p = cpu_count()

  print('Starting "%s" with %d processors and block size %d.' % (func.__name__, p, bs))
  args_cpy = copy.deepcopy(args)
  x = map(tuple, zip(*args))

  t0 = time.time()
  with Pool(p) as workers:
    result = workers.starmap_async(func, x, bs, callback = asyncCallback, error_callback = asyncErrorCallback)

    while (result._number_left > 0):
      print('\r%20d' % (result._number_left * result._chunksize), end = '')
      time.sleep(0.1)

    print('')
    y = result.get()

  dt = time.time() - t0

  N = len(y)
  print('Finishing "%s": N = %d, t*p/N = %.2f ms, t = %.2f s.' % (func.__name__, N, p * dt * 1e3 / N, dt))

  __saveData(func, args_cpy, p, bs, dt, N, y)

  print('')

  return y

k_B = 8.6173303e-5 # eV K^-1
m_electron = 0.5109989461e6 # eV
hbar = 6.582119514e-16 # eV s
c = 299792458 # m s^-1


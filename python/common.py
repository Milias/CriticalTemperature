import os
from multiprocessing import Pool, cpu_count

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

def parallelTable(func, *args, p = None, bs = 16):
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

k_B = 8.6173303e-5 # eV K^-1
m_electron = 0.5109989461e6 # eV
hbar = 6.582119514e-16 # eV s
c = 299792458 # m s^-1

"""
t0 = time.time()
for i in range(N):
  #r = complex(invTmatrixMB_real(w, E, mu, beta, a), a + I1(0.25 * E - mu - 0.5 * w))
  #r = polePos(E, mu, beta, a)
  #r = integralBranch(E, mu, beta, a)
  #r = integralDensityPole(mu, beta, a)
  #r = integralDensityBranch(mu, beta, a)
dt = time.time() - t0

print("result: (%.10f, %.10f)" % (real(r), imag(r)));
print("(%d) %.3f μs, %.3f s" % (N, dt * 1e6 / N, dt));
exit()
"""


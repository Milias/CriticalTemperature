import os
from multiprocessing import Pool, cpu_count

import time
import itertools

import ctypes

from numpy import *
import numpy.lib.scimath as sm
from scipy.integrate import *
from scipy.optimize import *
from scipy.interpolate import *
from scipy.special import gamma, erfc

import matplotlib.pyplot as plt

from integrals import *

initializeMPFR_GSL()

def parallelTable(func, *args):
  p = cpu_count()
  x = map(tuple, zip(*args))

  with Pool(p) as workers:
    y = workers.starmap(func, x, 1)

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


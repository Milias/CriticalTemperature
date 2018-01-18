import os
from multiprocessing import Pool, cpu_count

import time
import itertools

from numpy import *
import numpy.lib.scimath as sm
from scipy.integrate import *
from scipy.optimize import *
from scipy.interpolate import *
from scipy.special import gamma, erfc

from mpmath import fp
from mpmath import mp

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.ticker import MaxNLocator

mp.dps = 100
mp.pretty = True

def complexPlot(x, y, line_styles = [ 'r-', 'b-' ]):
  fig, axarr = plt.subplots(1, 1)
  yValues = [real(y), imag(y)]

  axplots = [ axarr.plot(x, yValues[i], line_styles[i]) for i in range(2) ]

  return fig, axarr, axplots

def densityPlot(x, y, Z):
  fig, axarr = plt.subplots(1, 1)
  axarr0 = axarr.imshow(Z, extent = (x[0], x[-1], y[0], y[-1]), cmap = cm.bone, aspect = 'auto')

  fig.colorbar(axarr0, ax=axarr, orientation = 'horizontal')

  return fig, axarr, axarr0

def contourPlot(X, Y, Z):
  levels = MaxNLocator(nbins=32).tick_values(Z.min(), Z.max())

  fig, axarr = plt.subplots(1, 1)
  axplot = axarr.contourf(X, Y, Z[::-1,:], cmap = cm.bone, levels = levels, corner_mask = True)

  fig.colorbar(axplot, ax=axarr, orientation = 'horizontal')

  return fig, axarr, axplot

def densityPlotComplex(x, y, Z, clip_real = (-1, 1), clip_imag = (-1, 1)):
  fig, axarr = plt.subplots(1, 2, sharey = True)
  zValues = [clip(real(Z).T, *clip_real), clip(imag(Z).T, *clip_imag)]

  axarrPlots = [ axarr[i].imshow(zValues[i], extent = (x[0], x[-1], y[0], y[-1]), cmap = cm.bone, aspect = 'auto') for i in range(2) ]

  fig.subplots_adjust(hspace = 0)

  for i in range(2):
    fig.colorbar(axarrPlots[i], ax=axarr[i], orientation = 'horizontal')

  return fig, axarr, axarrPlots

def contourPlotComplex(X, Y, Z):
  levels= MaxNLocator(nbins=16).tick_values(Z.min(), Z.max())
  fig, axarr = plt.subplots(1, 1)
  axarr0 = axarr.contourf(X, Y, Z, cmap = cm.bone, levels = levels)

def parallelTable(x, func, p = cpu_count()):
  X = array(x).T
  with Pool(p) as workers:
    y = asarray(workers.starmap(func, X))

  return y

def myPolyLog(s, x, xmax = 45, nmax = 0):
  # http://functions.wolfram.com/10.08.17.0011.01
  # https://en.wikipedia.org/wiki/Polylogarithm#Limiting_behavior
  if abs(x) == inf:
    return inf
  if abs(x) < xmax:
    #if nmax > 0: print((nmax, x))
    return fp.polylog(s, x)
  else:
    return 2**(s-1)*(myPolyLog(s, mp.sqrt(x), xmax, nmax+1) + myPolyLog(s, -mp.sqrt(x), xmax, nmax+1))


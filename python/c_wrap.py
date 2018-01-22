import ctypes
import time

### Global

class genericFunctor:
  def __init__(self, libc, func_name, argtypes, restype):
    self.func = libc.__getattr__(func_name)
    self.func.argtypes = argtypes
    self.func.restype = restype

  def __call__(self, *args):
    return self.func(*args)

integrals_so = ctypes.CDLL('bin/libintegrals.so')

### common.h

initializeMPFR_GSL = genericFunctor(integrals_so, "initializeMPFR_GSL", [], None)

initializeMPFR_GSL()

logExp_functor = genericFunctor(integrals_so, "logExp", [ ctypes.c_double, ctypes.c_double ], ctypes.c_double)

def logExp(x, xmax = 50):
  return logExp_functor(x, xmax)

### integrals.h

I1_functor = genericFunctor(integrals_so, "I1", [ ctypes.c_double ], ctypes.c_double)
I1dmu_functor = genericFunctor(integrals_so, "I1dmu", [ ctypes.c_double ], ctypes.c_double)

integrandI2part2_functor = genericFunctor(integrals_so, "integrandI2part2", [ ctypes.c_double, ctypes.c_void_p ], ctypes.c_double)

I2_real_functor = genericFunctor(integrals_so, "integralI2Real", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)
I2_imag_functor = genericFunctor(integrals_so, "integralI2Imag", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

invTmatrixMB_real_functor = genericFunctor(integrals_so, "invTmatrixMB_real", [ ctypes.c_double, ctypes.c_void_p ], ctypes.c_double)

invTmatrixMB_imag_functor = genericFunctor(integrals_so, "invTmatrixMB_imag", [ ctypes.c_double, ctypes.c_void_p ], ctypes.c_double)

polePos_functor = genericFunctor(integrals_so, "polePos", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

integrandPoleRes_functor = genericFunctor(integrals_so, "integrandPoleRes", [ ctypes.c_double,  ctypes.c_void_p ], ctypes.c_double)

poleRes_functor = genericFunctor(integrals_so, "poleRes", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

findLastPos_functor = genericFunctor(integrals_so, "findLastPos", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double ], ctypes.c_double)

integrandBranch_functor = genericFunctor(integrals_so, "integrandBranch", [ ctypes.c_double, ctypes.c_void_p ], ctypes.c_double)

integralBranch_functor = genericFunctor(integrals_so, "integralBranch", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

integrandDensityPole_functor = genericFunctor(integrals_so, "integrandDensityPole", [ ctypes.c_double,  ctypes.c_void_p ], ctypes.c_double)

integralDensityPole_functor = genericFunctor(integrals_so, "integralDensityPole", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double ], ctypes.c_double)

integralDensityBranch_functor = genericFunctor(integrals_so, "integralDensityBranch", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double ], ctypes.c_double)

analytic_n_id_functor = genericFunctor(integrals_so, "analytic_n_id", [ ctypes.c_double, ctypes.c_double ], ctypes.c_double)

analytic_n_ex_functor = genericFunctor(integrals_so, "analytic_n_ex", [ ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

analytic_n_sc_functor = genericFunctor(integrals_so, "analytic_n_sc", [ ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

analytic_n_functor = genericFunctor(integrals_so, "analytic_n", [ ctypes.c_double,  ctypes.c_void_p ], ctypes.c_double)

analytic_mu_functor = genericFunctor(integrals_so, "analytic_mu", [ ctypes.c_double, ctypes.c_double ], ctypes.c_double)

def I1(x):
  return I1_functor(x)

def I1dmu(x):
  return I1dmu_functor(x)

def integrandI2part2(x, *args):
  params = (ctypes.c_double * 5)()

  for i in range(len(args)):
    params[i] = args[i]

  return integrandI2part2_functor(x, ctypes.cast(params, ctypes.c_void_p))

def I2_real(w, E, mu, beta):
  return I2_real_functor(w, E, mu, beta)

def I2_imag(w, E, mu, beta):
  return I2_imag_functor(w, E, mu, beta)

def I2(w, E, mu, beta):
  return complex(I2_real(w, E, mu, beta), I2_imag(w, E, mu, beta))

def invTmatrixMB_real(w, *args):
  params = (ctypes.c_double * 4)()

  for i in range(len(args)):
    params[i] = args[i]

  return invTmatrixMB_real_functor(w, ctypes.cast(params, ctypes.c_void_p))

def invTmatrixMB_imag(w, *args):
  params = (ctypes.c_double * 4)()

  for i in range(len(args)):
    params[i] = args[i]

  return invTmatrixMB_imag_functor(w, ctypes.cast(params, ctypes.c_void_p))

def invTmatrixMB(w, *args):
  params = (ctypes.c_double * 4)()

  for i in range(len(args)):
    params[i] = args[i]

  return complex(invTmatrixMB_real_functor(w, ctypes.cast(params, ctypes.c_void_p)), invTmatrixMB_imag_functor(w, ctypes.cast(params, ctypes.c_void_p)))

def polePos(E, mu, beta, a):
  return polePos_functor(E, mu, beta, a)

def integrandPoleRes(x, *args):
  params = (ctypes.c_double * 4)()

  for i in range(len(args)):
    params[i] = args[i]

  return integrandPoleRes_functor(x, ctypes.cast(params, ctypes.c_void_p))

def poleRes(E, mu, beta, a):
  return poleRes_functor(E, mu, beta, a)

def findLastPos(mu, beta, a):
  return findLastPos_functor(mu, beta, a)

def integrandBranch(y, *args):
  params = (ctypes.c_double * 4)()

  for i in range(len(args)):
    params[i] = args[i]

  return integrandBranch_functor(y, ctypes.cast(params, ctypes.c_void_p))

def integralBranch(E, mu, beta, a):
  return integralBranch_functor(E, mu, beta, a)

def integrandDensityPole(E, *args):
  params = (ctypes.c_double * 4)()

  for i in range(len(args)):
    params[i] = args[i]

  return integrandDensityPole_functor(E, ctypes.cast(params, ctypes.c_void_p))

def integralDensityPole(mu, beta, a):
  return integralDensityPole_functor(mu, beta, a)

def integralDensityBranch(mu, beta, a):
  return integralDensityBranch_functor(mu, beta, a)

def analytic_n_id(mu, beta):
  return analytic_n_id_functor(mu, beta)

def analytic_n_ex(mu, beta, a):
  return analytic_n_ex_functor(mu, beta, a)

def analytic_n_sc(mu, beta, a):
  return analytic_n_sc_functor(mu, beta, a)

def analytic_n(mu, *args):
  params = (ctypes.c_double * 3)()

  for i in range(len(args)):
    params[i] = args[i]

  return analytic_n_functor(mu, ctypes.cast(params, ctypes.c_void_p))

def analytic_mu(beta, a):
  return analytic_mu_functor(beta, a)


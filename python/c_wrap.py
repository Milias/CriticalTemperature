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

I2_real_functor = genericFunctor(integrals_so, "integralI2Real", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)
I2_imag_functor = genericFunctor(integrals_so, "integralI2Imag", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

invTmatrixMB_real_functor = genericFunctor(integrals_so, "invTmatrixMB_real", [ ctypes.c_double, ctypes.c_void_p ], ctypes.c_double)

invTmatrixMB_imag_functor = genericFunctor(integrals_so, "invTmatrixMB_imag", [ ctypes.c_double, ctypes.c_void_p ], ctypes.c_double)

polePos_functor = genericFunctor(integrals_so, "polePos", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

poleRes_functor = genericFunctor(integrals_so, "poleRes", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

poleRes_pole_functor = genericFunctor(integrals_so, "poleRes_pole", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

integralBranch_functor = genericFunctor(integrals_so, "integralBranch", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

integralDensityPole_functor = genericFunctor(integrals_so, "integralDensityPole", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double ], ctypes.c_double)

integralDensityBranch_functor = genericFunctor(integrals_so, "integralDensityBranch", [ ctypes.c_double,  ctypes.c_double, ctypes.c_double ], ctypes.c_double)

def I1(x):
  return I1_functor(x)

def I1dmu(x):
  return I1dmu_functor(x)

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

def poleRes(E, mu, beta, a):
  return poleRes_functor(E, mu, beta, a)

def integralBranch(E, mu, beta, a):
  return integralBranch_functor(E, mu, beta, a)

def integralDensityPole(mu, beta, a):
  return integralDensityPole_functor(mu, beta, a)

def integralDensityBranch(mu, beta, a):
  return integralDensityBranch_functor(mu, beta, a)


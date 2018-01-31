import ctypes
import time

from numpy.ctypeslib import ndpointer
from numpy import *

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

## Scattering length: ODE

wavefunction_int_functor = genericFunctor(integrals_so, "wavefunction_int", [ ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

## Scattering length: ideal gas mu

ideal_mu_functor = genericFunctor(integrals_so, "ideal_mu", [ ctypes.c_double, ctypes.c_double ], ctypes.c_double)
ideal_mu_dn_functor = genericFunctor(integrals_so, "ideal_mu_dn", [ ctypes.c_double, ctypes.c_double ], ctypes.c_double)

## Analytic mu

analytic_mu_param_functor = genericFunctor(integrals_so, "analytic_mu_param", [ ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double ], ndpointer(dtype = float64, ndim = 1, shape = (2,), flags = 'C_CONTIGUOUS'))

analytic_mu_param_dn_functor = genericFunctor(integrals_so, "analytic_mu_param_dn", [ ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double ], ndpointer(dtype = float64, ndim = 1, shape = (2,), flags = 'C_CONTIGUOUS'))

analytic_mu_functor = genericFunctor(integrals_so, "analytic_mu", [ ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double ], ndpointer(dtype = float64, ndim = 1, shape = (3,), flags = 'C_CONTIGUOUS'))

### classical.h

integralSuscp_cr_functor = genericFunctor(integrals_so, "integralSuscp_cr", [ ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

integralSuscp_ci_functor = genericFunctor(integrals_so, "integralSuscp_ci", [ ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

suscp_cr_functor = genericFunctor(integrals_so, "suscp_cr", [ ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)
suscp_ci_functor = genericFunctor(integrals_so, "suscp_ci", [ ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

suscp_czr_functor = genericFunctor(integrals_so, "suscp_czr", [ ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)
suscp_czi_functor = genericFunctor(integrals_so, "suscp_czi", [ ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double ], ctypes.c_double)

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

## Scattering length: ODE

def wavefunction_int(eps_r, e_ratio, lambda_s):
  return wavefunction_int_functor(eps_r, e_ratio, lambda_s)

## Scattering length: ideal gas mu

def ideal_mu(n_dless, m_ratio):
  return ideal_mu_functor(n_dless, m_ratio)

def ideal_mu_dn(n_dless, m_ratio):
  return ideal_mu_dn_functor(n_dless, m_ratio)

# Analytic mu

def analytic_mu_param(n_dless, m_ratio_e, m_ratio_h, a):
  return analytic_mu_param_functor(n_dless, m_ratio_e, m_ratio_h, a)

def analytic_mu_param_dn(n_dless, m_ratio_e, m_ratio_h, a):
  return analytic_mu_param_dn_functor(n_dless, m_ratio_e, m_ratio_h, a)

def analytic_mu(n_dless, m_ratio_e, m_ratio_h, eps_r, e_ratio):
  return analytic_mu_functor(n_dless, m_ratio_e, m_ratio_h, eps_r, e_ratio)

### classical.h

def integralSuscp_cr(w, E, mu_ph, m_i, m_r):
  return integralSuscp_cr_functor(w, E, mu_ph, m_i, m_r)

def integralSuscp_ci(w, E, mu_ph, m_i, m_r):
  return integralSuscp_ci_functor(w, E, mu_ph, m_i, m_r)

def integralSuscp_cc(w, E, mu_ph, m_i, m_r):
  return complex(integralSuscp_cr_functor(w, E, mu_ph, m_i, m_r), integralSuscp_ci_functor(w, E, mu_ph, m_i, m_r))

def suscp_cr(w, p, mu_1, mu_2, m_1, m_2, m_r, beta, V_0):
  return suscp_cr_functor(w, p, mu_1, mu_2, m_1, m_2, m_r, beta, V_0)

def suscp_ci(w, p, mu_1, mu_2, m_1, m_2, m_r, beta, V_0):
  return suscp_ci_functor(w, p, mu_1, mu_2, m_1, m_2, m_r, beta, V_0)

def suscp_cc(w, p, mu_1, mu_2, m_1, m_2, m_r, beta, V_0):
  return complex(suscp_cr_functor(w, p, mu_1, mu_2, m_1, m_2, m_r, beta, V_0), suscp_ci_functor(w, p, mu_1, mu_2, m_1, m_2, m_r, beta, V_0))

def suscp_czr(w, mu_1, mu_2, m_1, m_2, m_r, beta, V_0):
  return suscp_czr_functor(w, mu_1, mu_2, m_1, m_2, m_r, beta, V_0)

def suscp_czi(w, mu_1, mu_2, m_1, m_2, m_r, beta, V_0):
  return suscp_czi_functor(w, mu_1, mu_2, m_1, m_2, m_r, beta, V_0)

def suscp_czc(w, mu_1, mu_2, m_1, m_2, m_r, beta, V_0):
  return complex(suscp_czr_functor(w, mu_1, mu_2, m_1, m_2, m_r, beta, V_0), suscp_czi_functor(w, mu_1, mu_2, m_1, m_2, m_r, beta, V_0))


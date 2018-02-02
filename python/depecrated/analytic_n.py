from common import *

### Integrals ###
from I1 import *
from I2 import *

def n_id(mu, beta):
  bmu = beta * mu
  return - 1/(4 / 3 / sqrt(pi) * beta**1.5) * real(myPolyLog(1.5, -mp.exp(bmu)))

def n_ex(mu, beta, a):
  if a < 0: return 0
  ba = beta * 8 * a**2 + 2 * beta * mu
  return 1/(4 / 3 / sqrt(pi) * beta**1.5) * 2 * real(myPolyLog(1.5, mp.exp(ba)))

def n_sc_integrand(y, mu, beta, a):
  li_exp = beta * (- 0.5 * y**2 + 2 * mu)
  return myPolyLog(1.5, mp.exp(li_exp)) / (y**2 + 16 * a**2)

def n_sc(mu, beta, a):
  return - 1/(4 / 3 / sqrt(pi) * beta**1.5) * 8 / pi * a * quad(n_sc_integrand, 0, inf, args = (mu, beta, a), limit = 1000, epsabs = 1e-20)[0]

def densitySolveFunc(mu, beta, a):
  # a -> 1 / a
  mu = mp.mpf(float(mu))
  beta = mp.mpf(float(beta))
  a = mp.mpf(float(a))

  return - 1 + n_id(mu, beta) + n_ex(mu, beta, a) + n_sc(mu, beta, a)

def computeChemPot(beta, a = -1, mu0 = -10000):
  mu1 = -4*a**2 if a > 0 else 0

  r1, r2 = densitySolveFunc(mu0, beta, a), densitySolveFunc(mu1, beta, a)
  if r1 * r2 > 0:
    return 0 #float('nan')
  else:
    return bisect(densitySolveFunc, mu0, mu1, args = (beta, a))


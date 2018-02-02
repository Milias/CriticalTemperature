from common import *

### Integrals ###
from I1 import *
from I2 import *

def invTmatrixMB(w, E, mu, beta, a):
  return a + I1(w, E, mu) + I2(w, E, mu, beta)

def invTmatrixMBReal(w, E, mu, beta, a):
  return real(invTmatrixMB(w, E, mu, beta, a))

def specFuncPart1(w, k, mu, beta, a):
  E = k**2
  return I1dmu(w, E, mu) / invTmatrixMB(w, E, mu, beta, a)

def specFunc(w, k, mu, beta, a):
  E = k**2
  return (I1dmu(w, E, mu) + 2 * I2dmu(w, E, mu, beta)) / invTmatrixMB(w, E, mu, beta, a)

def polePos(E, mu, beta, a, wmin = -1e10):
  z1 = 0.5 * E - 2 * mu
  if invTmatrixMBReal(wmin, E, mu, beta, a) * invTmatrixMBReal(z1 - 1e-10, E, mu, beta, a) < 0:
    return bisect(invTmatrixMBReal, wmin, z1 - 1e-10, args = (E, mu, beta, a))
  else:
    return None

def integrandI2Pole(x, z0, E, mu, beta):
  return integrandI2part1(x, E, mu, beta) / (x + 0.5 * (0.5 * E - 2 * mu - z0))**2

def integralI2Pole(z0, E, mu, beta):
  return quad(integrandI2Pole, 0, inf, args = (z0, E, mu, beta), limit = 1000, epsabs = 1e-20)[0]

def poleRes(E, mu, beta, a):
  z1 = 0.5 * E - 2 * mu
  z0 = polePos(E, mu, beta, a)

  if z0 == None:
    return 0

  return (I1dmu(z0, E, mu)) / (mp.exp(beta * z0) + 1) / ( 1/sm.sqrt(2*(z1-z0)) - integralI2Pole(z0, E, mu, beta) / pi )

def integrandBranch(y, E, mu, beta, a):
  return imag( ( I1dmu(y, E, mu) ) / ( a + I1(y, E, mu) + I2(y, E, mu, beta) ) ) / ( 2 * pi ) / (mp.exp(beta * y) - 1)
  #return imag( ( 1 ) / ( a + I1(y, E, mu) + I2(y, E, mu, beta) ) ) / ( 2 * pi ) / (mp.exp(beta * y) - 1)

def integralBranch(E, mu, beta, a):
  z1 = 0.5 * E - 2 * mu
  return quad(integrandBranch, z1, inf, args = (E, mu, beta, a), limit = 1000)[0]
  #return mp.quad(lambda x: integrandBranch(x, E, mu, beta, a), [z1, inf], method = 'tanh-sinh')

def specFuncPolesDmu(w, E, mu, beta, a):
  return invTmatrixMB(w, E, mu, beta, a)

def specFuncBranchDmu(w, E, mu, beta, a):
  return 1 / (I1dmu(w, E, mu) + 2 * I2dmu(w, E, mu, beta))


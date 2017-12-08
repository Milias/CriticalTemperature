from common import *

def integrandI2part1(x, E, mu, beta):
  if E < 1e-6:
    return sm.sqrt(x) / (mp.exp(beta * (x - mu)) + 1)

  else:
    E_ex = 0.25 * E - mu + x
    Ep, Em = E_ex + sm.sqrt(E * x), E_ex - sm.sqrt(E * x)
    return sm.sqrt(x) + (float(mp.log(1 + mp.exp(beta * Em)) - mp.log(1 + mp.exp(beta * Ep)))) / ( 2 * beta * sm.sqrt(E) )
    #return sm.sqrt(x) + (log(1 + exp(beta * Em)) - log(1 + exp(beta * Ep))) / ( 2 * beta * sm.sqrt(E) )

def integrandI2part2(x, w, E, mu, beta):
  z2 = 0.5 * w - 0.25 * E + mu
  return integrandI2part1(x, E, mu, beta) / (x - z2)

def integralI2Real(w, E, mu, beta):
  z2 = 0.5 * w - 0.25 * E + mu

  if z2 < 0:
    return quad(integrandI2part2, 0, inf, args = (w, E, mu, beta), limit = 1000)[0]
  else:
    return quad(integrandI2part1, 0, 2 * z2, args = (E, mu, beta), limit = 1000, weight = 'cauchy', wvar = z2)[0] + quad(integrandI2part2, 2 * z2, inf, args = (w, E, mu, beta), limit = 1000, epsabs = 1e-20)[0]

def integralI2Imag(w, E, mu, beta):
  z2 = 0.5 * w - 0.25 * E + mu

  if z2 < 0:
    return 0

  return integrandI2part1(z2, E, mu, beta)

def I2(w, E, mu, beta):
  return complex(2 / pi * integralI2Real(w, E, mu, beta) + 2j * integralI2Imag(w, E, mu, beta) )

def I2dmuPartAIntegrand(x, E, mu, beta):
  if E < 1e-6:
    return beta * mp.exp(beta * x) / (mp.exp(beta * x) + 1)**2

  E_ex = 0.25 * E - mu + x
  Ep, Em = E_ex + 0.25 * sm.sqrt(E * x), E_ex - 0.25 * sm.sqrt(E * x)
  return ( mp.exp(Ep) / (mp.exp(beta * Ep) + 1) - mp.exp(Em) / (mp.exp(beta * Em) + 1) ) / mp.sqrt(E * x)

def I2dmuPartBIntegrand(x, E, mu, beta):
  return 2 * integrandI2part1(x, E, mu, beta)

def I2dmuIntegrand(x, w, E, mu, beta):
  return sm.sqrt(x) / (1j * w - 2 * ( x + 0.25 * E - mu ) ) * ( I2dmuPartAIntegrand(x, E, mu, beta) - I2dmuPartBIntegrand(x, E, mu, beta) / (1j * w - 2 * ( x + 0.25 * E - mu ) ) )

def I2dmuIntegrandReal(x, w, E, mu, beta):
  return real(I2dmuPartBIntegrand(x, E, mu, beta))

def I2dmuIntegrandImag(x, w, E, mu, beta):
  return imag(I2dmuPartBIntegrand(x, E, mu, beta))

def I2dmu(w, E, mu, beta):
  return 2 / pi * ( quad(I2dmuIntegrandReal, 0, inf, args = (w, E, mu, beta), limit = 1000, epsabs = 1e-20)[0] + 1j * quad(I2dmuIntegrandImag, 0, inf, args = (w, E, mu, beta), limit = 1000, epsabs = 1e-20)[0])


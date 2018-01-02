from common import *

### Integrals ###
from I1 import *
from I2 import *

### Density computation ###
from analytic_n import *
from spec_func_full import *

def plotSpecFunc(N, k, beta, a):
  mu = computeChemPot(beta, a)

  print('N: %d, β: %f, μ: %f, a: %f' % (N*N, beta, mu, a))

  x = linspace(-1, 5, N) # real(w)
  y = linspace(-1, 1, N) # imag(w)

  ones_arr = ones(N*N)

  X, Y = meshgrid(x, y)
  Z = parallelTable([X.ravel() - 1j * Y.ravel(), k * ones_arr, mu * ones_arr, beta * ones_arr, a * ones_arr], specFuncPart1).reshape((N, N))

  densityPlotComplex(x, y, Z, (-20, 20), (-20, 20))
  plt.show()

def plotSpecFunc2D(N, k, beta, a):
  mu = computeChemPot(beta, a)

  print('N: %d, β: %f, a: %f, μ; %f' % (N, beta, a, mu))

  x = linspace(-3, 3, N)

  ones_arr = ones(N)

  y1 = parallelTable([x - 1e-10j, k * ones_arr, mu * ones_arr, beta * ones_arr, a * ones_arr], specFunc)

  complexPlot(x, y1)
  plt.show()

def plotDensityContributions(N, beta, a):
  ones_arr = ones(N)
  print(u'N: %d, β: %f' % (N, beta))

  mu = parallelTable([beta * ones_arr, a], computeChemPot)

  n_id_data = parallelTable([mu, beta * ones_arr], n_id)
  n_ex_data = parallelTable([mu, beta * ones_arr, a], n_ex)
  n_sc_data = parallelTable([mu, beta * ones_arr, a], n_sc)

  fig, axarr = plt.subplots(2, 2, sharex = True)
  axarrPlots = [ axarr[0,0].plot(a, mu, 'r-'), axarr[1,0].plot(a, n_id_data + n_sc_data + n_ex_data, 'g-'), axarr[0,1].plot(a, n_sc_data + n_ex_data, 'b-'), axarr[1,1].plot(a, n_id_data, 'k-') ]

  fig.subplots_adjust(hspace = 0)

  plt.show()

def plotT00(N, beta, a):
  ones_arr = ones(N)
  mu = parallelTable([beta, a * ones_arr], computeChemPot)
  #plt.plot(beta, mu)
  #plt.show()

  y = parallelTable([0 * ones_arr, 0 * ones_arr, mu, beta, a * ones_arr, 1 * ones_arr, 1 * ones_arr], invTmatrixMB)

  complexPlot(beta, y)
  #plt.plot(beta, y, 'b.')
  plt.show()

def plotDensity(N, mu, beta, a):
  ones_arr = ones(N)

  y = parallelTable([mu, beta * ones_arr, a * ones_arr], densitySolveFunc)

  plt.plot(mu, y, 'b.')
  plt.show()

N = 1<<5

print(fp.polylog(1.5, exp(1)))
print(fp.polylog(1.5, -exp(1)))
exit()

a = -1
beta = 1.5
#plotDensity(N, linspace(-1, 0, N), beta, a)
#plotT00(N, linspace(1e-2, 2, N), a)
#plotDensityContributions(N, beta, linspace(-1, 1, N))
#exit()

plotSpecFunc(N, 0, 0.8, -0.1)
#plotSpecFunc2D(N, 0, 3, 0.2)
plt.show()
exit()

#"""
beta = linspace(1e-1, 1, N)
a = linspace(-3, 3, N)

beta_mu = linspace(beta[0] * 0.5, beta[-1] * 2, N)
a_mu = linspace(a[0] * 2 if a[0] < 0 else a[0] * 0.5, a[-1] * 2, N)

Beta_mu, A_mu = meshgrid(beta_mu, a_mu)

print('Size: %d\n' % Beta_mu.size)

mu_data = parallelTable([Beta_mu.ravel(), A_mu.ravel()], computeChemPot).reshape(beta_mu.size, a_mu.size)

muFunc = interp2d(beta_mu, a_mu, mu_data, kind = 'linear', bounds_error = False, fill_value = 0)
#"""

#"""
densityPlot(beta_mu, a_mu, mu_data)
#contourPlot(Beta_mu, A_mu, mu_data)
plt.show()
exit()
#"""

"""
a_float = -1

plt.title(r'Chemical potential $\mu(\beta)$, $d$ = 3, $g_s$ = 1')
plt.xlabel(r'$\beta$ / units of $\epsilon_F$')
plt.ylabel(r'$\mu(\beta)$ / units of $\epsilon_F$')
#plt.axis([0, beta[-1], -10, 2])
plt.plot(beta, muFunc(beta, a_float), 'r-', label='Numeric')
#plt.plot(beta, (log(2/3/gamma(1.5)) + 1.5 * log(beta))/beta, 'k--', label=r'$\beta\ll1$')
#plt.plot(beta, 1 - pi**2/12/beta**2, 'b--', label=r'$\beta\gg1$')
plt.legend(loc=0)
plt.show()
exit()
"""

#a = linspace(-1, 0, N)
#x, y = computeCritTemp(a, n, m)
#x = 1/((3*pi**2*n)**(1/3.0)*a)
#y = 1/(y*kB)
#fit_params = analyticFit(x, y.ravel(), (m,))
#print("eF = %f, kF = %f, kF(eF) = %f, kF(mu) = %f" % (fit_params[0], fit_params[1], sqrt(2*m*fit_params[0])/hbar, kF))
#plt.plot(x, analyticCritTemp(x, *fit_params), 'b--')

def alphaCoeffNoA(beta):
  beta = float(beta)
  mu = float(muFunc(beta))
  return  (sqrt(-mu) if mu < 0 else 0) + mpIntegral(beta, mu) * 2 / pi

def alphaCoeffFromA(alphaNoA, a):
  return  a - alphaNoA

def alphaCoeffFull(beta, a):
  beta = float(beta)
  mu = float(muFunc(beta, a))
  return  a - (sqrt(-mu) if mu < 0 else 0) - mpIntegral(beta, mu) * 2 / pi

def critTemp(a):
  #beta0 = float(pi / 64.0 * mp.exp(2 - mp.euler - 0.5 * pi * a)) if a < 0 else 0.5 * a**2 + 0.6
  try:
    return bisect(alphaCoeffFull, beta[0], beta[-1], args = (a, ))
  except Exception as e:
    print((str(e), a))
    return beta[-1]


def computeAlphaCoeff(a, beta):
  #alphaNoA = parallelTable([beta], alphaCoeffNoA)

  for a in a:
    aa = a * ones(beta.size)

    y = parallelTable([beta, aa], alphaCoeffFull)

    plt.plot(beta, y, 'o-', label=a)
    del aa

  plt.legend(loc=0)

def computeCritTemp(a):
  y = parallelTable(a, critTemp).flatten()
  plt.title(r'Critical temperature $T_c$, d = 3, $g_s$ = 1')
  plt.xlabel(r'$(k_F a)^{-1}$')
  plt.ylabel(r'$k_B T_c \epsilon_F^{-1}$')
  a = a[0]
  aa = linspace(a[0], a[-1], 2**10)
  #plt.axis([a[0], 0, 0, 8 / pi * exp(float(mp.euler) - 2 - 0.5 * pi * abs(a[0]))])
  plt.plot(aa[aa<0], 8 / pi * exp(float(mp.euler) - 2 + 0.5 * pi * aa[aa<0]), 'k--', label = 'Approximation')
  plt.plot(aa[aa<0], 64.0 / pi * exp(float(mp.euler) - 2 + 0.5 * pi * aa[aa<0]), 'g--', label = r'$\beta_0$')
  plt.plot(aa[aa>0], 0.6 + 0.5 * aa[aa>0]**2, 'b--', label = r'$\beta_0$')
  plt.plot(a, 1/y, 'ro', label = 'Numeric')
  plt.legend(loc=0)

a = linspace(-1, 1, 10).T

#computeAlphaCoeff(a, beta)
#plt.show()

computeCritTemp([linspace(-1, 1, 32)])
plt.show()


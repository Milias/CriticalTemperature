from common import *
from c_wrap import *

#print(os.getpid())

def parallelTable(func, *args):
  p = cpu_count()
  x = map(tuple, zip(*args))

  with Pool(p) as workers:
    #y = workers.starmap(func, x, int(ceil(len(args[0]) / p)))
    y = workers.starmap(func, x, 1)

  return y

def compExcitonicDensity(mu, beta, a):
  t0 = time.time()
  y = array(parallelTable(integralDensityPole, mu, itertools.repeat(beta, N), itertools.repeat(a, N)))
  dt = time.time() - t0

  print("(%d) %.3f μs, %.3f s" % (N, dt * 1e6 / N, dt));
  return y

k_B = 8.6173303e-5 # eV K^-1
m_electron = 0.5109989461e6 # eV
hbar = 6.582119514e-16 # eV s
c = 299792458 # m s^-1

N = 1<<10
w, E, mu, beta, a = -1, 0, -1, 0.1, 10
p = 0
mu_1, mu_2 = -20, -20
mu_ph = mu_1 + mu_2
m_1, m_2 = 0.28 * m_electron, 0.59 * m_electron # eV
m_r = 1.0 / (1.0 / m_1 + 1.0 / m_2) # eV

T = 300 # K
beta = 1 / (k_B * T) # eV^-1
lambda_th = c * hbar * sqrt(2 * pi * beta / m_r) # m
energy_th = 1 / ( 4 * pi * beta )
eps_r, e_ratio = 6.56, m_r / energy_th

print('%f nm, %f meV, %.2e' % (lambda_th*1e9, energy_th * 1e3, m_r / energy_th))

"""
pole_pos = polePos(E, mu, beta, a)
pole_pos_last = findLastPos(mu, beta, a) - 1e-10
dz = 0.5 * E - 2 * mu - pole_pos
print(pole_pos)
"""
z0 = 0.5 * E - 2 * mu

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
#"""
#x = linspace(0, pole_pos_last if not isnan(pole_pos_last) and pole_pos_last < 100 else 50, N)
#x = linspace(4 * mu if mu > 0 else 0, 40, N)
#x = linspace(2 * dz, 1, N)
#x = linspace(2 * pole_pos if pole_pos < 0 else 2 * pole_pos - (0.5 * E - 2 * mu), 0.5 * E - 2 * mu, N)
#x = linspace(0, 8 * mu if mu > 0 else 10, N)
#x = linspace(-10, z0, N)
#x = linspace(0, (16 / beta + 4 * (mu + a**2)), N)
#x = linspace(0.99 * 4 * mu, 1.01 * 4 * mu, N)
#x = linspace(0, 0.5 * E, N)
#x = linspace(-2.5, 2.5, N)
#x = linspace(-16 * a**2, -4*a**2, N) if a > 0 else linspace(-10, 0, N)
#x = linspace(-1e3, 1e3, N)
#x = linspace(1e-3, 1e3, N)
x = logspace(16, 22, N) * lambda_th**3
#print(x)

lambda_s = 1/sqrt(1/array(parallelTable(mu_ideal_dn, x, itertools.repeat(m_r/m_1, N))) + 1/array(parallelTable(mu_ideal_dn, x, itertools.repeat(m_r/m_2, N))))

#mu_arr = itertools.repeat(mu, N)
#mu_arr = array(parallelTable(analytic_mu, itertools.repeat(beta, N), x))
#y_id_fixed = analytic_n_id(mu, beta)

t0 = time.time()
#y = sqrt(E) * array(parallelTable(poleRes, itertools.repeat(E, N), itertools.repeat(mu, N), itertools.repeat(beta, N), x))
#y = array(parallelTable(integrandPoleRes, x, itertools.repeat(0.5 * E - 2 * mu - pole_pos, N), itertools.repeat(E, N), itertools.repeat(mu, N), itertools.repeat(beta, N)))
#y = array(parallelTable(polePos, x, itertools.repeat(mu, N), itertools.repeat(beta, N), itertools.repeat(a, N)))
#y = array(parallelTable(polePos, itertools.repeat(E, N), itertools.repeat(mu, N), itertools.repeat(beta, N), x))
#y = array(parallelTable(polePos, itertools.repeat(E, N), itertools.repeat(mu, N), itertools.repeat(beta, N), x))

#y = real(array(parallelTable(invTmatrixMB, x, itertools.repeat(E, N), itertools.repeat(mu, N), itertools.repeat(beta, N), itertools.repeat(a, N))))

#y = array(parallelTable(integrandI2part2, x, itertools.repeat(w, N), itertools.repeat(E, N), itertools.repeat(mu, N), itertools.repeat(beta, N), itertools.repeat(0.5 * w - 0.25 * E + mu, N)))
#y = real(array(parallelTable(invTmatrixMB, 0.5 * x - 2 * mu, x, itertools.repeat(mu, N), itertools.repeat(beta, N), itertools.repeat(a, N))))
#y = array(parallelTable(integrandDensityPole, x, itertools.repeat(mu, N), itertools.repeat(beta, N), itertools.repeat(a, N)))
#y = array(parallelTable(integralDensityPole, mu_arr, itertools.repeat(beta, N), x))

#y = array(parallelTable(invTmatrixMB_real, x, itertools.repeat(E, N), itertools.repeat(mu, N), itertools.repeat(beta, N), itertools.repeat(a, N)))

#y = array(parallelTable(integrandBranch, x, itertools.repeat(E, N), itertools.repeat(mu, N), itertools.repeat(beta, N), itertools.repeat(a, N)))
#y = sqrt(x) * array(parallelTable(integralBranch, x, itertools.repeat(mu, N), itertools.repeat(beta, N), itertools.repeat(a, N)))
#y = array(parallelTable(integralDensityBranch, itertools.repeat(mu, N), itertools.repeat(beta, N), x))
#y_id = array(parallelTable(analytic_n_id, mu_arr, itertools.repeat(beta, N)))
#y_ex = array(parallelTable(analytic_n_ex, mu_arr, itertools.repeat(beta, N), x))
#y_sc = array(parallelTable(analytic_n_sc, mu_arr, itertools.repeat(beta, N), x))
#y_n = y_ex + y_sc

#y = array(parallelTable(analytic_n, x, itertools.repeat(beta, N), itertools.repeat(a, N), itertools.repeat(1, N)))
#y = array(parallelTable(analytic_mu, itertools.repeat(beta, N), x))

y = array(parallelTable(wavefunction_int, itertools.repeat(eps_r, N), itertools.repeat(e_ratio, N), lambda_s))
#y = array(parallelTable(mu_ideal_dn, x, itertools.repeat(m_r/m_1, N))) + array(parallelTable(mu_ideal_dn, x, itertools.repeat(m_r/m_2, N)))
#y = (array(parallelTable(mu_ideal, x, itertools.repeat(m_r/m_1, N))) + array(parallelTable(mu_ideal, x, itertools.repeat(m_r/m_2, N)))) * energy_th

#y = array(parallelTable(integralSuscp_cc, x, itertools.repeat(E, N), itertools.repeat(mu_ph, N), itertools.repeat(m_2, N), itertools.repeat(m_r, N)))
#y = imag(array(parallelTable(suscp_czc, x, itertools.repeat(mu_1, N), itertools.repeat(mu_2, N), itertools.repeat(m_1, N), itertools.repeat(m_2, N), itertools.repeat(m_r, N), itertools.repeat(beta, N), itertools.repeat(a, N))))

dt = time.time() - t0

#print(y_ex)

#y = -log10(abs(y))

#y_pos = 0.5 * x - 2 * mu - 2 * a**2

#print(y_approx)

#print(amax(abs(2*(y-y_approx)/(y + y_approx))))
#y = log10(abs(2*(y - y_approx)/(y + y_approx)))

#print(y)

fig, axarr, axarr0 = complexPlot(x * lambda_th**-3, y, ('r-', 'b-'), 'semilogx')
axarr.autoscale(enable = True, axis = 'x', tight = True)
#fig, axarr, axarr0 = realPlot(x, y, ('r-',))
#fig, axarr, axarr0 = complexPlot(x, y)
#axarr.plot(x, y_ex, 'g-')
#axarr.plot(x, y_sc, 'm-')
#axarr.plot(x, y + y1)
#axarr.plot(x, beta * y_pos)
#axarr.plot(x, - 0.25 * E + mu + 0.5 * x)
#axarr.plot(x, y_approx)
#axarr.plot(x, 0.5 * x + polePos(0, mu, beta, a))
#axarr.plot(x, 0.5 * E - 2 * mu - 2 * x**2)
#axarr.axhline(y = invTmatrixMB_real(2 * mu, E, mu, beta, a))
#axarr.axvline(x = pole_pos_last)
#axarr.axvline(x = pole_pos)
#axarr.axvline(x = 0.25 * E)
#axarr.axvline(x = z0)
#if mu >= 0: axarr.axvline(x = 4 * mu)
#axarr.axvline(x = 4 * (mu + a**2))
#axarr.axvline(x = 4 / beta + 4 * (mu + a**2))
#if 2 * w + 4 * mu > 0:
#  axarr.axvline(x = 2 * w + 4 * mu)
#axarr.axvline(x = 0.5 * E - 2 * mu)
axarr.set_ylim(-1e2, 1e2)

#axarr.set_title('%f' % invTmatrixMB_real(0.5 * E - 2 * mu, E, mu, beta, a))
#axarr.set_title('%f' % pole_pos)
#axarr.set_title('Chemical potential')

#axarr.set_xlabel(r'$1 / a$')
#axarr.set_ylabel(r'$\mu$')
#axarr.set_title('Excitonic density contribution, T = %.2f, a = %.2f' % (1/beta, 1/a))

#fig.savefig('python/graphs/chem_pot_analytic.eps')

print("(%d) %.3f μs, %.3f s" % (N, dt * 1e6 / N, dt));
plt.show()
exit()
#"""

v = [ -1, -0.5, 0, 0.5, 1 ]
mu = [linspace(v[i], v[i + 1] - (abs(v[i+1] - v[i])/N if i + 2 < len(v) else 0), N) for i in range(len(v) - 1)]
x = array(mu).reshape(len(mu) * N)

t0 = time.time()
y = array([ compExcitonicDensity(mu_ele, beta, a) for mu_ele in mu ]).reshape(x.size)
dt = time.time() - t0
print("(%d) %.3f μs, %.3f s" % (x.size, dt * 1e6 / x.size, dt));

fig, axarr, axarr0 = complexPlot(x, y)

axarr.set_xlabel(r'$\mu$')
axarr.set_ylabel(r'$n_{ex}$')
axarr.set_title('Excitonic density contribution, T = %.2f, a = %.2f' % (1/beta, 1/a))

plt.show()

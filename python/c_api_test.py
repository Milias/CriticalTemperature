from common import *
from c_wrap import *

def parallelTable(func, *args):
  p = cpu_count()
  x = map(tuple, zip(*args))

  with Pool(p) as workers:
    #y = workers.starmap(func, x, int(ceil(len(args[0]) / p)))
    y = workers.starmap(func, x)

  return y

def compExcitonicDensity(mu, beta, a):
  t0 = time.time()
  y = array(parallelTable(integralDensityPole, mu, itertools.repeat(beta, N), itertools.repeat(a, N)))
  dt = time.time() - t0

  print("(%d) %.3f μs, %.3f s" % (N, dt * 1e6 / N, dt));
  return y

N = 1<<10
w, E, mu, beta, a = 0, 1e5, -1, 1, 1

pole_pos = polePos(E, mu, beta, a)

"""
t0 = time.time()
for i in range(N):
  #r = complex(invTmatrixMB_real(w, E, mu, beta, a), a + I1(0.25 * E - mu - 0.5 * w))
  r = polePos(E, mu, beta, a)
  #r = integralBranch(E, mu, beta, a)
  #r = integralDensityPole(mu, beta, a)
  #r = integralDensityBranch(mu, beta, a)
dt = time.time() - t0

print("result: (%.10f, %.10f)" % (real(r), imag(r)));
print("(%d) %.3f μs, %.3f s" % (N, dt * 1e6 / N, dt));
exit()
"""
#"""
x = linspace(0, 1e3, N)
#x = linspace(2 * pole_pos if pole_pos < 0 else 2 * pole_pos - (0.5 * E - 2 * mu), 0.5 * E - 2 * mu, N)

t0 = time.time()
#y = array(parallelTable(poleRes, x, itertools.repeat(mu, N), itertools.repeat(beta, N), itertools.repeat(a, N)))
#y = array(parallelTable(invTmatrixMB, x, itertools.repeat(E, N), itertools.repeat(mu, N), itertools.repeat(beta, N), itertools.repeat(a, N)))
y = array(parallelTable(polePos, x, itertools.repeat(mu, N), itertools.repeat(beta, N), itertools.repeat(a, N)))
dt = time.time() - t0

print(y)
print(- 2 * a**2 + 0.5 * x - 2 * mu)

y = log10(abs(y + 2 * a**2 - 0.5 * x + 2 * mu))

print(y)

fig, axarr, axarr0 = complexPlot(x, y)
#fig, axarr, axarr0 = complexPlot(x, y)
#axarr.plot(x, - 0.25 * E + mu + 0.5 * x)
#axarr.plot(x, 0.5 * x + polePos(0, mu, beta, a))
#axarr.axhline(y = invTmatrixMB_real(0.5 * E - 2 * mu, E, mu, beta, a))
#axarr.axvline(x = pole_pos)
#axarr.set_ylim(-3, 1.5)

#axarr.set_title('%f' % invTmatrixMB_real(0.5 * E - 2 * mu, E, mu, beta, a))
#axarr.set_title('%f' % pole_pos)

#axarr.set_xlabel(r'$\mu$')
#axarr.set_ylabel(r'$n_{ex}$')
#axarr.set_title('Excitonic density contribution, T = %.2f, a = %.2f' % (1/beta, 1/a))

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

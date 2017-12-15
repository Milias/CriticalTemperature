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

N = 1<<7
w, E, mu, beta, a = 3, 0, -10, 0.1, 10

"""
t0 = time.time()
for i in range(N):
  #r = invTmatrixMB_real(w, E, mu, beta, a)
  r = integralBranch(E, mu, beta, a)
  #r = integralDensityPole(mu, beta, a)
  #r = integralDensityBranch(mu, beta, a)
dt = time.time() - t0

print("result: (%.10f, %.10f)" % (real(r), imag(r)));
print("(%d) %.3f μs, %.3f s" % (N, dt * 1e6 / N, dt));
exit()
"""
"""
x = linspace(0, 10, N)

t0 = time.time()
y = array(parallelTable(poleRes, x, itertools.repeat(mu, N), itertools.repeat(beta, N), itertools.repeat(a, N)))
dt = time.time() - t0

fig, axarr, axarr0 = complexPlot(x, sqrt(x) * y)

axarr.set_xlabel(r'$\mu$')
axarr.set_ylabel(r'$n_{ex}$')
axarr.set_title('Excitonic density contribution, T = %.2f, a = %.2f' % (1/beta, 1/a))

print("(%d) %.3f μs, %.3f s" % (N, dt * 1e6 / N, dt));
plt.show()
exit()
"""

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

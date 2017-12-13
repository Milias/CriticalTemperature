from common import *
from c_wrap import *

def parallelTable(func, *args):
  p = cpu_count()
  x = map(tuple, zip(*args))

  with Pool(p) as workers:
    y = workers.starmap(func, x, int(ceil(len(args[0]) / p)))

  return y

N = 1<<14
w, E, mu, beta, a = 3, 6e2, -1, 2, -0.1

"""
t0 = time.time()
for i in range(N):
  r = invTmatrixMB_real(w, E, mu, beta, a)
  #r = I2(w, E, mu, beta)
dt = time.time() - t0

print("result: (%.10f, %.10f)" % (real(r), imag(r)));
print("(%d) %0.3f μs" %( N, dt / N * 1e6));
exit()
"""

#"""
x = linspace(0, 1e3, N)

t0 = time.time()
y = parallelTable(I2, x, itertools.repeat(E, N), itertools.repeat(mu, N), itertools.repeat(beta, N))
dt = time.time() - t0

fig, axarr, axarr0 = complexPlot(x, y)
#axarr.set_ylim(-0.05, 0.05)

print('\n'.join([str(i) for i in y if real(i) > 1]))

print("(%d) %0.3f μs" % (N, dt * 1e6 / N));

plt.show()
#"""


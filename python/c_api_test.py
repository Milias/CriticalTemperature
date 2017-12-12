from common import *
from c_wrap import *

def parallelTablev2(x, func, p = cpu_count()):
  with Pool(p) as workers:
    y = workers.map(func, x, int(ceil(len(x) / p)))

  return y

N = 1<<20
w, E, mu, beta, a = 3, 0, -1, 2, -0.01

"""
t0 = time.time()
for i in range(N):
  #r = polePos(E, mu, beta, a)
  r = I2(w, E, mu, beta)
dt = time.time() - t0

print("result: (%.10f, %.10f)" % (real(r), imag(r)));
print("(%d) %0.3f μs" %( N, dt / N * 1e6));
"""

#"""
ones_arr = ones(N)
x = linspace(0, 10, N)

def f(x):
  r = 0
  while r < 1000*x:
    r += 1
  return r

t0 = time.time()
y = parallelTable([x], f)
dt = time.time() - t0

fig, axarr, axarr0 = complexPlot(x, y)

print("(%d) %0.3f μs" %( N, dt * 1e6));

plt.show()
#"""


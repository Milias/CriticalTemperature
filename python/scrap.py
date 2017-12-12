from common import *

### Integrals ###
from I1 import *
from I2 import *

### Density computation ###
from analytic_n import *
from spec_func_full import *

N = 1<<2
ones_arr = ones(N)

w, E, mu, beta, a = 3, 0, -1, 2, -0.01

z1 = 0.5 * E - 2 * mu
z2 = 0.5 * w - 0.5 * z1

print((z1, z2))

#"""
t0 = time.time()
for i in range(N):
  r = polePos(E, mu, beta, a)
dt = time.time() - t0

print("result: (%.10f, %.10f)" % (real(r), imag(r)));
print("(%d) %0.3f ms" %( N, dt / N * 1e3));
exit()
#"""

#print(integralBranch(E, mu, beta, a))
#exit()

#"""
x = linspace(z1 + 1e-14 if z1 > 0 else - 1e-10, 4 * z1, N)
t1 = time.time()
y = parallelTable([x, E * ones_arr, mu * ones_arr, beta * ones_arr, a * ones_arr], integrandBranch)
#y2 = parallelTable([x, E * ones_arr, mu * ones_arr], I1)
print((time.time() - t1)/N*cpu_count())

fig, axarr, axarr0 = complexPlot(x, y)
#axarr.set_ylim(0, 1)
#axarr.plot(x, y2, 'g-')
#"""

"""
x = linspace(-1, 1, N)
y = parallelTable([E * ones_arr, mu * ones_arr, beta * ones_arr, x], poleRes)

x = x[y > 0]
y = y[y > 0]

fig, axarr, axarr0 = complexPlot(x, y)
"""

"""
x = linspace(-1, 1, N)
y = parallelTable([E * ones_arr, mu * ones_arr, beta * ones_arr, x], polePos)

x = x[y != None]
y = y[y != None]

fig, axarr, axarr0 = complexPlot(x, y)
axarr.plot(x[x>0], z1 - 2 * x[x>0]**2, 'r--')
axarr.axhline(y = z1, color = 'g', ls = '--')
"""

"""
z1 = computeFullPole(E, mu, beta, a)
z1Line = True

if z1 == None:
  print('No pole found')
  z1 = 2 * z0 if z0 < 0 else 0
  z1Line = False

x = linspace( min(2 * z0 - 4 * a**2, 2 * z1 - z0), z0 - 1e-10, N)
y1 = parallelTable([x, E * ones_arr, mu * ones_arr, beta * ones_arr, a * ones_arr], specFuncPolesDmu)
fig, axarr, axarr0 = complexPlot(x, y1)
axarr.axvline(x = z0, color = 'g', ls = '--')
axarr.axvline(x = z0 - 2 * a**2, color = 'r', ls = '--')
if z1Line:
  axarr.axvline(x = z1, color = 'k', ls = '--')
"""

plt.show()


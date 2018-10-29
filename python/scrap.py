from common import *

def kernel(x, s):
  return 1

def f(x):
  return 2

def quad_arg(s, u, x):
  return kernel(x, s) * u(s)

def sol_iter(c, u, x):
  return c * scipy.integrate.quad(quad_arg, 0, 1, args = (u, x))[0] + f(x)

def sol_matrix(c, u, x):
  return 1 - c
c = 2
x_vec = linspace(0, 1, 100)
s_vec = linspace(0, 1, 100)

X, S = meshgrid(x_vec, s_vec)

result = []
u_func = f
for i in range(20):
  print(i)
  u_res = array([sol_iter(c, u_func, x) for x in x_vec])
  u_func = scipy.interpolate.interp1d(x_vec, u_res, kind = 'linear')

  print(u_res)

  result.append(u_res)

for n, res in enumerate(result):
  plt.plot(x_vec, res, '.-', label = n)

plt.legend(loc = 0)
plt.show()


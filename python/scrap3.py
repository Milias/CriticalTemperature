from common import *

def rho(mu, a):
  return 0.5 / mu / (log(a / sqrt(-mu)) - euler_gamma + 0.5j * pi)

N = 1<<8

mu, a = -1, 1

x = linspace(-1, 1, N)
y = imag(rho(mu, x))

plt.plot(x, y, 'r-')
plt.show()


from common import *

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K
sys = system_data(m_e, m_h, eps_r, T)

def plot_density(n, sys):
  print('n: %f' % n)
  a_arr = linspace(-5, 40, 100)
  mu_e = [analytic_mu_init_mu(n, a, sys) for a in a_arr]
  y = array([analytic_mu_f(mu_e, a, n, sys) for a, mu_e in zip(a_arr, mu_e)])

  plt.plot(a_arr, y[:, 0], 'r.-', label = n)
  plt.plot(a_arr, y[:, 1], 'b.--', label = n)
  plt.plot(a_arr, mu_e, 'g--', label = n)

  #plt.axis([a_arr[0], a_arr[-1], -5, 20])

def plot_mu_eqs(a, sys):
  N = 400
  n_arr = logspace(log10(1e23), log10(1e25), N) * sys.lambda_th**3
  mu_e = [analytic_mu_init_mu(n, a, sys) for n in n_arr]
  y = array([analytic_mu_f(mu_e, a, n, sys) for n, mu_e in zip(n_arr, mu_e)])

  plt.semilogx(n_arr * sys.lambda_th**-3, y[:, 0], '.-')
  plt.semilogx(n_arr * sys.lambda_th**-3, y[:, 1], '.--')

  plt.axis([n_arr[0] * sys.lambda_th**-3, n_arr[-1] * sys.lambda_th**-3, -50, 50])

def plot_ls(n, sys):
  a_arr = linspace(-5, 40, 400)
  mu_e = [analytic_mu_init_mu(n, a, sys) for a in a_arr]
  y = array([ideal_ls(2*mu_e, sys) for mu_e in mu_e])

  plt.plot(a_arr, y, '.-')

def plot_mu(n, a, sys):
  N = 100
  mu_e_init = analytic_mu_init_mu(n, a, sys)
  print(mu_e_init)
  mu_e = linspace(2 * mu_e_init, mu_e_init, N)

  y = array([analytic_mu_f(mu_e, a, n, sys) for mu_e in mu_e])

  #plt.plot(mu_e, y[:, 0], 'r.-')
  plt.plot(mu_e, y[:, 1], 'b.-')

def plot_steps(axarr, n, x_init, sys):
  y = array(analytic_mu_follow(n, x_init, sys))
  x = range(len(y) // 4)

  print(y[-4:])
  print('%e' % (abs(y[-1]) + abs(y[-3])))
  print(len(y) // 4)

  axarr[0].plot(y[::4], y[2::4], '.--')
  axarr[0].plot([x_init[0]], [x_init[1]], 'kx')
  axarr[0].plot([y[-4]], [y[-2]], 'ko')

  axarr[0].set_ylabel('Scattering length')
  axarr[0].set_xlabel('Chemical potential')

  axarr[1].axhline(y = 0, color = 'k', linestyle = '-')
  axarr[1].axvline(x = 0, color = 'k', linestyle = '-')

  axarr[1].plot(y[1::4], y[3::4], '.--')
  axarr[1].plot([y[1]], [y[3]], 'kx')
  axarr[1].plot([y[-3]], [y[-1]], 'ko')

  axarr[1].set_xlabel('Equation of state')
  axarr[1].set_ylabel('Self-consistency condition')

  #print(y.reshape((len(x), 4))[:,::2])

#plt.axhline(y = 0, color = 'k', linestyle = '-')
#plt.axvline(x = 0, color = 'k', linestyle = '-')
"""
n_arr = logspace(log10(3.9e23), log10(4e23), 8) * sys.lambda_th**3
for n in n_arr:
  plot_density(n)

plt.legend(loc = 0)
"""
"""

for a in [-1, 1, 2, 3, 6, 10]:
  plot_mu(n = 3.62e23, a = a, sys = sys)

plt.show()
exit()

a = -1
plot_mu_eqs(a, sys)
plt.show()
exit()

a = 60
n = 3.97e23 * sys.lambda_th**3
#print(analytic_mu_f(analytic_mu_init_mu(n, a, sys), a, n, sys))
#exit()

plot_ls(n, sys)
plt.show()
exit()

plot_density(n, sys)
plt.show()
exit()
"""

fig, axarr = plt.subplots(1, 2, figsize = (18, 6), dpi = 96)

N = 10
M = 10
n = 7.97e23 * sys.lambda_th**3
a_v =  linspace(1, 11, N)
mu_e_v = array([ideal_mu(n, sys.m_pe) for a in a_v])
f = repeat(linspace(1, 2, M).reshape(1, M), N, 0)

mu_e = (f * repeat(mu_e_v.reshape(N, 1), M, 1)).flatten()
a = repeat(a_v, M)

params = stack((mu_e, a)).T

for mu_e, a in params:
  plot_steps(axarr, n, (mu_e, a), sys)

plt.show()


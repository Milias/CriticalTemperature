from common import *

N = 1<<17
bs = 1<<8

m_e, m_h = 0.28 * m_electron, 0.59 * m_electron # eV
m_p = 1.0 / (1.0 / m_e + 1.0 / m_h) # eV
m_m = 1.0 / (1.0 / m_h - 1.0 / m_e) # eV

T = 300 # K
beta = 1 / (k_B * T) # eV^-1
lambda_th = c * hbar * sqrt(2 * pi * beta / m_p) # m
energy_th = 1 / ( 4 * pi * beta )
eps_r, e_ratio = 6.56, m_p / energy_th
mr_ep = m_p / m_e
mr_hp = m_p / m_h

print('%f nm, %f meV' % (lambda_th * 1e9, energy_th * 1e3))

n_dless = 3e24 * lambda_th**3

E = 10
a = 5
#mu_e, mu_h = ideal_mu(n_dless, mr_ep), ideal_mu(n_dless, mr_hp)
#mu_e, mu_h, a = analytic_mu(n_dless, 1/mr_ep, 1/mr_hp, eps_r, e_ratio)
mu_e, mu_h = 3, 3
z1 = 0.25 * (1 - (mr_hp - mr_ep)**2) * E - mu_e - mu_h
print('z1: %f' % z1)

print('%.3f meV, %.3f meV, %.3f, %.3f, %.3f' % (mu_e * energy_th * 1e3, mu_h * energy_th * 1e3, mu_e, mu_h, a))

z0 = fluct_pf(a, E, mr_ep, mr_hp, mu_e, mu_h)
print(z0)

#exit()

#x = linspace(0.7 * z0, 1.3 * z0 if 1.3 * z0 < z1 else z1, N)
#x = linspace(0.999 * z1, z1, N)
x = linspace(-20, 30, N)
#x = logspace(log10(1e22), log10(2e24), N) * lambda_th**3
#x = range(0, N)

"""
y = array(parallelTable(
  fluct_pf,
  x,
  itertools.repeat(E, N),
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N)
))
"""

y = array(parallelTable(
  fluct_pr,
  x,
  itertools.repeat(E, N),
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

y2 = array(parallelTable(
  analytic_prf,
  x,
  itertools.repeat(E, N),
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

#x *= lambda_th**-3

#y = 2 * (y - y2)/(y + y2)
#y2 = - x**2 / 4 + z1

#y = 0.5 * pi * a - pi * sqrt(z1 - x) + y

plot_type = 'plot'

axplots = []
fig, axarr = plt.subplots(1, 1, sharex = True, figsize = (16, 5), dpi = 96)
#fig.subplots_adjust(hspace=0)
fig.tight_layout()

axplots.append(getattr(axarr, plot_type)(x, y, 'r-'))
axarr.autoscale(enable = True, axis = 'x', tight = True)
axarr.plot(x, y2, 'b--')
#axarr.axvline(x = z1 - 0.25 * a * a, linestyle = '-', color = 'k', linewidth = 0.5)
#axarr.axvline(x = fluct_pf(a, E, mr_ep, mr_hp, mu_e, mu_h), linestyle = '-', color = 'g', linewidth = 1)
axarr.axhline(y = 0, linestyle = '-', color = 'k', linewidth = 0.5)
if x[0] < 0 and x[-1] > 0:
  axarr.axvline(x = 0, linestyle = '-', color = 'k', linewidth = 0.5)

#axarr.set_ylim(15, y[-1])
#axarr.set_yscale('symlog')

#axarr.set_xticks([0.0], minor = False)
#axarr.set_yticks([0.0], minor = False)
#axarr.grid(color = 'k', linestyle = '-', linewidth = 0.5)

plt.show()


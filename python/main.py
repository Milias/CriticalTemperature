from common import *

N = 1<<14
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

E = 130
mu_e, mu_h = ideal_mu(n_dless, mr_ep), ideal_mu(n_dless, mr_hp)
#mu_e, mu_h = -10, -20
z1 = 0.25 * (1 - (mr_hp - mr_ep)**2) * E - mu_e - mu_h

a = -6

print('%.3f meV, %.3f meV, %.3f, %.3f' % (mu_e * energy_th * 1e3, mu_h * energy_th * 1e3, mu_e, mu_h))

print(fluct_pf(a, E, mr_ep, mr_hp, mu_e, mu_h))
#exit()

#x = linspace(0.25 * z1, z1, N)
x = linspace(-20, 20, N)
#x = range(0, N)

t0 = time.time()

"""
y = array(parallelTable(
  fluct_es_f,
  x,
  itertools.repeat(E, N),
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N)
))
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

dt = time.time() - t0

#y = 0.5 * pi * a - pi * sqrt(z1 - x) + y

plot_type = 'plot'

axplots = []
fig, axarr = plt.subplots(1, 1, sharex = True, figsize = (8, 5), dpi = 96)
#fig.subplots_adjust(hspace=0)
fig.tight_layout()

axplots.append(getattr(axarr, plot_type)(x, y, 'r-'))
axarr.autoscale(enable = True, axis = 'x', tight = True)
axarr.plot(x, z1 - 0.25 * x**2, 'b--')
#axarr.axvline(x = z1 - 0.25 * a * a, linestyle = '-', color = 'k', linewidth = 0.5)
#axarr.axvline(x = fluct_pf(a, E, mr_ep, mr_hp, mu_e, mu_h), linestyle = '-', color = 'g', linewidth = 1)
axarr.axhline(y = 0, linestyle = '-', color = 'k', linewidth = 0.5)
axarr.axvline(x = 0, linestyle = '-', color = 'k', linewidth = 0.5)
#axarr.set_ylim(-10, 10)
#axarr.set_yscale('symlog')

#axarr.set_xticks([0.0], minor = False)
#axarr.set_yticks([0.0], minor = False)
#axarr.grid(color = 'k', linestyle = '-', linewidth = 0.5)

print("(%d) %.3f μs, %.3f s" % (N, dt * 1e6 / N, dt));
plt.show()


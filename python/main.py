from common import *

N = 1<<10
bs = 1<<3

m_e, m_h = 0.28 * m_electron, 0.59 * m_electron # eV
#m_e, m_h = 1.1 * m_electron, 0.1 * m_electron # eV
#m_e, m_h = 1.0 * m_electron, 1.0 * m_electron # eV
m_p = 1.0 / (1.0 / m_e + 1.0 / m_h) # eV

T = 300 # K
beta = 1 / (k_B * T) # eV^-1
lambda_th = c * hbar * sqrt(2 * pi * beta / m_p) # m
energy_th = 1 / ( 4 * pi * beta )
eps_r, e_ratio = 6.56, m_p / energy_th
mr_ep = m_p / m_e
mr_hp = m_p / m_h

print('%f nm, %f meV' % (lambda_th * 1e9, energy_th * 1e3))

n_dless = 3e24 * lambda_th**3

#mu_e, mu_h = ideal_mu(n_dless, mr_ep), ideal_mu(n_dless, mr_hp)
#mu_e, mu_h, a = analytic_mu(n_dless, 1/mr_ep, 1/mr_hp, eps_r, e_ratio)
mu_e, mu_h = -10, -1

E = 0
ac = fluct_ac(mr_ep, mr_hp, mu_e, mu_h)
ac_E = fluct_ac_E(E, mr_ep, mr_hp, mu_e, mu_h)
a = 4

Ec_a = fluct_Ec_a(a, mr_ep, mr_hp, mu_e, mu_h)

z1 = 0.25 * (1 - (mr_hp - mr_ep)**2) * E - mu_e - mu_h
print('z1: %f, %f' % (z1, 0.25 * (1 - (mr_hp - mr_ep)**2)))

print('mu_e: %.3f meV, mu_h: %.3f meV, mu_e: %.3f, mu_h: %.3f, ac: %.3f, ac_E: %.3f' % (mu_e * energy_th * 1e3, mu_h * energy_th * 1e3, mu_e, mu_h, ac, ac_E))

pp0_E = fluct_pp0(a, mr_ep, mr_hp, mu_e, mu_h)

z0 = fluct_pp(a, E, mr_ep, mr_hp, mu_e, mu_h)
print('z0: %f' % z0)

a_min, a_max = 2 * sqrt(abs(mu_e+mu_h)) + ac, 2 * sqrt(abs(mu_e+mu_h))
ac_max = fluct_pp0c(mr_ep, mr_hp, mu_e, mu_h)

print(fluct_t_c(0, E, mr_ep, mr_hp, mu_e, mu_h, a))
#print(fluct_pmi(a, mr_ep, mr_hp, mu_e, mu_h))

exit()

#x = linspace(0.7 * z0, 1.3 * z0 if 1.3 * z0 < z1 else z1, N)
#x = linspace(z1 - 1e-3, z1 - 1e-4, N)
#x = linspace(a_c, 0.5, N)
#x = logspace(log10(1e22), log10(2e24), N) * lambda_th**3
#x = range(0, N)
x = linspace(ac, ac_max, N)
#x = linspace(0, 1e2, N)
#x = linspace(0, Ec_a if Ec_a < float('inf') else 1e2, N)

"""
pp = array(parallelTable(
  fluct_pp_b,
  x,
  itertools.repeat(0, N),
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))
"""

y = array(parallelTable(
  fluct_pmi_nc,
  x,
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

"""
y = array(parallelTable(
  fluct_pr,
  itertools.repeat(a, N),
  x,
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

"""
"""
y2 = array(parallelTable(
  fluct_pp_b,
  itertools.repeat(a, N),
  x,
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))


ac_E_vec = array(parallelTable(
  fluct_ac_E,
  x,
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

y = sqrt(x) * array(parallelTable(
  fluct_pr,
  0.5 * ac_E_vec,
  x,
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))


y2 = array(parallelTable(
  fluct_pp_b,
  x,
  itertools.repeat(E, N),
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))


y2 = sqrt(x) * array(parallelTable(
  analytic_prf,
  itertools.repeat(-0.5, N),
  x,
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

y = array(parallelTable(
  fluct_pp,
  itertools.repeat(a, N),
  x,
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

y = array(parallelTable(
  fluct_pr,
  itertools.repeat(a, N),
  x,
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

y2 = array(parallelTable(
  analytic_prf,
  itertools.repeat(a, N),
  x,
  itertools.repeat(mr_ep, N),
  itertools.repeat(mr_hp, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))
"""

#x *= lambda_th**-3

#y = 2 * (y - y2)/(y + y2)
#y2 = -(x - ac_E)**2 / 4 + z1
#y3 = z1 - 3 * (x - ac_E)**2 / 16
#y3 = 0.5 * (z1 + y2)
#y2 = - ac * exp(log(ac_E/ac) / E * x)

#y2 = ( 4 * abs(mu_e + mu_h) + (x - ac)**2) / (1 - (mr_ep - mr_hp)**2) + 1
#y3 = zeros_like(x)

#y = 0.5 * pi * a - pi * sqrt(z1 - x) + y

plot_type = 'plot'

axplots = []
fig, axarr = plt.subplots(1, 1, sharex = True, figsize = (16, 5), dpi = 96)
#fig.subplots_adjust(hspace=0)
fig.tight_layout()

axplots.append(getattr(axarr, plot_type)(x, y, 'r-'))
axarr.autoscale(enable = True, axis = 'x', tight = True)
#axarr.plot(x, y2, 'b--')
#axarr.plot(x, y3, 'r--')
#axarr.plot(x, pp, 'g--')
#axarr.axvline(x = z1 - 0.25 * a * a, linestyle = '-', color = 'k', linewidth = 0.5)
#axarr.axhline(y = z1, linestyle = '-', color = 'g')
axarr.axvline(x = ac, linestyle = '-', color = 'g')
#axarr.axvline(x = a_min, linestyle = '-', color = 'r')
#axarr.axvline(x = a_max, linestyle = '-', color = 'b')
axarr.axvline(x = ac_max, linestyle = '--', color = 'g')
#axarr.axvline(x = pp0_E, linestyle = '--', color = 'g')
#axarr.axvline(x = ac_E, linestyle = '--', color = 'g')
#axarr.axvline(x = fluct_pf(a, E, mr_ep, mr_hp, mu_e, mu_h), linestyle = '-', color = 'g', linewidth = 1)

if nanmax(y) >= 0 and nanmin(y) <= 0:
  axarr.axhline(y = 0, linestyle = '-', color = 'k', linewidth = 0.5)

if x[0] <= 0 and x[-1] >= 0:
  axarr.axvline(x = 0, linestyle = '-', color = 'k', linewidth = 0.5)

#axarr.set_ylim(-25, 3)
axarr.set_xlim(x[0], x[-1])
#axarr.set_yscale('symlog')

#axarr.set_xticks([0.0], minor = False)
#axarr.set_yticks([0.0], minor = False)
#axarr.grid(color = 'k', linestyle = '-', linewidth = 0.5)

plt.show()


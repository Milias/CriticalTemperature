from common import *

N = 1<<12
bs = 1<<2

m_e, m_h = 0.28 * m_electron, 0.59 * m_electron # eV
#m_e, m_h = 1.1 * m_electron, 0.1 * m_electron # eV
#m_e, m_h = 1.0 * m_electron, 1.0 * m_electron # eV
m_p = 1.0 / (1.0 / m_e + 1.0 / m_h) # eV

T = 300 # K
beta = 1 / (k_B * T) # eV^-1
lambda_th = c * hbar * sqrt(2 * pi * beta / m_p) # m
energy_th = 1 / ( 4 * pi * beta )
eps_r, e_ratio = 6.56, m_p / energy_th
m_pe = m_p / m_e
m_ph = m_p / m_h

print('%f nm, %f meV' % (lambda_th * 1e9, energy_th * 1e3))

n = 3e24 * lambda_th**3

mu_e, mu_h = ideal_c_mu(n, m_pe, m_ph)
#mu_e, mu_h, a = analytic_mu(n, 1/m_pe, 1/m_ph, eps_r, e_ratio)
#mu_e, mu_h = -2e2, -2e2

z = 0
E = 1
ac = fluct_ac(m_pe, m_ph, mu_e, mu_h)
ac_E = fluct_ac_E(E, m_pe, m_ph, mu_e, mu_h)
a = -200

Ec_a = fluct_Ec_a(a, m_pe, m_ph, mu_e, mu_h)

z1 = 0.25 * (1 - (m_ph - m_pe)**2) * E - mu_e - mu_h
print('z1: %f, n: %f' % (z1, n))

print('mu_e: %.3f meV, mu_h: %.3f meV, mu_e: %.3f, mu_h: %.3f, ac: %.3f, ac_E: %.3f' % (mu_e * energy_th * 1e3, mu_h * energy_th * 1e3, mu_e, mu_h, ac, ac_E))

z0 = fluct_pp(a, E, m_pe, m_ph, mu_e, mu_h)

a_min, a_max = 2 * sqrt(abs(mu_e+mu_h)) + ac, 2 * sqrt(abs(mu_e+mu_h))
ac_max = fluct_pp0c(m_pe, m_ph, mu_e, mu_h)
print('z0: %f, ac_max: %f' % (z0, ac_max))

#print(analytic_mu_param_b(n, m_pe, m_ph, a))
#print(fluct_mu_a(n, a, m_pe, m_ph))
#exit()

#x = linspace(0.7 * z0, 1.3 * z0 if 1.3 * z0 < z1 else z1, N)
#x = linspace(z1 - 1e-3, z1 - 1e-4, N)
#x = linspace(a_c, 0.5, N)
#x = logspace(log10(1e22), log10(2e24), N) * lambda_th**3
#x = range(0, N)
x = linspace(-1e2, 70, N)
#x = linspace(0, Ec_a, N)
#x = linspace(z1, 2 * z1, N)
#x = linspace(0, Ec_a if Ec_a < float('inf') else 1e2, N)

y = zeros_like(x)
y2 = zeros_like(x)
y3 = zeros_like(x)
y4 = zeros_like(x)

y = array(parallelTable(
  analytic_mu_param_b,
  itertools.repeat(n, N),
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  x,
  bs = bs
))

y2 = y[:, 1]
y = y[:, 0]
y3[:] = ideal_mu(n, m_pe)
y4[:] = ideal_mu(n, m_ph)


"""
pp = array(parallelTable(
  fluct_pp_b,
  x,
  itertools.repeat(0, N),
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))
"""

"""
y = array(parallelTable(
  fluct_pp0c,
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  x,
  itertools.repeat(mu_h, N),
  bs = bs
))

y2 = array(parallelTable(
  fluct_ac,
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  x,
  itertools.repeat(mu_h, N),
  bs = bs
))

y = array(parallelTable(
  fluct_bmi,
  x,
  #itertools.repeat(0.9 * Ec_a, N),
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  #itertools.repeat(a, N),
  bs = bs
))

y2 = array(parallelTable(
  fluct_pmi,
  x,
  itertools.repeat(ac_max, N),
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

n_id = (analytic_n_id(mu_e, 1/m_pe) + analytic_n_id(mu_h, 1/m_ph))
print('n_id: %f' % n_id)

y_total = y + y2 + n_id
y3 = y2 / y_total
y2 = n_id / y_total
y4 = y / y_total
y = y3 + y4
"""
"""

y2 = array(parallelTable(
  fluct_bfi_spi_d,
  x,
  itertools.repeat(E, N),
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  itertools.repeat(a, N),
  bs = bs
))

y *= amax(abs(y2)) / amax(abs(y))

y2 = array(parallelTable(
  analytic_n_sc,
  itertools.repeat(mu_e + mu_h, N),
  itertools.repeat(m_pe + m_ph, N),
  x,
  bs = bs
))

y2 = array(parallelTable(
  analytic_n_ex,
  itertools.repeat(mu_e + mu_h, N),
  itertools.repeat(m_pe + m_ph, N),
  x,
  bs = bs
))
"""
"""
y2 = array(parallelTable(
  fluct_pmi_nc,
  x,
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

#print(y)

n_id = analytic_n_id(mu_e, 1/m_pe) + analytic_n_id(mu_h, 1/m_ph)

#y = (y + y2) / (y + y2 + n_id)


y = array(parallelTable(
  fluct_bfi_spi,
  x,
  itertools.repeat(E, N),
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  itertools.repeat(a, N),
  bs = bs
))

y2 = array(parallelTable(
  fluct_i_c,
  x,
  itertools.repeat(0, N),
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

y = array(parallelTable(
  fluct_pmi_nc,
  x,
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

y = array(parallelTable(
  fluct_pr,
  itertools.repeat(a, N),
  x,
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
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
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))


ac_E_vec = array(parallelTable(
  fluct_ac_E,
  x,
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

y = sqrt(x) * array(parallelTable(
  fluct_pr,
  0.5 * ac_E_vec,
  x,
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))


y2 = array(parallelTable(
  fluct_pp_b,
  x,
  itertools.repeat(E, N),
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))


y2 = sqrt(x) * array(parallelTable(
  analytic_prf,
  itertools.repeat(-0.5, N),
  x,
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

y = array(parallelTable(
  fluct_pp,
  itertools.repeat(a, N),
  x,
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

y = array(parallelTable(
  fluct_pr,
  itertools.repeat(a, N),
  x,
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  bs = bs
))

y2 = array(parallelTable(
  analytic_prf,
  itertools.repeat(a, N),
  x,
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
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
#y2 = 0.25 * (1 - (m_ph - m_pe)**2) * x - mu_e - mu_h

#y2 = ( 4 * abs(mu_e + mu_h) + (x - ac)**2) / (1 - (m_pe - m_ph)**2) + 1
#y3 = zeros_like(x)

#y = 0.5 * pi * a - pi * sqrt(z1 - x) + y

plot_type = 'plot'

axplots = []
fig, axarr = plt.subplots(1, 1, sharex = True, figsize = (16, 5), dpi = 96)
#fig.subplots_adjust(hspace=0)
fig.tight_layout()

axplots.append(getattr(axarr, plot_type)(x, real(y), 'r-'))
axarr.autoscale(enable = True, axis = 'x', tight = True)
#axarr.plot(x, imag(y), 'b-')
#axarr.plot(x, real(y2), 'g--')
#axarr.plot(x, imag(y2), 'b--')
axarr.plot(x, y2, 'b-')
axarr.plot(x, y3, 'r--')
axarr.plot(x, y4, 'b--')
#axarr.plot(x, pp, 'g--')
#axarr.axvline(x = z1 - 0.25 * a * a, linestyle = '-', color = 'k', linewidth = 0.5)
#axarr.axhline(y = z0, linestyle = '-', color = 'g')
#axarr.axhline(y = ac, linestyle = '-', color = 'g')
#axarr.axhline(y = ac_max, linestyle = '-', color = 'g')
#axarr.axvline(x = z0, linestyle = '--', color = 'g', linewidth = 1)
#axarr.axvline(x = ac, linestyle = '-.', color = 'g')
#axarr.axvline(x = ac_max, linestyle = '--', color = 'g')
#axarr.axvline(x = a_min, linestyle = '-', color = 'r')
#axarr.axvline(x = a_max, linestyle = '-', color = 'b')
#axarr.axvline(x = pp0_E, linestyle = '--', color = 'g')
#axarr.axvline(x = ac_E, linestyle = '--', color = 'g')

"""
for c, Eb in zip(('g', 'b', 'b', 'g'), fluct_i_c_fbv(z, E, m_ph, mu_e + mu_h)):
  axarr.axvline(x = Eb, linestyle = '--', color = c)

for c, Eb in zip(('g', 'r', 'r', 'g'), fluct_i_c_fbv(z, E, m_pe, mu_e + mu_h)):
  axarr.axvline(x = Eb, linestyle = '--', color = c)
"""

#axarr.axvline(x = fluct_pf(a, E, m_pe, m_ph, mu_e, mu_h), linestyle = '-', color = 'g', linewidth = 1)

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

#fig.savefig('bin/plots/fluct_n_ex_sc_v4.eps')

plt.show()


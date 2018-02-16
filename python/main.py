from common import *

N = 1<<7
bs = 1<<0

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
m_sigma = 1/m_pe + 1/m_ph

print('%f nm, %f meV' % (lambda_th * 1e9, energy_th * 1e3))

n = 8e24 * lambda_th**3
a = 5

mu_e = ideal_mu(n, m_pe)
mu_h = ideal_mu(n, m_ph)
print('mu_e: %.3f, mu_h: %.3f, mu_t: %.3f\n' % (mu_e, mu_h, mu_e + mu_h))

#exit()

#x = logspace(log10(1e22), log10(7e24), N) * lambda_th**3
x = linspace(10, 20, N)

y = zeros_like(x)
y2 = zeros_like(x)
y3 = zeros_like(x)
y4 = zeros_like(x)

pp0c_mu = array(parallelTable(
  fluct_pp0c_mu,
  x[x>=0],
  itertools.repeat(n, len(x[x>=0])),
  itertools.repeat(m_pe, len(x[x>=0])),
  itertools.repeat(m_ph, len(x[x>=0])),
  bs = bs
))

y3 = -0.25 *x**2 *(x>=0) + 4*pi*invPolylogExp(1.5, 0.25 * m_sigma**-1.5 * n)
y4[x<0] = ideal_mu(n, m_pe)
y4[x>=0] = minimum(ones_like(pp0c_mu) * ideal_mu(n, m_pe), pp0c_mu)

"""
y = array(parallelTable(
  fluct_mu_a,
  itertools.repeat(n, N),
  x,
  itertools.repeat(m_pe, N),
  itertools.repeat(m_ph, N),
  bs = bs
))
"""

data = loadData('bin/data/data_fluct_mu_a_1518779889003259.json.gz')
y = array(data['result'])

y2 = y[:, 1]
y = y[:, 0]

#x *= lambda_th**-3

plot_type = 'plot'

axplots = []
fig, axarr = plt.subplots(1, 1, sharex = True, figsize = (16, 5), dpi = 96)
#fig.subplots_adjust(hspace=0)
fig.tight_layout()

axplots.append(getattr(axarr, plot_type)(x, real(y), 'r-', marker = '.'))

axarr.autoscale(enable = True, axis = 'x', tight = True)

axarr.plot(x, imag(y), 'b-', marker = '.')
axarr.plot(x, real(y2), 'r--', marker = '.')
axarr.plot(x, imag(y2), 'b--', marker = '.')
axarr.plot(x, y3, 'g-', marker = '.')
axarr.plot(x, y4, 'm-', marker = '.')

#axarr.axhline(y = z0, linestyle = '-', color = 'g')
#axarr.axvline(x = z0, linestyle = '--', color = 'g', linewidth = 1)

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

t_now = time.time()
fig.savefig('bin/plots/saved_%d.eps' % t_now)
fig.savefig('bin/plots/saved_%d.png' % t_now)

plt.show()


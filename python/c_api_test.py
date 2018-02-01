from common import *
from c_wrap import *

#print(os.getpid())

def parallelTable(func, *args):
  p = cpu_count()
  x = map(tuple, zip(*args))

  with Pool(p) as workers:
    y = workers.starmap(func, x, 1)

  return y

def compExcitonicDensity(mu, beta, a):
  t0 = time.time()
  y = array(parallelTable(integralDensityPole, mu, itertools.repeat(beta, N), itertools.repeat(a, N)))
  dt = time.time() - t0

  print("(%d) %.3f μs, %.3f s" % (N, dt * 1e6 / N, dt));
  return y

k_B = 8.6173303e-5 # eV K^-1
m_electron = 0.5109989461e6 # eV
hbar = 6.582119514e-16 # eV s
c = 299792458 # m s^-1

N = 1<<8
w, E, mu, beta, a = -1, 0, -1, 0.1, -1

m_1, m_2 = 0.28 * m_electron, 0.59 * m_electron # eV
m_r = 1.0 / (1.0 / m_1 + 1.0 / m_2) # eV

T = 300 # K
beta = 1 / (k_B * T) # eV^-1
lambda_th = c * hbar * sqrt(2 * pi * beta / m_r) # m
energy_th = 1 / ( 4 * pi * beta )
eps_r, e_ratio = 6.56, m_r / energy_th
m_ratio_e = m_1 / m_r
m_ratio_h = m_2 / m_r

print('%f nm, %f meV' % (lambda_th * 1e9, energy_th * 1e3))

"""
pole_pos = polePos(E, mu, beta, a)
pole_pos_last = findLastPos(mu, beta, a) - 1e-10
dz = 0.5 * E - 2 * mu - pole_pos
print(pole_pos)
"""

"""
t0 = time.time()
for i in range(N):
  #r = complex(invTmatrixMB_real(w, E, mu, beta, a), a + I1(0.25 * E - mu - 0.5 * w))
  #r = polePos(E, mu, beta, a)
  #r = integralBranch(E, mu, beta, a)
  #r = integralDensityPole(mu, beta, a)
  #r = integralDensityBranch(mu, beta, a)
dt = time.time() - t0

print("result: (%.10f, %.10f)" % (real(r), imag(r)));
print("(%d) %.3f μs, %.3f s" % (N, dt * 1e6 / N, dt));
exit()
"""

x = logspace(log10(3e23), log10(3e24), N) * lambda_th**3

t0 = time.time()

mu_arr = array(parallelTable(analytic_mu, x, itertools.repeat(m_ratio_e, N), itertools.repeat(m_ratio_h, N), itertools.repeat(eps_r, N), itertools.repeat(e_ratio, N)))

ideal_mu_arr_e = array(parallelTable(ideal_mu, x, itertools.repeat(m_ratio_e, N)))
ideal_mu_arr_h = array(parallelTable(ideal_mu, x, itertools.repeat(m_ratio_h, N)))

mu_total = mu_arr[:, 0] + mu_arr[:, 1]
ideal_mu_total = ideal_mu_arr_e + ideal_mu_arr_h

y_id = array(parallelTable(analytic_n_id, mu_arr[:,0], itertools.repeat(m_ratio_e, N))) + array(parallelTable(analytic_n_id, mu_arr[:,1], itertools.repeat(m_ratio_h, N)))

y_ex = array(parallelTable(analytic_n_ex, mu_total, itertools.repeat(m_ratio_e + m_ratio_h, N), mu_arr[:, 2]))
y_sc = array(parallelTable(analytic_n_sc, mu_total, itertools.repeat(m_ratio_e + m_ratio_h, N), mu_arr[:, 2]))
y_ex_norm = y_ex / (y_id + y_sc + y_ex)
y = (y_sc + y_ex) / (y_id + y_sc + y_ex)

dt = time.time() - t0

x *= lambda_th**-3

mu_arr[:, 0:2] *= energy_th
ideal_mu_arr_e[:] *= energy_th
ideal_mu_arr_h[:] *= energy_th

xi_vline = argmin((mu_arr[:, 2])**2)
x_vline = x[xi_vline]

y_ex_norm[xi_vline] = float('nan')
mu_arr[xi_vline, 2] = float('nan')

#plt.style.use('dark_background')

plot_type = 'semilogx'

axplots = []
fig, axarr = plt.subplots(3, 1, sharex = True, figsize = (8, 12), dpi = 96)
fig.subplots_adjust(hspace=0)
#fig.tight_layout()

axplots.append(getattr(axarr[0], plot_type)(x, y, 'r-', label = r'$(n_{ex} + n_{sc})/n$'))
axarr[0].autoscale(enable = True, axis = 'x', tight = True)
axarr[0].plot(x, y_ex_norm, 'm--', label = '$n_{ex}/n$')
axarr[0].set_ylabel('Density contributions, T = %.0f K' % T)
axarr[0].legend(loc = 0)
axarr[0].axvline(x = x_vline, linestyle = '-', color = 'g')

axplots.append(getattr(axarr[1], plot_type)(x, mu_arr[:, 0], 'r-', label = r'$m_e$'))
axarr[1].autoscale(enable = True, axis = 'x', tight = True)
axarr[1].plot(x, mu_arr[:, 1], 'b-', label = r'$m_h$')
axarr[1].plot(x, ideal_mu_arr_e, 'r--', label = r'$m_e$ (ideal)')
axarr[1].plot(x, ideal_mu_arr_h, 'b--', label = r'$m_h$ (ideal)')
axarr[1].set_ylabel(r'Chemical potential --- $\mu$ (eV)')
axarr[1].legend(loc = 0)
axarr[1].axvline(x = x_vline, linestyle = '-', color = 'g')

axplots.append(getattr(axarr[2], plot_type)(x, 1/mu_arr[:, 2], 'g-'))
axarr[2].autoscale(enable = True, axis = 'x', tight = True)
axarr[2].set_xlabel(r'$n$ (m$^{-3}$)')
axarr[2].set_ylabel(r'Scattering length --- $a/\Lambda_{th}$')
axarr[2].set_ylim(-50, 50)
axarr[2].axvline(x = x_vline, linestyle = '-', color = 'g')

fig.savefig('python/graphs/analytic_n_ex_sc_v3.eps')

print("(%d) %.3f μs, %.3f s" % (N, dt * 1e6 / N, dt));
plt.show()


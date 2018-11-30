from common import *

N_u0 = 1 << 10

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300  # K
sys = system_data(m_e, m_h, eps_r, T)
"""
mu_e = 1e1
mu_h = sys.m_eh * mu_e
"""

mu_e = 4 / sys.beta
mu_h = sys.get_mu_h_ht(mu_e)

k0, k1, w0 = 1, 1, 0.1

u0, du0 = linspace(0, pi, N_u0, retstep=True)

k_vec = sqrt(k0**2 + k1**2 - 2 * k0 * k1 * cos(u0))

wk_vec = list(itertools.product(itertools.repeat(w0, 1), k_vec))
"""
pole = plasmon_disp_th((w0, k0, k1), mu_e, mu_h, sys)
plt.axvline(x=pole, color='m', linestyle='-')
"""

plt.axhline(y=0, color='k', linestyle='-')

t0 = time.time()
green = plasmon_green_ht_v(wk_vec, mu_e, mu_h, sys)
green_inv = plasmon_green_ht_inv_v(wk_vec, mu_e, mu_h, sys)

print('[%e], Elapsed: %.2fs' % (mu_e, time.time() - t0))

green_arr = array(green)
green_inv_arr = array(green_inv)

#plt.plot(u0, real(green_arr), 'r-')
#plt.plot(u0, imag(green_arr), 'b-')

plt.plot(k_vec, real(green_inv_arr), 'r--')
plt.plot(k_vec, imag(green_inv_arr), 'b--')

plt.savefig('plots/green_cx_1d.eps')

plt.show()

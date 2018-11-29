from common import *

N_u0 = 1 << 13
N_u1 = 1 << 0

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

u_max = 1

mu_e = 1e1
mu_h = sys.m_eh * mu_e

k0, k1, w0 = 25, 10, 0

u0, du0 = linspace(0, pi, N_u0, retstep=True)

k_vec = sqrt(k0**2 + k1**2 - 2 * k0 * k1 * cos(u0))

wk_vec = list(itertools.product(itertools.repeat(w0, 1), k_vec))

pole = plasmon_disp_th((w0, k0, k1), mu_e, mu_h, sys)

print(pole)

plt.axhline(y=0, color='k', linestyle='-')
plt.axvline(x=pole, color='m', linestyle='-')

print(plasmon_potcoef((w0, k0, k1), mu_e, mu_h, sys))

t0 = time.time()
green = plasmon_green_inv_v(wk_vec, mu_e, mu_h, sys)

print('[%e], Elapsed: %.2fs' % (mu_e, time.time() - t0))

green_arr = array(green).reshape((N_u1, N_u0))

plt.plot(u0, real(green_arr[0, :]), 'r-')
plt.plot(u0, imag(green_arr[0, :]), 'b-')

plt.savefig('plots/green_cx_1d.eps')

plt.show()

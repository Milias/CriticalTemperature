from common import *

N_u0_lwl = 1 << 8

N_u0 = 1 << 11
#N_u1 = (1 << 2) + 1
N_u1 = (1 << 0)

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300  # K
sys = system_data(m_e, m_h, eps_r, T)

u_max = 1

u0_lwl, du0_lwl = linspace(
    1 / N_u0_lwl, 1 - 1 / N_u0_lwl, N_u0_lwl, retstep=True)

u0, du0 = linspace(1 / N_u0, 1 - 1 / N_u0, N_u0, retstep=True)
u1_nw, du1_nw = array([0.0]), float('nan')

if N_u1 > 1:
    u1, du1 = linspace(-1 + 1 / N_u1, 1 - 1 / N_u1, N_u1, retstep=True)
else:
    u1, du1 = array([0.0]), float('nan')

vu_vec = list(itertools.product(u0_lwl, repeat=2))
vuvu_vec = list(itertools.product(u0, u1, repeat=2))
vuvu_nw_vec = list(itertools.product(u0, u1_nw, repeat=2))

r_u0, r_u1 = list(range(N_u0)), list(range(N_u1))
r_u0_lwl = list(range(N_u0_lwl))

id_lwl_vec = list(itertools.product(r_u0_lwl, repeat=2))
id4_vec = list(itertools.product(r_u0, r_u1, repeat=2))
id2_vec = list(itertools.product(r_u1, r_u0, r_u0))
id_nw_vec = list(itertools.product(r_u0, itertools.repeat(0, 1), repeat=2))

z_cou_lwl = plasmon_det_zero_lwl(vu_vec, id_lwl_vec, N_u0_lwl, du0_lwl, 1e-8,
                                 sys)
z_sys_lwl = plasmon_det_zero_lwl(vu_vec, id_lwl_vec, N_u0_lwl, du0_lwl,
                                 sys.sys_ls, sys)

plt.axvline(x=z_sys_lwl, color='m')
plt.axvline(x=z_cou_lwl, color='g')
plt.axhline(y=0, color='y')

print('sys_ls: \t%8.6f nm' % (1 / sys.sys_ls))
print('###   lwl    ###')
print('z_cou:   \t%8.6f eV, z_sys:   \t%8.6f eV' % (z_cou_lwl, z_sys_lwl))

mu_e = 8 / sys.beta
mu_h = sys.get_mu_h_ht(mu_e)
"""
z_static = plasmon_det_zero_r(vuvu_nw_vec, id_nw_vec, du0, du1_nw, N_u0, 1,
                              mu_e, mu_h, sys)
"""

#plt.plot([z_static], [0], 'ro')

#print('z_static:\t%8.6f eV' % (z_static))

N_z_real = 1 << 6

z0_real, z1_real = -1e-1, -1e-6

z_arr = -linspace(z0_real, z1_real, N_z_real)

t0 = time.time()
det_cx = plasmon_det_ht(z_arr, N_u0, N_u1, mu_e, mu_h, sys)
print('[%e], Elapsed: %.2fs' % (mu_e, time.time() - t0))

det_cx_arr = array(det_cx)

plt.plot(-z_arr, real(det_cx), 'r-')
plt.plot(-z_arr, imag(det_cx), 'b-')

plt.show()

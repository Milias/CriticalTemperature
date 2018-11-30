from common import *

N_u0_lwl = 1 << 8

N_u0 = 1 << 9
#N_u1 = (1 << 2) + 1
N_u1 = (1 << 0)

N_x = 1 << 3

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300  # K
sys = system_data(m_e, m_h, eps_r, T)

u0_lwl, du0_lwl = linspace(
    1 / N_u0_lwl, 1 - 1 / N_u0_lwl, N_u0_lwl, retstep=True)

vu_vec = list(itertools.product(u0_lwl, repeat=2))
r_u0_lwl = list(range(N_u0_lwl))
id_lwl_vec = list(itertools.product(r_u0_lwl, repeat=2))

z_cou_lwl = plasmon_det_zero_lwl(
    vu_vec,
    id_lwl_vec,
    N_u0_lwl,
    du0_lwl,
    1e-8,
    sys,
)
z_sys_lwl = plasmon_det_zero_lwl(
    vu_vec,
    id_lwl_vec,
    N_u0_lwl,
    du0_lwl,
    sys.sys_ls,
    sys,
)

plt.axhline(y=z_sys_lwl, color='m')
plt.axhline(y=z_cou_lwl, color='g')
plt.axhline(y=0, color='k')
plt.axvline(x=1 / sys.beta, color='k', linestyle='--')

print('sys_ls: \t%8.6f nm' % (1 / sys.sys_ls))
print('###   lwl    ###')
print('z_cou:   \t%8.6f eV, z_sys:   \t%8.6f eV' % (z_cou_lwl, z_sys_lwl))
print('β^-1: %f eV' % (1 / sys.beta))

mu_e_vec = linspace(0.1, 10, N_x) / sys.beta
poles_list = []

for mu_e in mu_e_vec:
    mu_h = sys.get_mu_h_ht(mu_e)

    t0 = time.time()
    poles_list.append(plasmon_det_zero_ht(N_u0, N_u1, mu_e, mu_h, sys))
    print('[%e], Elapsed: %.2fs' % (mu_e, time.time() - t0))

plt.plot(mu_e_vec, poles_list, 'r-', label='T: %f K' % sys.T)

plt.legend(loc=0)
plt.show()

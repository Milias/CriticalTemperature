from common import *

N_u0 = 1 << 7
N_u1 = 1 << 5

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)
sys_ls = 0.5 / pi * sys.c_aEM / sys.eps_r * (sys.m_e + sys.m_h) / sys.c_hbarc

u_max = 1

u0, du0 = linspace(u_max / N_u0, 1 - u_max / N_u0, N_u0, retstep=True)
if N_u1 > 1:
    u1, du1 = linspace(-1 + u_max / N_u1, 1 - u_max / N_u1, N_u1, retstep=True)
else:
    u1, du1 = array([0.0]), float('nan')

vu_vec = list(itertools.product(u0, repeat=2))
vuvu_vec = list(itertools.product(u0, u1, repeat=2))

r_u0, r_u1 = list(range(N_u0)), list(range(N_u1))
id_lwl_vec = list(itertools.product(r_u0, repeat=2))
id_vec = list(itertools.product(r_u0, r_u1, repeat=2))

x_list = logspace(-8, 4, 16)

z_cou_lwl_wf = wf_2d_E_cou_py(sys)
z_sys_lwl_wf = wf_2d_E_lim_py(sys_ls, sys)

z_cou_lwl = plasmon_det_zero_vu_simp_lwl(vu_vec, id_lwl_vec, N_u0, du0, 1e-8,
                                     sys)
z_sys_lwl = plasmon_det_zero_vu_simp_lwl(vu_vec, id_lwl_vec, N_u0, du0, sys_ls,
                                     sys)

#z_cou_wf, z_sys_wf = plasmon_static_eB_v([1e-8, 1e3], sys)
z_cou_wf, z_sys_wf = float('nan'), float('nan')
z_cou = plasmon_det_zero_vu(vuvu_vec, id_vec, du0, du1, N_u0, N_u1, 1e-8,
                            sys.m_eh * 1e-8, sys, 1e-12)
z_sys = plasmon_det_zero_vu(vuvu_vec, id_vec, du0, du1, N_u0, N_u1, 1e3,
                            sys.m_eh * 1e3, sys, 1e-12)

print('sys_ls: \t%8.6f nm' % (1 / sys_ls))
print('###  static  ###')
print(
    'z_cou_wf:\t%8.6f eV, z_sys_wf:\t%8.6f eV\nz_cou:   \t%8.6f eV, z_sys:   \t%8.6f eV'
    % (z_cou_wf, z_sys_wf, z_cou, z_sys))
print('###   lwl    ###')
print(
    'z_cou_wf:\t%8.6f eV, z_sys_wf:\t%8.6f eV\nz_cou:   \t%8.6f eV, z_sys:   \t%8.6f eV'
    % (z_cou_lwl_wf, z_sys_lwl_wf, z_cou_lwl, z_sys_lwl))

plt.axhline(y=0, color='k')

plt.axhline(y=z_cou, color='g', label='(D) $\epsilon_{B,Cou}$')
plt.axhline(
    y=z_cou_wf, color='g', linestyle='--', label='(W) $\epsilon_{B,Cou}$')

plt.axhline(
    y=z_sys, color='m', label='(D) $\epsilon_{B,lwl}(\lambda_{s,0}^{-1})$')
plt.axhline(
    y=z_sys_wf,
    color='m',
    linestyle='--',
    label='(W) $\epsilon_{B,lwl}(\lambda_{s,0}^{-1})$')

z_list = []

t0 = time.time()
#z_list.extend(plasmon_static_eB_v(x_list, sys))
z_list.extend([0] * int(x_list.size))
print(z_list)
print('Elapsed: %.2fs' % (time.time() - t0))

for mu_e in x_list:
    mu_h = sys.m_eh * mu_e

    t0 = time.time()
    z = plasmon_det_zero_vu(vuvu_vec, id_vec, du0, du1, N_u0, N_u1, mu_e, mu_h,
                            sys, 1e-12)

    print('[%e] z: %f, Elapsed: %.2fs' % (mu_e, z, time.time() - t0))

    z_list.append(z)

z_arr = real(array(z_list)).reshape((2, x_list.size))

if z_arr.shape[0] > 1:
    plt.semilogx(x_list, z_arr[1, :], '.-', label='(D)')

plt.semilogx(x_list, z_arr[0, :], '.--', label='(W)')

plt.title('$\epsilon_B$ vs. $\mu_e$\nStatic approximation')
plt.xlabel('$\mu_e$ / eV')
plt.ylabel('$\epsilon_B$ / eV')

plt.legend(loc=0)

plt.savefig('plots/poles_static.eps')

plt.show()

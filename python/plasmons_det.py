from common import *

N_u0 = 1 << 7
N_u1 = 1 << 0

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

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

z_cou_lwl_wf = wf_2d_E_cou_py(sys)
z_sys_lwl_wf = wf_2d_E_lim_py(sys.sys_ls, sys)

z_cou_lwl = plasmon_det_zero_lwl(vu_vec, id_lwl_vec, N_u0, du0, 1e-8, sys)
z_sys_lwl = plasmon_det_zero_lwl(vu_vec, id_lwl_vec, N_u0, du0, sys.sys_ls,
                                 sys)

#z_cou_wf, z_sys_wf = plasmon_static_eB_v([ 1e-8, 1e3 ], sys)
z_cou_wf, z_sys_wf = float('nan'), float('nan')
z_cou = plasmon_det_zero_r(vuvu_vec, id_vec, du0, du1, N_u0, N_u1, 1e-8,
                           sys.m_eh * 1e-8, sys, 1e-12)
z_sys = plasmon_det_zero_r(vuvu_vec, id_vec, du0, du1, N_u0, N_u1, 1e3,
                           sys.m_eh * 1e3, sys, 1e-12)

print('sys_ls: \t%8.6f nm' % (1 / sys.sys_ls))
print('###  static  ###')
print(
    'z_cou_wf:\t%8.6f eV, z_sys_wf:\t%8.6f eV\nz_cou:   \t%8.6f eV, z_sys:   \t%8.6f eV'
    % (z_cou_wf, z_sys_wf, z_cou, z_sys))
print('###   lwl    ###')
print(
    'z_cou_wf:\t%8.6f eV, z_sys_wf:\t%8.6f eV\nz_cou:   \t%8.6f eV, z_sys:   \t%8.6f eV'
    % (z_cou_lwl_wf, z_sys_lwl_wf, z_cou_lwl, z_sys_lwl))

plt.axhline(y=0, color='k')
plt.axvline(x=-z_cou, color='g', label='Coulomb binding energy')
plt.axvline(
    x=-z_sys,
    color='m',
    linestyle='-',
    label='Binding energy at $\lambda_0^{-1}$')

#z_arr = linspace(-z_sys, -z_cou, 1 << 7)
z_arr = logspace(-4, 3, 1 << 7)
x_arr = logspace(-8, 3, 8)
"""
default_colors = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
]
"""

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, x_arr.size)
]

for c, mu_e in zip(colors, x_arr):
    mu_h = sys.m_eh * mu_e

    t0 = time.time()
    det = plasmon_det_v(z_arr, vuvu_vec, id_vec, du0, du1, N_u0, N_u1, mu_e,
                        mu_h, sys)

    print('[%e], Elapsed: %.2fs' % (mu_e, time.time() - t0))

    plt.plot(z_arr, real(det), '.-', color=c, label=r'$\mu_e$: %.2f' % (mu_e))

    plt.plot(z_arr, imag(det), '.--', color=c)

plt.title('det($\mathbb{1} - \hat V \hat G_0$) vs. $-z$\nStatic approximation')
plt.xlabel('$-z$ / eV')
plt.ylabel('det($\mathbb{1} - \hat V \hat G_0$) / dimensionless')

#plt.ylim(-0.009, 0.009)
plt.legend(loc=0)

plt.savefig('plots/det_static.eps')

plt.show()

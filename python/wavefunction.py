from common import *

N_u0 = 1 << 9
N_u1 = 1 << 0

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

E_vec = -linspace(0.08, -sys.get_E_n(0.5), 8)

z_cou_lwl_wf = wf_2d_E_cou_py(sys)
z_sys_lwl_wf = wf_2d_E_lim_py(sys_ls, sys)

z_cou_lwl = plasmon_det_zero_vu_simp_lwl(vu_vec, id_lwl_vec, N_u0, du0, 1e-8,
                                     sys)
z_sys_lwl = plasmon_det_zero_vu_simp_lwl(vu_vec, id_lwl_vec, N_u0, du0, sys_ls,
                                     sys)

#z_cou_wf, z_sys_wf = plasmon_static_eB_v([1e-8, 1e3], sys)
z_cou_wf, z_sys_wf = float('nan'), float('nan')
z_cou = plasmon_det_zero_vu(vuvu_vec, id_vec, du0, du1, N_u0, N_u1, 1e-8,
                            sys.m_eh * 1e-8, sys, 1e-2)
z_sys = plasmon_det_zero_vu(vuvu_vec, id_vec, du0, du1, N_u0, N_u1, 1e3,
                            sys.m_eh * 1e3, sys, 1e-2)

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

z_list = []

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, E_vec.size)
]

mu_e = 1e-2

for E in E_vec:
    mu_h = sys.m_eh * mu_e

    t0 = time.time()
    wf = wf_2d_static_py(E, mu_e, sys)
    #wf = wf_2d_s_py(E, mu_e, sys)

    print('[%e,%f] Elapsed: %.2fs' % (mu_e, E, time.time() - t0))

    z_list.append(wf)

E_d = plasmon_det_zero_vu(vuvu_vec, id_vec, du0, du1, N_u0, N_u1, mu_e, mu_h,
                          sys, 1e-2)
wf_d = wf_2d_static_py(E_d, mu_e, sys)
wf_d = array(wf_d).reshape((len(wf_d) // 3, 3))

plt.plot(wf_d[:, 2], wf_d[:, 0], 'm.-', label='E = %f' % E_d)
x_max = 0

for c, wf, E in zip(colors, z_list, E_vec):
    wf_arr = array(wf).reshape((len(wf) // 3, 3))
    plt.plot(wf_arr[:, 2], wf_arr[:, 0], '-', label='E = %f' % E, color=c)
    #plt.plot(wf_arr[:, 2], wf_arr[:, 1], '--', color=c)

    if wf_arr[-1, 2] > x_max:
        x_max = wf_arr[-1, 2]

x_vec = logspace(-3, log10(x_max), 1 << 10)
real_potcoef = array(plasmon_real_potcoef_k(x_vec, mu_e, mu_h, sys.v_1, sys))
real_potcoef_lwl = array([
    -sys_ls * sys.c_aEM / sys.eps_r * sys.c_hbarc * pot_limit_2d(x * sys_ls)
    for x in x_vec
])
real_potcoef_cou = -sys.c_aEM / sys.eps_r * sys.c_hbarc / x_vec

plt.plot(x_vec, real_potcoef, 'm-')
plt.plot(x_vec, real_potcoef_cou, 'r-')
plt.plot(x_vec, real_potcoef_lwl, 'b-')

plt.legend(loc=0)

plt.ylim(-1, 1)
plt.xlim(0, x_max)
plt.savefig('plots/wf_static.eps')

plt.show()

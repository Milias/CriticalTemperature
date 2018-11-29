from common import *

N_k = 1 << 9
N_ls = 12

N_u, N_v = N_k, N_k

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)
sys_ls = 0.5 / pi * sys.c_aEM / sys.eps_r * (sys.m_e + sys.m_h) / sys.c_hbarc

k0, k1, u_max = 2.0, 0, 1

k, dk = linspace(0, k0, N_k, retstep=True)
kk_vec = list(itertools.product(k, repeat=2))

u, du = linspace(0, u_max, N_u, retstep=True)
#vu_vec = list(itertools.product(itertools.repeat(1.0, 1), u))
vu_vec = list(itertools.product(u, repeat=2))

r_k = range(N_k)
id_vec = list(itertools.product(r_k, repeat=2))
#id_vu_vec = list(itertools.product(itertools.repeat(0, 1), r_k))
id_vu_vec = list(itertools.product(r_k, repeat=2))

z_sys_wf = wf_2d_E_lim_py(sys_ls, sys)
#z_sys = plasmon_sysmat_det_zero_lwl(kk_vec, id_vec, dk, N_k, sys_ls, sys.v_1, sys)

print('z_sys_wf: %f eV' % z_sys_wf)

#plt.axhline(y=0, color='k')

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, N_ls)
]

ls_vec = logspace(-4, 1, N_ls)
id_v = 1

for c, ls in zip(colors, ls_vec):
    #potcoef_lwl = array(plasmon_potcoef_lwl_v(kk_vec, ls, sys)).reshape((N_k, ))

    t0 = time.time()
    potcoef_lwl = array(
        plasmon_mat_vu_simp_lwl(vu_vec, id_vu_vec, N_v, N_u, du, ls, z_sys_wf,
                                sys)).reshape((N_v, N_u))
    print('Elapsed: %.2fs' % (time.time() - t0))

    plt.plot(
        u,
        potcoef_lwl[id_v, :],
        '-',
        color=c,
        label='$\lambda_s^{-1}: %f$' % ls)

#plt.xlim(k[0], k[-1])
#plt.xlim(u[0], u[-1])
plt.xlim(0, 0.1)
plt.ylim(None, 0)

#plt.legend(loc=0)
plt.tight_layout()
plt.show()

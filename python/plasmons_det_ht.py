from common import *

N_u0_lwl = 1 << 8

N_u0 = 1 << 10
#N_u1 = (1 << 2) + 1
N_u1 = (1 << 0)

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 10  # K
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

plt.axvline(x=z_sys_lwl, color='m')
plt.axvline(x=z_cou_lwl, color='g')
plt.axhline(y=0, color='y')

print('sys_ls: \t%8.6f nm' % (1 / sys.sys_ls))
print('###   lwl    ###')
print('z_cou:   \t%8.6f eV, z_sys:   \t%8.6f eV' % (z_cou_lwl, z_sys_lwl))

mu_vec = linspace(0.1, 8, 6) / sys.beta

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, mu_vec.size)
]

for c, mu_e in zip(colors, mu_vec):
    mu_h = sys.get_mu_h_ht(mu_e)

    N_z_real = 1 << 6

    z0_real, z1_real = z_cou_lwl, -1e-6

    z_arr = logspace(-6, log10(-z_cou_lwl), N_z_real)

    t0 = time.time()
    det_cx = plasmon_det_ht(z_arr, N_u0, N_u1, mu_e, mu_h, sys)
    print('[%e], Elapsed: %.2fs' % (mu_e, time.time() - t0))

    det_cx_arr = array(det_cx)

    plt.plot(-z_arr, det_cx, '-', color=c, label='%.3f' % mu_e)

plt.ylim(-1, 1)
plt.legend(loc=0)
plt.show()

from common import *

N_u0_lwl = 1 << 8

N_u0 = 1 << 8
eb_cou = 0.193

m_e, m_h, eps_r, T = 0.12, 0.3, 4.90185, 294  # K
sys = system_data(m_e, m_h, eps_r, T)

eps_r = sys.c_aEM * sqrt(2 * sys.m_p / eb_cou)
sys = system_data(m_e, m_h, eps_r, T)

z_cou_lwl = plasmon_det_zero_lwl(
    N_u0_lwl,
    1e-8,
    sys,
)
z_sys_lwl = plasmon_det_zero_lwl(
    N_u0_lwl,
    sys.sys_ls,
    sys,
)

plt.axvline(x=z_sys_lwl, color='m')
plt.axvline(x=z_cou_lwl, color='g')
plt.axhline(y=0, color='y')

print('sys_ls: \t%8.6f nm' % (1 / sys.sys_ls))
print('###   lwl    ###')
print('z_cou:   \t%8.6f eV, z_sys:   \t%8.6f eV' % (z_cou_lwl, z_sys_lwl))

mu_e = 4 / sys.beta
mu_h = sys.get_mu_h_ht(mu_e)

N_z_real, N_z_imag = 1 << 3, (1 << 3) + 1

z0_real, z1_real = 2 * z_cou_lwl, 0  #z_sys_lwl * (1 + 5e-1), z_sys_lwl * (1 - 5e-1)
z0_imag, z1_imag = -0.2, 0.2

z_real_vec = -linspace(z0_real, z1_real, N_z_real)
z_imag_vec = -linspace(z0_imag, z1_imag, N_z_imag)

z_real_arr, z_imag_arr = meshgrid(z_real_vec, z_imag_vec)

z_arr = (z_real_arr + 1j * z_imag_arr).flatten()

t0 = time.time()
det_cx = plasmon_det_ht_cx(z_arr, N_u0, N_u1, mu_e, mu_h, sys)

print('[%e], Elapsed: %.2fs' % (mu_e, time.time() - t0))

det_cx_arr = array(det_cx).reshape(z_real_arr.shape).T
print(det_cx_arr)

det_r, det_ph = abs(det_cx_arr), angle(det_cx_arr)

det_h = 0.5 + 0.5 * det_ph / pi
det_s = 0.9 * ones_like(det_r)
det_v = det_r / (1.0 + det_r)

det_hsv = array([det_h, det_s, det_v]).T

det_colors = matplotlib.colors.hsv_to_rgb(det_hsv)

plt.imshow(
    det_colors,
    aspect='auto',
    extent=(z0_real, z1_real, z0_imag, z1_imag),
)

plt.contour(-z_real_arr.T, -z_imag_arr.T, det_v, 16, cmap=cm.cool)

plt.savefig('plots/det_cx_2d.eps')

plt.show()

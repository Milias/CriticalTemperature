from common import *

N_k_lwl = 1 << 8

N_k = 1 << 8
#N_w = (1 << 2) + 1
N_w = (1 << 0)

#m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300  # K
m_e, m_h, eps_r, T = 0.12, 0.3, 4.90185, 294  # K
sys = system_data(m_e, m_h, eps_r, T)

eps_r = sys.c_aEM * sqrt(2 * sys.m_p / 0.194)
sys = system_data(m_e, m_h, eps_r, T)

z_cou_lwl = plasmon_det_zero_lwl(
    N_k_lwl,
    1e-8,
    sys,
)

z_sys_lwl = plasmon_det_zero_lwl(
    N_k_lwl,
    sys.sys_ls,
    sys,
)

plt.axhline(y=z_sys_lwl, color='m')
plt.axhline(y=z_cou_lwl, color='g')
plt.axhline(y=sys.get_E_n(1.5), color='g', linestyle='--')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k', linestyle=':')

print('sys_ls: \t%8.6f nm' % (1 / sys.sys_ls))
print('###   lwl    ###')
print('z_cou:   \t%8.6f eV, z_sys:   \t%8.6f eV' % (z_cou_lwl, z_sys_lwl))
print('z_cou_analytic: %8.6f eV' % (sys.get_E_n(0.5)))
print('β^-1: %f eV' % (1 / sys.beta))

T_vec = linspace(294, 314, 1)
N_x = 48

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, T_vec.size)
]

for c, T in zip(colors, T_vec):
    sys = system_data(m_e, m_h, eps_r, T)

    mu_max = min(0.3, 18 / sys.beta)
    mu_vec = linspace(-mu_max, mu_max, N_x)

    poles_list = array(time_func(plasmon_det_zero_ht_v, N_k, mu_vec, sys))
    poles_list_1 = array(time_func(plasmon_det_zero_ht_v1, N_k, mu_vec, sys))

    plt.plot(mu_vec, poles_list, '.-', label='T: %.0f K' % sys.T, color=c)
    plt.plot(mu_vec, poles_list_1, '.--', label='T: %.0f K' % sys.T, color=c)

plt.axvline(x=0, color='k')
plt.legend(loc=0)

plt.title('Binding energy vs. chemical potential\nMaxwell-Boltzmann -- Static')
plt.xlabel('$\mu_e$ / eV')
plt.ylabel('$\epsilon_B$ / eV')

plt.show()

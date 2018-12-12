from common import *

N_u0_lwl = 1 << 10

N_u0 = 1 << 10
#N_u1 = (1 << 2) + 1
N_u1 = (1 << 0)

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300  # K
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

plt.axhline(y=z_sys_lwl, color='m')
plt.axhline(y=z_cou_lwl, color='g')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k', linestyle=':')

print('sys_ls: \t%8.6f nm' % (1 / sys.sys_ls))
print('###   lwl    ###')
print('z_cou:   \t%8.6f eV, z_sys:   \t%8.6f eV' % (z_cou_lwl, z_sys_lwl))
print('z_cou_analytic: %8.6f eV' % (sys.get_E_n(0.5)))
print('β^-1: %f eV' % (1 / sys.beta))

T_vec = linspace(10, 300, 5)
N_x = 1

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, T_vec.size)
]

for c, T in zip(colors, T_vec):
    sys = system_data(m_e, m_h, eps_r, T)

    mu_vec = linspace(-8, 8, N_x) / sys.beta
    poles_list = zeros_like(mu_vec)

    for i, mu_e in enumerate(mu_vec):
        mu_h = sys.get_mu_h(mu_e)

        t0 = time.time()
        poles_list[i] = plasmon_det_zero_ht(N_u0, mu_e, mu_h, sys)
        print('[%.0f, %e] %f Elapsed: %.2fs' % (T, mu_e, poles_list[i],
                                                time.time() - t0))

        if poles_list[i] == 0:
            poles_list[i:] = 0
            break

    plt.plot(mu_vec, poles_list, '.-', label='T: %.0f K' % sys.T, color=c)

plt.axvline(x=0, color='k')
plt.legend(loc=0)

plt.title('Binding energy vs. chemical potential\nMaxwell-Boltzmann')
plt.xlabel('$\mu_e$ \ eV')
plt.ylabel('$\epsilon_B$ \ eV')

plt.show()

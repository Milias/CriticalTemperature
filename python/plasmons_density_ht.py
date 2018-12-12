from common import *

N_k = 1 << 8

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 10  # K
T_vec = linspace(10, 310, 4)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, T_vec.size)
]

for c, T in zip(colors, T_vec):
    sys = system_data(m_e, m_h, eps_r, T)

    u_vec = linspace(-30, -10, 1 << 6)

    n_total = 1
    plt.axvline(x=sys.mu_exc_u(n_total), color=c, linestyle='--')
    plt.axhline(y=n_total, color=c, linestyle='--')

    n_total_vec = array(time_func(plasmon_density_mu_ht_v, u_vec, N_k, sys))

    plt.plot(
        u_vec,
        n_total_vec,
        '.-',
        color=c,
        label='$T: %.0f$' % sys.T,
    )

#plt.axvline(x = 0, color = 'k')
#plt.axhline(y = 0, color = 'k')

plt.title(
    'Carrier density vs. chemical potential\nMaxwell-Boltzmann -- Static')
plt.xlabel('$\mu_e$ / eV')
plt.ylabel(r'$n_\alpha / n_{total}$')

plt.ylim(0, None)
plt.legend(loc=0)

plt.show()

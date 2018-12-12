from common import *

N_k = 1 << 8

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 10  # K
T_vec = linspace(100, 300, 3)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, T_vec.size)
]

for c, T in zip(colors, T_vec):
    sys = system_data(m_e, m_h, eps_r, T)

    n_vec = logspace(-1, 1, 1 << 4)

    u_vec = array(time_func(plasmon_density_ht_v, n_vec, N_k, sys))

    plt.plot(
        n_vec,
        u_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

#plt.axvline(x = 0, color = 'k')
#plt.axhline(y = 0, color = 'k')

plt.title(
    'Carrier density vs. chemical potential\nMaxwell-Boltzmann -- Static')
plt.xlabel(r'$n_{total}$')
#plt.ylabel('$\mu_e$ / eV')
plt.ylabel('$u$ / dimensionless')

#plt.ylim(0, None)
plt.legend(loc=0)

plt.show()

from common import *

N_k = 1 << 8

#m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 10  # K
m_e, m_h, eps_r, T = 0.12, 0.3, 49.0185, 294  # K
T_vec = linspace(10, 410, 5)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, T_vec.size)
]

for c, T in zip(colors, T_vec):
    sys = system_data(m_e, m_h, eps_r, T)
    eb = -0.1

    u_vec = linspace(-20, 5, 1 << 7)

    n_vec = array([sys.density_exc_exp(u, eb) for u in u_vec])

    plt.plot(
        u_vec,
        n_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

#plt.axvline(x = 0, color = 'k')
#plt.axhline(y = 0, color = 'k')

plt.title(
    'Exciton density vs. $u$\nMaxwell-Boltzmann -- Static')
plt.ylabel(r'$n_{exc}$')
#plt.ylabel('$\mu_e$ / eV')
plt.xlabel('$u$ / dimensionless')

#plt.ylim(0, None)
plt.legend(loc=0)

plt.show()

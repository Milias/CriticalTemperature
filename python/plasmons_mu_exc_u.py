from common import *

N_k = 1 << 8

m_e, m_h, eps_r, T = 0.12, 0.3, 4.90185, 294  # K
sys = system_data(m_e, m_h, eps_r, T)

eps_r = sys.c_aEM * sqrt(2 * sys.m_p / 0.194)
sys = system_data(m_e, m_h, eps_r, T)

T_vec = linspace(10, 410, 5)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, T_vec.size)
]

for c, T in zip(colors, T_vec):
    sys = system_data(m_e, m_h, eps_r, T)

    n_vec = logspace(-10, 1, 1 << 7)

    mu_vec = array([sys.mu_exc_u(u) for u in n_vec])

    plt.plot(
        n_vec,
        mu_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

plt.legend(loc=0)

plt.show()

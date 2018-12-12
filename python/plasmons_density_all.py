from common import *

N_k = 1 << 10

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 30  # K
T_vec = linspace(50, 300, 12)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, T_vec.size)
]

mu_e_vec = linspace(-3e-1, 3e-1, (1 << 10) + 1)

for c, T in zip(colors, T_vec):
    sys = system_data(m_e, m_h, eps_r, T)
    eb = sys.get_E_n(0.5)

    n_t0_vec = array([sys.density_ideal_t0(mu_e) for mu_e in mu_e_vec])
    n_ht_vec = array([sys.density_ideal_ht(mu_e) for mu_e in mu_e_vec])
    n_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_ht_vec = array([
        sys.density_exc_ht(mu_e + sys.get_mu_h(mu_e), eb) for mu_e in mu_e_vec
    ])
    n_exc_vec = array(
        [sys.density_exc(mu_e + sys.get_mu_h(mu_e), eb) for mu_e in mu_e_vec])

    plt.plot(mu_e_vec, n_t0_vec, ':', color=c)
    plt.plot(mu_e_vec, n_ht_vec, '--', color=c)
    plt.plot(mu_e_vec, n_vec, '-', color=c)
    plt.plot(mu_e_vec, n_exc_ht_vec, '.--', color=c)
    plt.plot(mu_e_vec, n_exc_vec, '.-', color=c)

plt.axvline(x=0, color='k')
plt.axhline(y=0, color='k')
plt.axis([mu_e_vec[0], mu_e_vec[-1], 0, 0.2])
plt.show()

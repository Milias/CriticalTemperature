from common import *

N_k = 1 << 8

m_e, m_h, eps_r, T = 0.12, 0.3, 4.90185, 294  # K
sys = system_data(m_e, m_h, eps_r, T)

eps_r = sys.c_aEM * sqrt(2 * sys.m_p / 0.194)
sys = system_data(m_e, m_h, eps_r, T)

T_vec = linspace(294, 310, 1)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, T_vec.size)
]

n_vec = logspace(-5, 1, 1 << 7)

n_x, n_y = 2, 1
ax = [plt.subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]

data = loadtxt('../data2.txt', delimiter=',').reshape((1, n_vec.size + 2))
exp_data = loadtxt('bin/quantum_yield_charges_versus_N.csv', delimiter=',')

surf_area = 326.4  # nm^2
plot_exp_data = True

for c, (i, T) in zip(colors, enumerate(T_vec)):
    L, mob_R, mob_I, pol, freq = 1e-3, 77e-2, 23e-2, 4.6e-36, 2e12 * pi
    sys = system_data(m_e, m_h, eps_r, T)

    if data is None:
        exc_list = time_func(plasmon_density_ht_v, n_vec, N_k, sys)
    else:
        exc_list = data[i]

    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    if plot_exp_data:
        exp_n_vec = exp_data[:, 0] / surf_area
        exp_n_id_vec = exp_data[:, 1] * exp_n_vec
        exp_n_exc_vec = (1 - exp_data[:, 1]) * exp_n_vec

    cond_vec = array(
        time_func(plasmon_cond_v, n_id_vec * 1e18 * surf_area,
                  n_exc_vec * 1e18 * surf_area, L, mob_R, mob_I, pol, freq,
                  sys))

    exp_cond_vec = array(
        time_func(plasmon_cond_v, exp_n_id_vec * 1e18 * surf_area,
                  exp_n_exc_vec * 1e18 * surf_area, L, mob_R, mob_I, pol, freq,
                  sys))

    ax[0].set_title(
        'Particle number vs. photoexcitation number\nMaxwell-Boltzmann -- Static'
    )
    ax[0].set_xlabel(r'$\langle N_\gamma \rangle$')
    ax[0].set_ylabel(r'$\langle N_\alpha \rangle$')

    ax[0].loglog(
        n_vec * surf_area,
        n_id_vec * surf_area,
        '.-',
        color=c,
        label='T: $%.0f$ K, $n_e$' % sys.T,
    )

    ax[0].loglog(
        n_vec * surf_area,
        n_exc_vec * surf_area,
        '.--',
        color=c,
        label='T: $%.0f$ K, $n_{exc}$' % sys.T,
    )

    if plot_exp_data:
        ax[0].loglog(
            exp_n_vec * surf_area,
            exp_n_id_vec * surf_area,
            '^',
            color='k',
            label='Experiment, $e$',
        )

        ax[0].loglog(
            exp_n_vec * surf_area,
            exp_n_exc_vec * surf_area,
            'x',
            color='k',
            label='Experiment, $exc$',
        )

    ax[0].legend(loc=0)

    ax[1].set_title(
        'Conductivity vs. photoexcitation number\nMaxwell-Boltzmann -- Static')
    ax[1].set_xlabel(r'$\langle N_\gamma \rangle$')
    ax[1].set_ylabel(r'$\Delta\sigma$ / S m$^{-1}$')

    ax[1].semilogx(
        n_vec * surf_area,
        real(cond_vec),
        '.-',
        color=c,
        label='T: $%.0f$ K, real part' % sys.T,
    )

    ax[1].semilogx(
        n_vec * surf_area,
        -imag(cond_vec),
        '.--',
        color=c,
        label='T: $%.0f$ K, imag part' % sys.T,
    )

    if plot_exp_data:
        ax[1].semilogx(
            exp_n_vec * surf_area,
            real(exp_cond_vec),
            '^',
            color='k',
            label='Experiment, real part',
        )

        ax[1].semilogx(
            exp_n_vec * surf_area,
            -imag(exp_cond_vec),
            'x',
            color='k',
            label='Experiment, imag part',
        )

    ax[1].axhline(y=0, color='k')
    ax[1].legend(loc=0)

    plot_exp_data = False

plt.tight_layout()
plt.show()

from common import *

N_k = 1 << 8

m_e, m_h, eps_r, T = 0.12, 0.3, 4.90185, 294  # K
sys = system_data(m_e, m_h, eps_r, T)

eps_r = sys.c_aEM * sqrt(2 * sys.m_p / 0.194)
sys = system_data(m_e, m_h, eps_r, T)

surf_area = 326.4  # nm^2

T_vec = linspace(294, 310, 1)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, T_vec.size)
]

n_vec = linspace(0.5 / surf_area, 144 / surf_area, (1 << 7) + (1 << 6))

ax = [plt.subplot(2, 2, i + 1) for i in range(4)]

data = array([loadtxt('../data_exp_comp.txt', delimiter=',')])
exp_data = loadtxt('bin/quantum_yield_charges_versus_N.csv', delimiter=',')

plot_exp_data = True

for c, (i, T) in zip(colors, enumerate(T_vec)):
    sys = system_data(m_e, m_h, eps_r, T)

    if data is None:
        exc_list = time_func(plasmon_density_ht_v, n_vec, N_k, sys)
    else:
        exc_list = data[i]

    print(exc_list)

    if plot_exp_data:
        exp_n_vec = exp_data[:, 0] / surf_area
        exp_n_id_vec = exp_data[:, 1] * exp_n_vec
        exp_n_exc_vec = (1 - exp_data[:, 1]) * exp_n_vec

    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    #poles_list_1 = array(time_func(plasmon_det_zero_ht_v1, N_k, mu_e_vec, sys))

    ax[0].set_title(
        'Densities vs. photoexcitation density\nMaxwell-Boltzmann -- Static'
    )
    ax[0].set_xlabel(r'$n$ / nm$^{-2}$')
    ax[0].set_ylabel(r'$n_{id}$ / nm$^{-2}$')

    ax[0].semilogy(
        n_vec,
        n_id_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K, $e$' % sys.T,
    )

    ax[0].semilogy(
        n_vec,
        n_exc_vec,
        '.--',
        color=c,
        label='T: $%.0f$ K, $exc$' % sys.T,
    )

    if plot_exp_data:
        ax[0].semilogy(
            exp_n_vec,
            exp_n_id_vec,
            '^',
            color='k',
            label='Experiment, $e$',
        )

        ax[0].semilogy(
            exp_n_vec,
            exp_n_exc_vec,
            'x',
            color='k',
            label='Experiment, $exc$',
        )

    ax[0].legend(loc=0)

    ax[1].set_title(
        '$y$ vs. photoexcitation density\nMaxwell-Boltzmann -- Static')
    ax[1].set_xlabel(r'$n$ / nm$^{-2}$')
    ax[1].set_ylabel(r'$y$ / dimensionless')

    ax[1].plot(
        n_vec,
        n_id_vec / n_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    if plot_exp_data:
        ax[1].plot(
            exp_n_vec,
            exp_n_id_vec / exp_n_vec,
            'x',
            color='k',
            label='Experiment',
        )

    ax[1].legend(loc=0)

    ax[2].set_title(
        r'$\mu_\alpha$' +
        ' vs. photoexcitation density\nMaxwell-Boltzmann -- Static')
    ax[2].set_xlabel(r'$n$ / nm$^{-2}$')
    ax[2].set_ylabel(r'$\mu_\alpha$ / eV')

    ax[2].plot(
        n_vec,
        mu_e_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    ax[2].plot(
        n_vec,
        mu_h_vec,
        '.--',
        color=c,
    )

    ax[2].axhline(y=mu_e_lim, color=c, linestyle='-')
    ax[2].axhline(y=sys.get_mu_h(mu_e_lim), color=c, linestyle='--')

    ax[2].legend(loc=0)

    eb_vec = array(
        time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys, eb_lim))

    ax[3].set_title(
        'Binding energy and $\mu_{exc}$ vs. photoexcitation density\nMaxwell-Boltzmann -- Static'
    )
    ax[3].set_xlabel(r'$n$ / nm$^{-2}$')
    ax[3].set_ylabel(r'$\epsilon_B$ and $\mu_{exc}$ / eV')

    """
    ax[3].plot(
        n_vec, poles_list_1, '.--', label='T: $%.0f$ K' % sys.T, color=c)
    """

    ax[3].plot(
        n_vec,
        eb_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    ax[3].plot(
        n_vec,
        mu_e_vec + mu_h_vec,
        '.--',
        color=c,
        label='$\mu_{exc}$',
    )

    ax[3].axhline(y=eb_lim, color=c, linestyle='-')

    ax[3].legend(loc=0)

    plot_exp_data = False

#plt.axvline(x = 0, color = 'k')
#plt.axhline(y = 0, color = 'k')

#plt.ylim(0, None)

plt.show()

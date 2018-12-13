from common import *

N_k = 1 << 9

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 10  # K
T_vec = linspace(10, 310, 4)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, T_vec.size)
]

n_exc_vec = logspace(-8, 2, 1 << 6)

ax = [plt.subplot(2, 2, i + 1) for i in range(4)]

#data = loadtxt('../data.txt', delimiter=',')

for c, (i, T) in zip(colors, enumerate(T_vec)):
    sys = system_data(m_e, m_h, eps_r, T)

    exc_list = time_func(plasmon_density_exc_ht_v, n_exc_vec, N_k, sys)
    #exc_list = data[i]

    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])

    ax[0].set_title(
        'Electron density vs. exciton density\nMaxwell-Boltzmann -- Static')
    ax[0].set_xlabel(r'$n_{exc}$ / nm$^{-2}$')
    ax[0].set_ylabel(r'$n_{id}$ / nm$^{-2}$')

    ax[0].loglog(
        n_exc_vec,
        n_id_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    ax[0].legend(loc=0)

    ax[1].set_title('$y$ vs. exciton density\nMaxwell-Boltzmann -- Static')
    ax[1].set_xlabel(r'$n_{exc}$ / nm$^{-2}$')
    ax[1].set_ylabel(r'$y$ / dimensionless')

    ax[1].semilogx(
        n_exc_vec,
        n_id_vec / (n_id_vec + n_exc_vec),
        '.-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    ax[1].legend(loc=0)

    ax[2].set_title(r'$\mu_\alpha$' +
                    ' vs. exciton density\nMaxwell-Boltzmann -- Static')
    ax[2].set_xlabel(r'$n_{exc}$ / nm$^{-2}$')
    ax[2].set_ylabel(r'$\mu_\alpha$ / eV')

    ax[2].semilogx(
        n_exc_vec,
        mu_e_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    ax[2].semilogx(
        n_exc_vec,
        mu_h_vec,
        '.--',
        color=c,
    )

    ax[2].axhline(y=mu_e_lim, color=c, linestyle='-')
    ax[2].axhline(y=sys.get_mu_h(mu_e_lim), color=c, linestyle='--')

    ax[2].legend(loc=0)

    eb_vec = array([
        time_func(plasmon_det_zero_ht, N_k, mu_e, mu_h, sys, eb_lim)
        for mu_e, mu_h in zip(mu_e_vec, mu_h_vec)
    ])

    ax[3].set_title(
        'Binding energy vs. exciton density\nMaxwell-Boltzmann -- Static')
    ax[3].set_xlabel(r'$n_{exc}$ / nm$^{-2}$')
    ax[3].set_ylabel(r'$\epsilon_B$ / eV')

    ax[3].semilogx(
        n_exc_vec,
        eb_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    ax[3].axhline(y=eb_lim, color=c, linestyle='-')

    ax[3].legend(loc=0)

#plt.axvline(x = 0, color = 'k')
#plt.axhline(y = 0, color = 'k')

#plt.ylim(0, None)

plt.show()

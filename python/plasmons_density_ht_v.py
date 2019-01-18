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

ax = [plt.subplot(2, 2, i + 1) for i in range(4)]

data = None #loadtxt('../data.txt', delimiter=',')

for c, (i, T) in zip(colors, enumerate(T_vec)):
    sys = system_data(m_e, m_h, eps_r, T)

    if data is None:
        exc_list = time_func(plasmon_density_ht_v, n_vec, N_k, sys)
    else:
        exc_list = data[i]

    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    print(exc_list)

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    #poles_list_1 = array(time_func(plasmon_det_zero_ht_v1, N_k, mu_e_vec, sys))

    ax[0].set_title(
        'Densities vs. photoexcitation density\nMaxwell-Boltzmann -- Static')
    ax[0].set_xlabel(r'$n$ / nm$^{-2}$')
    ax[0].set_ylabel(r'$n_{\alpha}$ / nm$^{-2}$')

    ax[0].loglog(
        n_vec,
        n_id_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K, $n_e$' % sys.T,
    )

    ax[0].loglog(
        n_vec,
        n_exc_vec,
        '.--',
        color=c,
        label='T: $%.0f$ K, $n_{exc}$' % sys.T,
    )

    ax[0].legend(loc=0)

    ax[1].set_title(
        '$y$ vs. photoexcitation density\nMaxwell-Boltzmann -- Static')
    ax[1].set_xlabel(r'$n$ / nm$^{-2}$')
    ax[1].set_ylabel(r'$y$ / dimensionless')

    ax[1].semilogx(
        n_vec,
        n_id_vec / n_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    ax[1].legend(loc=0)

    ax[2].set_title(
        r'$\mu_\alpha$' +
        ' vs. photoexcitation density\nMaxwell-Boltzmann -- Static')
    ax[2].set_xlabel(r'$n$ / nm$^{-2}$')
    ax[2].set_ylabel(r'$\mu_\alpha$ / eV')

    ax[2].semilogx(
        n_vec,
        mu_e_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    ax[2].semilogx(
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
    ax[3].set_ylabel(r'$\epsilon_B$ / eV')
    """
    ax[3].semilogx(
        n_vec, poles_list_1, '.--', label='T: $%.0f$ K' % sys.T, color=c)
    """

    ax[3].semilogx(
        n_vec,
        eb_vec,
        '.-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    ax[3].semilogx(
        n_vec,
        mu_e_vec + mu_h_vec,
        '.--',
        color=c,
    )

    ax[3].axhline(y=eb_lim, color=c, linestyle='-')

    ax[3].legend(loc=0)

#plt.axvline(x = 0, color = 'k')
#plt.axhline(y = 0, color = 'k')

#plt.ylim(0, None)

plt.show()

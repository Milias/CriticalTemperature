from common import *

import statsmodels.api as sm

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
const_data = loadtxt(
    '../data_const.txt', delimiter=',').reshape((1, n_vec.size + 2))
p_data = loadtxt('../data_points.txt', delimiter=',').reshape((8, ))
exp_data = loadtxt('bin/quantum_yield_charges_versus_N.csv', delimiter=',')
exp_points = loadtxt('bin/cdse_platelet_data.csv', delimiter=',')
exp_fit = loadtxt('bin/cdse_platelet_fit_data_update.csv')

surf_area = 326.4  # nm^2
plot_exp_data = True

plot_func = {'log': ('loglog', 'semilogx'), 'linear': ('semilogy', 'plot')}

plot_type = 'linear'

for c, (i, T) in zip(colors, enumerate(T_vec)):
    L, mob_R, mob_I, pol, freq = 2e-3, 54e-4, 7e-4, 3.1e-36, 0.6e12
    p_Na_vec = exp_points[:, 3] * 1e4
    Na_vec = n_vec * p_Na_vec[0] / (exp_points[0, 0] / surf_area)
    sys = system_data(m_e, m_h, eps_r, T)

    if data is None:
        n_vec = exp_points[:, 0] / surf_area
        exc_list = time_func(plasmon_density_ht_c_v, n_vec, N_k, sys)
        print(exc_list)
        exit()
    else:
        exc_list = data[i]

    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    q_yield_vec = n_id_vec / n_vec
    q_yield_exc_vec = n_exc_vec / n_vec

    p_mu_e_lim, p_eb_lim = p_data[:2]
    p_mu_e_vec = array(p_data[2:])

    p_n_vec = exp_points[:, 0] / surf_area
    p_mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in p_mu_e_vec])
    p_n_id_vec = array([sys.density_ideal(mu_e) for mu_e in p_mu_e_vec])
    p_n_exc_vec = p_n_vec - p_n_id_vec

    p_q_yield_vec = p_n_id_vec / p_n_vec
    p_q_yield_exc_vec = p_n_exc_vec / p_n_vec

    if plot_exp_data:
        exp_n_vec = exp_data[:, 0] / surf_area
        exp_n_id_vec = exp_data[:, 1] * exp_n_vec
        exp_n_exc_vec = (1 - exp_data[:, 1]) * exp_n_vec

    real_x = (sys.c_e_charge / L * p_Na_vec * p_q_yield_vec).reshape(
        p_data.size - 2, 1)
    real_y = exp_points[:, 1]

    real_model = sm.WLS(
        real_y, real_x, weights=1 / exp_points[:, 4]**2, has_const=False)
    real_fit = real_model.fit(use_t=True)

    imag_x0 = sys.c_e_charge * p_Na_vec / L * p_q_yield_vec
    imag_x1 = p_Na_vec / L * freq * 2 * pi * p_q_yield_exc_vec
    imag_x = array([imag_x0, imag_x1]).T
    imag_y = exp_points[:, 2]

    imag_model = sm.WLS(
        imag_y, imag_x, weights=1 / exp_points[:, 5]**2, has_const=False)
    imag_fit = imag_model.fit(use_t=True)

    fit_mob_R, = real_fit.params
    fit_mob_I, fit_pol = imag_fit.params

    mob_R_err, mob_I_err, pol_err = real_fit.bse, *imag_fit.bse

    print((real_x.flatten() * mob_R, imag_x0 * mob_I + imag_x1 * pol))
    print((real_y, imag_y))
    print('mob_R: %f±%1.0e, mob_I: %e±%1.0e, pol: %e±%1.0e' %
          (fit_mob_R * 1e4, mob_R_err * 1e4, fit_mob_I * 1e4, mob_I_err * 1e4,
           fit_pol, pol_err))

    cond_vec = array(
        time_func(plasmon_cond_v, q_yield_vec, Na_vec, L, fit_mob_R, fit_mob_I,
                  fit_pol, freq, sys))

    ax[0].set_title(
        'Densities vs. photoexcitation density\nMaxwell-Boltzmann -- Static')
    ax[0].set_xlabel(r'$n_\gamma$ / nm$^{-2}$')
    ax[0].set_ylabel(r'$n_{\alpha}$ / nm$^{-2}$')

    getattr(ax[0], plot_func[plot_type][0])(
        n_vec,
        n_id_vec,
        '-',
        color=c,
        label='T: $%.0f$ K, $n_e$' % sys.T,
    )

    getattr(ax[0], plot_func[plot_type][0])(
        n_vec,
        n_exc_vec,
        '--',
        color=c,
        label='T: $%.0f$ K, $n_{exc}$' % sys.T,
    )

    if plot_exp_data and False:
        getattr(ax[0], plot_func[plot_type][0])(
            exp_n_vec,
            exp_n_id_vec,
            'o',
            color='k',
            label='Experiment, $e$',
        )

        getattr(ax[0], plot_func[plot_type][0])(
            exp_n_vec,
            exp_n_exc_vec,
            '^',
            color='k',
            label='Experiment, $exc$',
        )

    if plot_type == 'log':
        ax[0].set_ylim(0.1 / surf_area, 100 / surf_area)
        ax[0].set_xlim(3 / surf_area, 80 / surf_area)
    else:
        ax[0].set_ylim(0.1 / surf_area, 100 / surf_area)
        ax[0].set_xlim(0, 60 / surf_area)
    ax[0].legend(loc=0)

    ax[1].set_title(
        'Conductivity vs. photoexcitation density\nMaxwell-Boltzmann -- Static'
    )
    ax[1].set_xlabel(r'$n_\gamma$ / nm$^{-2}$')
    ax[1].set_ylabel(r'$\Delta\sigma$ / S m$^{-1}$')

    if plot_exp_data:
        getattr(ax[1], plot_func[plot_type][1])(
            exp_points[:, 0] / surf_area,
            exp_points[:, 1],
            'o',
            color='k',
            label='T: $%.0f$ K, real part' % sys.T,
        )

        getattr(ax[1], plot_func[plot_type][1])(
            exp_points[:, 0] / surf_area,
            -exp_points[:, 2],
            '^',
            color='k',
            label='T: $%.0f$ K, imag part' % sys.T,
        )

        getattr(ax[1], plot_func[plot_type][1])(
            exp_fit[:, 0] / surf_area,
            exp_fit[:, 1],
            '-',
            color='k',
            label='Saha fit, real',
        )

        getattr(ax[1], plot_func[plot_type][1])(
            exp_fit[:, 0] / surf_area,
            -exp_fit[:, 2],
            '--',
            color='k',
            label='Saha fit, imag',
        )

        ax[1].errorbar(
            exp_points[:, 0] / surf_area,
            exp_points[:, 1],
            yerr=exp_points[:, 4],
            fmt='none',
            capsize=5,
            color='k')

        ax[1].errorbar(
            exp_points[:, 0] / surf_area,
            -exp_points[:, 2],
            yerr=exp_points[:, 5],
            fmt='none',
            capsize=5,
            color='k')

    getattr(ax[1], plot_func[plot_type][1])(
        n_vec,
        real(cond_vec),
        '-',
        color=c,
        label='T: $%.0f$ K, real part' % sys.T,
    )

    getattr(ax[1], plot_func[plot_type][1])(
        n_vec,
        -imag(cond_vec),
        '--',
        color=c,
        label='T: $%.0f$ K, imag part' % sys.T,
    )

    ax[1].set_ylim(-0.0125, 0.004)
    if plot_type == 'log':
        ax[1].set_xlim(3 / surf_area, 80 / surf_area)
    else:
        ax[1].set_xlim(0, 60 / surf_area)

    ax[1].axhline(y=0, color='k')
    ax[1].legend(loc=0)

    plot_exp_data = False

plt.tight_layout()
plt.show()

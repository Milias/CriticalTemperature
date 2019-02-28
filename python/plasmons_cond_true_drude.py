from common import *

import statsmodels.api as sm

N_k = 1 << 8

eb_cou = 0.193
err_eb_cou = 0.005

m_e, m_h, eps_r, T = 0.12, 0.3, 4.90185, 294  # K
sys = system_data(m_e, m_h, eps_r, T)

eps_r = sys.c_aEM * sqrt(2 * sys.m_p / eb_cou)
sys = system_data(m_e, m_h, eps_r, T)

T_vec = linspace(294, 310, 1)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, T_vec.size)
]

n_vec = logspace(-5, 1, 1 << 7)

n_x, n_y = 2, 1
fig = plt.figure(figsize=(5.8, 8.3), dpi=150)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]

eb_type = 'Constant Binding Energy'
eb_type = 'Maxwell-Boltzmann -- Static'

data = loadtxt('extra/data_mu_e.csv', delimiter=',').reshape((1, -1))
p_data = loadtxt('../data_points.txt', delimiter=' ').reshape((-1, ))

exp_data = loadtxt('bin/quantum_yield_charges_versus_N.csv', delimiter=',')
exp_points = loadtxt('bin/cdse_platelet_data.csv', delimiter=',')
exp_fit = loadtxt('bin/cdse_platelet_fit_data_update.csv')

p_eb_vec = loadtxt('../data_eb_points.txt', delimiter=' ')

surf_area = 326.4  # nm^2
plot_exp_data = True

plot_func = {'log': ('loglog', 'semilogx'), 'linear': ('semilogy', 'plot')}

plot_type = 'linear'

for c, (i, T) in zip(colors, enumerate(T_vec)):
    L, mob_R, mob_I, pol, freq = 2e-3, 54e-4, 7e-4, 3.1e-36, 0.6e12
    p_Na_vec = exp_points[:, 3] * 1e4
    Na_vec = n_vec * p_Na_vec[0] / (exp_points[0, 0] / surf_area)
    sys = system_data(m_e, m_h, eps_r, T)
    lambda_th = sys.c_hbarc * sqrt(2 * pi * sys.beta / sys.m_p)

    th_pol = 21 / 2**8 * 16 * sys.c_aEM**2 * (
        sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / eb_cou**3

    err_pol = 21 / 2**8 * (sys.c_e_charge * sys.c_hbarc
                           )**2 / sys.m_p / sys.c_aEM / eb_cou**3 * err_eb_cou

    print('lambda_th: %f nm' % lambda_th)

    if data is None:
        exc_list = time_func(plasmon_density_ht_v, n_vec, N_k, sys)
        savetxt("extra/data_mu_e.csv", exc_list, delimiter=",")
        print(exc_list)
        exit()
    else:
        exc_list = data[i]

    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec
    """
    print('eps_0: %e' % 8.854e-12)
    print(exp_points[-1, 0] / (surf_area * 1e-18) / (6e-9))
    print(4 * pi * th_pol * exp_data[-1, 0] / (surf_area * 1e-18) / (6e-9))
    """

    print(sys.c_hbarc**2 * 1e-9**2 / sys.m_p * 4 * pi * 8.854e-12 * eps_r /
          sys.c_e_charge)

    print(exp_points[-1, 0] / surf_area / 1e-18 *
          (sys.c_hbarc**2 * 1e-9**2 / sys.m_p * 4 * pi * 8.854e-12 * eps_r /
           sys.c_e_charge)**2)

    q_yield_vec = n_id_vec / n_vec
    q_yield_exc_vec = n_exc_vec / n_vec

    if p_data is None:
        p_data = array(
            time_func(plasmon_density_ht_c_v, exp_points[:, 0] / surf_area,
                      N_k, sys))
        print(p_data)

    p_mu_e_lim, p_eb_lim = p_data[:2]
    p_mu_e_vec = array(p_data[2:])

    if p_eb_vec is None:
        p_eb_vec = array(
            time_func(plasmon_det_zero_ht_v, N_k, p_mu_e_vec, sys, p_eb_lim))
        print(p_eb_vec)

    p_eps_r = sys.c_aEM * sqrt(2 * sys.m_p / abs(p_eb_vec))
    p_th_pol_vec = 21 / 2**8 * 16 * sys.c_aEM**2 * (
        sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / abs(p_eb_vec)**3

    th_pol_vec = 21 / 2**8 * 16 * sys.c_aEM**2 * (
        sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / abs(eb_vec)**3

    print(p_th_pol_vec)

    p_n_vec = exp_points[:, 0] / surf_area
    p_mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in p_mu_e_vec])
    p_n_id_vec = array([sys.density_ideal(mu_e) for mu_e in p_mu_e_vec])
    p_n_exc_vec = p_n_vec - p_n_id_vec

    p_q_yield_vec = p_n_id_vec / p_n_vec
    p_q_yield_exc_vec = p_n_exc_vec / p_n_vec

    saha_const = sys.m_p / (2 * pi * sys.c_hbarc**2 * sys.beta) * exp(
        -sys.beta * eb_cou)

    saha_q_yield_vec = -saha_const * 0.5 / n_vec * (
        1 - sqrt(1 + 4 * n_vec / saha_const))

    saha_p_q_yield_vec = -saha_const * 0.5 / p_n_vec * (
        1 - sqrt(1 + 4 * p_n_vec / saha_const))
    saha_p_q_yield_exc_vec = 1 - saha_p_q_yield_vec

    saha_n_id_vec = saha_q_yield_vec * n_vec
    saha_n_exc_vec = (1 - saha_q_yield_vec) * n_vec

    export_p_data = zeros((p_n_vec.size, 3))
    export_p_data[:, 0] = p_n_vec * surf_area
    export_p_data[:, 1] = p_q_yield_vec
    export_p_data[:, 2] = -p_eb_vec

    savetxt("extra/p_data.csv", export_p_data, delimiter=",")

    export_data = zeros((n_vec.size, 3))
    export_data[:, 0] = n_vec * surf_area
    export_data[:, 1] = q_yield_vec
    export_data[:, 2] = eb_cou

    savetxt("extra/data.csv", export_data, delimiter=",")

    if plot_exp_data:
        exp_n_vec = exp_data[:, 0] / surf_area
        exp_n_id_vec = exp_data[:, 1] * exp_n_vec
        exp_n_exc_vec = (1 - exp_data[:, 1]) * exp_n_vec

    real_x = (sys.c_e_charge / L * p_Na_vec * p_q_yield_vec).reshape(-1, 1)
    real_y = exp_points[:, 1]

    real_model = sm.WLS(
        real_y, real_x, weights=1 / exp_points[:, 4]**2, has_const=False)
    real_fit = real_model.fit(use_t=True)

    imag_x = (sys.c_e_charge * p_Na_vec / L * p_q_yield_vec).reshape(-1, 1)
    imag_y = -p_Na_vec / L * freq * 2 * pi * p_q_yield_exc_vec * p_th_pol_vec + exp_points[:,2]
    imag_model = sm.WLS(
        imag_y, imag_x, weights=1 / exp_points[:, 5]**2, has_const=False)
    imag_fit = imag_model.fit(use_t=True)

    fit_mob_R, = real_fit.params
    fit_mob_I, = imag_fit.params

    err_mob_R, err_mob_I = real_fit.bse, imag_fit.bse

    print('mob_R: %f±%1.0e, mob_I: %e±%1.0e, pol: %e±%1.0e' %
          (fit_mob_R * 1e4, err_mob_R * 1e4, fit_mob_I * 1e4, err_mob_I * 1e4,
           th_pol, err_pol))

    fit_export = zeros((3, 2))
    fit_export[0, 0] = fit_mob_R * 1e4
    fit_export[0, 1] = err_mob_R * 1e4
    fit_export[1, 0] = fit_mob_I * 1e4
    fit_export[1, 1] = err_mob_I * 1e4
    fit_export[2, 0] = th_pol
    fit_export[2, 1] = err_pol

    savetxt("extra/fit.csv", fit_export, delimiter=",")

    cond_vec = array(
        time_func(plasmon_cond_v, q_yield_vec, Na_vec, L, fit_mob_R, fit_mob_I,
                  th_pol, freq, sys))

    saha_cond_vec = array(
        time_func(plasmon_cond_v, saha_q_yield_vec, Na_vec, L, fit_mob_R,
                  fit_mob_I, th_pol, freq, sys))

    #cond_vec = saha_cond_vec

    ax[0].set_title('Densities vs. photoexcitation density\n%s' % eb_type)
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

    if plot_exp_data:
        getattr(ax[0], plot_func[plot_type][0])(
            n_vec,
            saha_n_id_vec,
            '.-',
            color=c,
            label='T: $%.0f$ K, $n_e$' % sys.T,
        )

        getattr(ax[0], plot_func[plot_type][0])(
            n_vec,
            saha_n_exc_vec,
            '.--',
            color=c,
            label='T: $%.0f$ K, $n_{exc}$' % sys.T,
        )

    if plot_type == 'log':
        ax[0].set_ylim(0.01 / surf_area, 100 / surf_area)
        ax[0].set_xlim(0.5 / lambda_th**2, 80 / surf_area)
    else:
        ax[0].set_ylim(0.01 / surf_area, 100 / surf_area)
        ax[0].set_xlim(0, 60 / surf_area)

    ax[0].axvline(x=4 / lambda_th**2, color='g')

    ax[0].legend(loc=0)

    ax[1].set_title('Conductivity vs. photoexcitation density\n%s' % eb_type)
    ax[1].set_xlabel(r'$n_\gamma$ / nm$^{-2}$')
    ax[1].set_ylabel(r'$\Delta\sigma$ $(10^{-3})$ / S m$^{-1}$')

    if plot_exp_data:
        getattr(ax[1], plot_func[plot_type][1])(
            exp_points[:, 0] / surf_area,
            exp_points[:, 1] * 1e3,
            'o',
            color='k',
            label='T: $%.0f$ K, real part' % sys.T,
        )

        getattr(ax[1], plot_func[plot_type][1])(
            exp_points[:, 0] / surf_area,
            -exp_points[:, 2] * 1e3,
            '^',
            color='k',
            label='T: $%.0f$ K, imag part' % sys.T,
        )

        getattr(ax[1], plot_func[plot_type][1])(
            exp_fit[:, 0] / surf_area,
            exp_fit[:, 1] * 1e3,
            '-',
            color='k',
            label='Saha fit, real',
        )

        getattr(ax[1], plot_func[plot_type][1])(
            exp_fit[:, 0] / surf_area,
            -exp_fit[:, 2] * 1e3,
            '--',
            color='k',
            label='Saha fit, imag',
        )

        ax[1].errorbar(
            exp_points[:, 0] / surf_area,
            exp_points[:, 1] * 1e3,
            yerr=exp_points[:, 4] * 1e3,
            fmt='none',
            capsize=5,
            color='k')

        ax[1].errorbar(
            exp_points[:, 0] / surf_area,
            -exp_points[:, 2] * 1e3,
            yerr=exp_points[:, 5] * 1e3,
            fmt='none',
            capsize=5,
            color='k')

    getattr(ax[1], plot_func[plot_type][1])(
        n_vec,
        real(cond_vec) * 1e3,
        '-',
        color=c,
        label='T: $%.0f$ K, real part' % sys.T,
    )

    getattr(ax[1], plot_func[plot_type][1])(
        n_vec,
        -imag(cond_vec) * 1e3,
        '--',
        color=c,
        label='T: $%.0f$ K, imag part' % sys.T,
    )

    ax[1].set_ylim(-12.5, 4)
    if plot_type == 'log':
        ax[1].set_xlim(0.5 / lambda_th**2, 80 / surf_area)
    else:
        ax[1].set_xlim(0, 60 / surf_area)

    ax[1].axvline(x=4 / lambda_th**2, color='g')

    ax[1].axhline(y=0, color='k')
    ax[1].legend(loc=0)

    plot_exp_data = False

fig.tight_layout()
plt.show()

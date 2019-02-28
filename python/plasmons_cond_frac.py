from common import *

from sklearn.linear_model import LinearRegression

N_k = 1 << 8
eb_cou = 0.193

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

data = loadtxt('../data2.txt', delimiter=',').reshape((1, n_vec.size + 2))
p_data = loadtxt('../data_points.txt', delimiter=' ').reshape((8, ))
exp_data = loadtxt('bin/quantum_yield_charges_versus_N.csv', delimiter=',')
exp_points = loadtxt('bin/cdse_platelet_data.csv', delimiter=',')
exp_fit = loadtxt('bin/cdse_platelet_fit_data_update.csv')
eb_vec = loadtxt('../data_eb.csv', delimiter=',')

surf_area = 326.4  # nm^2
plot_exp_data = True

plot_func = {'log': ('loglog', 'semilogx'), 'linear': ('plot', 'plot')}

plot_type = 'log'

for c, (i, T) in zip(colors, enumerate(T_vec)):
    L, mob_R, mob_I, pol, freq = 2e-3, 54e-4, 7e-4, 3.1e-36, 0.6e12
    p_Na_vec = exp_points[:, 3] * 1e4
    Na_vec = n_vec * p_Na_vec[0] / (exp_points[0, 0] / surf_area)
    sys = system_data(m_e, m_h, eps_r, T)
    lambda_th = sys.c_hbarc * sqrt(2 * pi * sys.beta / sys.m_p)
    pol_theo = (sys.c_e_charge * sys.c_hbarc
                )**2 / 2 / sys.m_p / sys.c_aEM / eb_cou**2 * 21 / 2**8

    if data is None:
        n_vec = exp_points[:, 0] / surf_area
        exc_list = time_func(plasmon_density_ht_c_v, n_vec, N_k, sys)
        print(exc_list)
    else:
        exc_list = data[i]

    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    q_yield_vec = n_id_vec / n_vec
    q_yield_exc_vec = n_exc_vec / n_vec

    saha_const = sys.m_p / (2 * pi * sys.c_hbarc**2 * sys.beta) * exp(
        -sys.beta * eb_cou)

    saha_q_yield_vec = -saha_const * 0.5 / n_vec * (
        1 - sqrt(1 + 4 * n_vec / saha_const))

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

    real_x = (sys.c_e_charge / L * p_Na_vec * p_q_yield_vec).reshape(-1, 1)
    real_y = exp_points[:, 1]

    real_fit = LinearRegression(fit_intercept=False).fit(
        real_x, real_y, 1 / exp_points[:, 4]**2)

    imag_x = (sys.c_e_charge * p_Na_vec / L * p_q_yield_vec).reshape(-1, 1)
    imag_y = -p_Na_vec / L * freq * 2 * pi * p_q_yield_exc_vec * pol_theo + exp_points[:,
                                                                                       2]
    imag_fit = LinearRegression(fit_intercept=False).fit(
        imag_x, imag_y, 1 / exp_points[:, 5]**2)

    fit_mob_R, = real_fit.coef_
    fit_mob_I, = imag_fit.coef_

    print((fit_mob_R * 1e4, fit_mob_I * 1e4, pol_theo))

    cond_vec = array(
        time_func(plasmon_cond_v, q_yield_vec, Na_vec, L, fit_mob_R, fit_mob_I,
                  pol_theo, freq, sys))

    if eb_vec is None:
        eb_vec = array(
            time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys, eb_lim))

        savetxt('../data_eb.csv', eb_vec, delimiter=',')

    ax[0].set_title(
        'Quantum yield vs. photoexcitations\nMaxwell-Boltzmann -- Static')
    ax[0].set_xlabel(r'$\langle N \rangle$')
    ax[0].set_ylabel(r'$\phi(N)$')

    getattr(ax[0], plot_func[plot_type][0])(
        n_vec * surf_area,
        q_yield_vec,
        '-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    if plot_exp_data:
        getattr(ax[0], plot_func[plot_type][0])(
            n_vec * surf_area,
            saha_q_yield_vec,
            '--',
            color='k',
            label='Saha',
        )

    if plot_type == 'linear':
        ax[0].set_xlim(0, 60)
        ax[0].set_ylim(0, 0.02)

    ax[0].axvline(x=4 * surf_area / lambda_th**2, color='g')
    ax[0].axvline(x=exp_points[0, 0], color='m', linestyle='--')
    ax[0].axvline(x=exp_points[-1, 0], color='m', linestyle='--')
    ax[0].legend(loc=0)

    ax[1].set_title(
        'Binding energy vs. photoexcitations\nMaxwell-Boltzmann -- Static')
    ax[1].set_xlabel(r'$\langle N \rangle$')
    ax[1].set_ylabel(r'$\varepsilon_b$ \ eV')

    getattr(ax[1], plot_func[plot_type][1])(
        n_vec * surf_area,
        eb_vec,
        '-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    if plot_type == 'linear':
        ax[1].set_xlim(0, 60)

    ax[1].axvline(x=4 * surf_area / lambda_th**2, color='g')
    ax[1].axvline(x=exp_points[0, 0], color='m', linestyle='--')
    ax[1].axvline(x=exp_points[-1, 0], color='m', linestyle='--')
    ax[1].legend(loc=0)

    plot_exp_data = False

fig.tight_layout()
plt.show()

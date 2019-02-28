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

plot_func = {'log': ('semilogx', 'semilogx'), 'linear': ('plot', 'plot')}

plot_type = 'log'

for c, (i, T) in zip(colors, enumerate(T_vec)):
    L, freq = 2e-3, 0.6e12
    p_Na_vec = exp_points[:, 3] * 1e4
    Na_vec = n_vec * p_Na_vec[0] / (exp_points[0, 0] / surf_area)
    sys = system_data(m_e, m_h, eps_r, T)
    lambda_th = sys.c_hbarc * sqrt(2 * pi * sys.beta / sys.m_p)

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

    if eb_vec is None:
        eb_vec = array(
            time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys, eb_lim))

        savetxt('../data_eb.csv', eb_vec, delimiter=',')

    th_pol_vec = (sys.c_e_charge * sys.c_hbarc
                  )**2 / 2 / sys.m_p / sys.c_aEM / eb_vec**2 * 21 / 2**8

    ax[0].set_title(
        'Polarizability vs. photoexcitation density\nMaxwell-Boltzmann -- Static'
    )
    ax[0].set_xlabel(r'$n_\gamma$ / nm$^{-2}$')
    ax[0].set_ylabel(r'$\alpha$ $(10^{-36})$ / C m$^2$ V$^{-1}$')

    getattr(ax[0], plot_func[plot_type][0])(
        n_vec,
        th_pol_vec * 1e36,
        '-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    if plot_type == 'linear':
        ax[0].set_xlim(0, 60 / surf_area)

    ax[0].axvline(x=4 / lambda_th**2, color='g')
    ax[0].axvline(x=exp_points[0, 0] / surf_area, color='m', linestyle='--')
    ax[0].axvline(x=exp_points[-1, 0] / surf_area, color='m', linestyle='--')
    ax[0].legend(loc=0)

    ax[1].set_title(
        'Binding energy vs. photoexcitation density\nMaxwell-Boltzmann -- Static'
    )
    ax[1].set_xlabel(r'$n_\gamma$ / nm$^{-2}$')
    ax[1].set_ylabel(r'$\varepsilon_b$ / eV')

    getattr(ax[1], plot_func[plot_type][1])(
        n_vec,
        eb_vec,
        '-',
        color=c,
        label='T: $%.0f$ K' % sys.T,
    )

    if plot_type == 'linear':
        ax[1].set_xlim(0, 60 / surf_area)

    ax[1].axvline(x=4 / lambda_th**2, color='g')
    ax[1].axvline(x=exp_points[0, 0] / surf_area, color='m', linestyle='--')
    ax[1].axvline(x=exp_points[-1, 0] / surf_area, color='m', linestyle='--')
    ax[1].legend(loc=0)

    plot_exp_data = False

fig.tight_layout()
plt.show()

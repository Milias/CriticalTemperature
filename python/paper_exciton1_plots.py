from common import *
import statsmodels.api as sm

N_k = 1 << 8

eb_cou = 0.193
err_eb_cou = 0.005

m_e, m_h, eps_r, T = 0.12, 0.3, 4.90185, 294  # K
sys = system_data(m_e, m_h, eps_r, T)

eps_r = sys.c_aEM * sqrt(2 * sys.m_p / eb_cou)
sys = system_data(m_e, m_h, eps_r, T)

n_x, n_y = 1, 1
fig = plt.figure(figsize=(6.8, 5.3), dpi=300)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]
plot_func = {'log': ('semilogx', 'semilogx'), 'linear': ('plot', 'plot')}

a0 = sys.eps_r / sys.c_aEM * sys.c_hbarc / sys.m_p
Lx, Ly = 34.0, 10.0  # nm
surf_area = 326.4  # nm^2
#surf_area = Lx * Ly  # nm^2


def diffusion_cx(w, ax, ay, D):
    sqrt_factor_x = sqrt(-1j * w / D) * ax
    sqrt_factor_y = sqrt_factor_x / ax * ay
    return D * (1.0 + tan(-0.5 * sqrt_factor_x) / sqrt_factor_x +
                tan(-0.5 * sqrt_factor_y) / sqrt_factor_y)


def mob_integ_func(u_dc_vec, w_vec, power_norm_vec, mu_dc_bulk, sys):
    # d = mu / beta / e
    mu_dc_vec = mu_dc_bulk * exp(u_dc_vec)
    diff_factor = 1e14 / sys.beta

    d_vec = mu_dc_vec * diff_factor  # nm^2 s^-1

    mob_vec = array(
        [diffusion_cx(w_vec, Lx, Ly, d) / diff_factor for d in d_vec])

    mob_norm_vec = mob_vec * power_norm_vec

    return simps(mob_norm_vec, w_vec, axis=1)


print(a0)


def real_space_lwl_potential(plot_type='linear'):
    ls_vec = logspace(log10(0.05 * sys.sys_ls), log10(0.9 * sys.sys_ls), 5)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, ls_vec.size)
    ]

    x_vec = linspace(1e-2, 8, 300) / a0

    ax[0].set_xlabel(r'$r$ / $a_0$')
    ax[0].set_ylabel(r'$V(r)$ (eV)')

    cou_vec = array(time_func(plasmon_rpot_lwl_v, x_vec, 1e-8, sys))
    getattr(ax[0], plot_func[plot_type][0])(
        x_vec,
        cou_vec,
        '--',
        color='r',
        label=r'$\lambda_s \rightarrow \infty$',
    )

    for c, (i, ls) in zip(colors, enumerate(ls_vec)):
        y_vec = array(time_func(plasmon_rpot_lwl_v, x_vec, ls, sys))
        getattr(ax[0], plot_func[plot_type][0])(
            x_vec,
            y_vec,
            '-',
            color=c,
            label='$\lambda_s$: $%.1f$ nm' % (1 / ls),
        )

    sys_vec = array(time_func(plasmon_rpot_lwl_v, x_vec, sys.sys_ls, sys))
    getattr(ax[0], plot_func[plot_type][0])(
        x_vec,
        sys_vec,
        '--',
        color='k',
        label='$\lambda_{s}$: $\lambda_{s,0} = %.1f$ nm' % (1 / sys.sys_ls),
    )

    ax[0].set_ylim(-0.8, 0.1)
    ax[0].set_xlim(0, x_vec[-1])
    ax[0].axhline(y=0, color='k')

    x_vec_top = ax[0].xaxis.get_majorticklocs()[:-1]
    x_vec_vals = x_vec_top * a0
    x_vec_vals = ['%.1f' % v for v in x_vec_vals]

    ax_top = ax[0].twiny()
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel('$r$ (nm)')

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'real_space_lwl_potential'


def real_space_mb_potential(plot_type='linear'):
    mu_e_vec = -logspace(log(0.4), -2, 5)
    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, mu_e_vec.size)
    ]

    x_vec = linspace(1e-2, 4, 500)

    #ax[0].set_title('Real space potential\nClassical Limit')
    ax[0].set_xlabel(r'$r$ (nm)')
    ax[0].set_ylabel(r'$V(r)$ (eV)')

    cou_vec = array(time_func(plasmon_rpot_lwl_v, x_vec, 1e-8, sys))
    getattr(ax[0], plot_func[plot_type][0])(
        x_vec,
        cou_vec,
        '--',
        color='r',
        label='Coulomb limit',
    )

    for c, (i, (mu_e, mu_h)) in zip(colors, enumerate(zip(mu_e_vec,
                                                          mu_h_vec))):

        y_vec = array(time_func(plasmon_rpot_ht_v, x_vec, mu_e, mu_h, sys))
        getattr(ax[0], plot_func[plot_type][0])(
            x_vec,
            y_vec,
            '-',
            color=c,
            label='$\mu_e$: $%.1f\cdot10^{-2}$ eV' % (1e2 * mu_e),
        )

    sys_vec = array(time_func(plasmon_rpot_lwl_v, x_vec, sys.sys_ls, sys))
    getattr(ax[0], plot_func[plot_type][0])(
        x_vec,
        sys_vec,
        '--',
        color='k',
        label='$\lambda_{s,0}$: $%.1f$ nm' % (1 / sys.sys_ls),
    )

    ax[0].set_ylim(-0.5, 0.1)
    ax[0].set_xlim(0, x_vec[-1])
    ax[0].axhline(y=0, color='k')

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'real_space_mb_potential'


def real_space_mb_potential_density(plot_type='linear'):
    mu_e_vec = -logspace(log10(0.07), -2.3, 5)
    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])

    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, mu_e_vec.size)
    ]

    x_vec = linspace(1e-2, 8, 300) / a0

    #ax[0].set_title('Real space potential\nClassical Limit')
    ax[0].set_xlabel(r'$r$ / $a_0$')
    ax[0].set_ylabel(r'$V(r)$ (eV)')

    cou_vec = array(time_func(plasmon_rpot_lwl_v, x_vec, 1e-8, sys))
    getattr(ax[0], plot_func[plot_type][0])(
        x_vec,
        cou_vec,
        '--',
        color='r',
        label=r'$\langle N_e\rangle$: %d' % 0,
    )

    for c, (i, (mu_e, mu_h,
                n_id)) in zip(colors,
                              enumerate(zip(mu_e_vec, mu_h_vec, n_id_vec))):

        num_e = n_id * surf_area
        y_vec = array(time_func(plasmon_rpot_ht_v, x_vec, mu_e, mu_h, sys))
        getattr(ax[0], plot_func[plot_type][0])(
            x_vec,
            y_vec,
            '-',
            color=c,
            label=r'$\langle N_e\rangle$: %.1f' % num_e,
        )

    sys_vec = array(time_func(plasmon_rpot_lwl_v, x_vec, sys.sys_ls, sys))
    getattr(ax[0], plot_func[plot_type][0])(
        x_vec,
        sys_vec,
        '--',
        color='k',
        label=r'$\langle N_e\rangle\rightarrow\infty$',
    )

    ax[0].set_ylim(-0.8, 0.1)
    ax[0].set_xlim(0, x_vec[-1])
    ax[0].axhline(y=0, color='k')

    x_vec_top = ax[0].xaxis.get_majorticklocs()[:-1]
    x_vec_vals = x_vec_top * a0
    x_vec_vals = ['%.1f' % v for v in x_vec_vals]

    ax_top = ax[0].twiny()
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel('$r$ (nm)')

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'real_space_mb_potential_density'


def energy_level_mb(plot_type='linear'):
    T_vec = linspace(100, 400, 8)
    mu_e_vec = linspace(-0.12, 0.0, 48)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, T_vec.size)
    ]

    #ax[0].set_title('Exciton energy levels\nClassical Limit')
    ax[0].set_xlabel(r'$\mu_e$ (eV)')
    ax[0].set_ylabel(r'$\epsilon(\mu_e)$ (eV)')

    sys = system_data(m_e, m_h, eps_r, 300)

    ax[0].axhline(
        y=sys.get_E_n(0.5),
        color='r',
        linestyle='--',
        label='Coulomb limit',
    )

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        sys = system_data(m_e, m_h, eps_r, T)

        y_vec = array(time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys))
        getattr(ax[0], plot_func[plot_type][0])(
            mu_e_vec,
            y_vec,
            '-',
            color=c,
            label='T: %.0f K' % sys.T,
        )

    z_sys_lwl = time_func(plasmon_det_zero_lwl, N_k, sys.sys_ls, sys)

    ax[0].set_xlim(mu_e_vec[0], mu_e_vec[-1])
    ax[0].axhline(
        y=z_sys_lwl,
        color='k',
        linestyle='--',
        label='$\epsilon(\lambda_{s,0}) = %0.2f$ eV' % z_sys_lwl,
    )

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'energy_level_mb'


def energy_level_mb_density(plot_type='log'):
    T_vec = linspace(130, 350, 5)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, T_vec.size)
    ]

    ax[0].set_xlabel(r'$\langle N_e \rangle$')
    ax[0].set_ylabel(r'$\epsilon(\langle N_e \rangle)$ (eV)')

    sys = system_data(m_e, m_h, eps_r, 300)

    ax[0].axhline(
        y=sys.get_E_n(0.5),
        color='r',
        linestyle='--',
        label=r'$\langle N_e\rangle$: %d' % 0,
    )

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        sys = system_data(m_e, m_h, eps_r, T)
        mu_e_vec = linspace(-6, 0.0, 48) / sys.beta

        mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
        n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])

        y_vec = array(time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys))
        getattr(ax[0], plot_func[plot_type][0])(
            n_id_vec * surf_area,
            y_vec,
            '-',
            color=c,
            label='T: %.0f K' % sys.T,
        )
        getattr(ax[0], plot_func[plot_type][0])(
            n_id_vec[-1] * surf_area,
            y_vec[-1],
            'o',
            color=c,
        )

    z_sys_lwl = time_func(plasmon_det_zero_lwl, N_k, sys.sys_ls, sys)

    ax[0].set_xlim(1e-4 * surf_area, 1.1e-2 * surf_area)
    ax[0].axhline(
        y=z_sys_lwl,
        color='k',
        linestyle='--',
        label=r'$\langle N_e\rangle\rightarrow\infty$',
    )

    x_vec_top = ax[0].xaxis.get_majorticklocs()[2:-2]
    x_vec_vals = (x_vec_top / surf_area) * 1e3
    x_vec_vals = [('%%.%df' % (2 + min(0, -log10(v)))) % v for v in x_vec_vals]

    ax[0].set_xticklabels(
        ['%.1f' % n for n in ax[0].xaxis.get_majorticklocs()])

    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel(r'$n_e$ ($10^{-3}$ nm$^{-2}$)')

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'energy_level_mb_density'


def energy_level_mb_limit_density(plot_type='linear'):
    T_vec = linspace(130, 350, 5)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, T_vec.size)
    ]

    ax[0].set_xlabel(r'$\langle N_e \rangle$')
    ax[0].set_ylabel(r'$\epsilon(\langle N_e \rangle)$ (eV)')

    sys = system_data(m_e, m_h, eps_r, 300)

    ax[0].axhline(
        y=sys.get_E_n(0.5),
        color='r',
        linestyle='--',
        label=r'$\langle N_e\rangle$: %d' % 0,
    )

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        sys = system_data(m_e, m_h, eps_r, T)
        mu_e_vec = linspace(-8, -1, 48) / sys.beta

        mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
        n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])

        y_vec = array(time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys))
        line_vec = mu_e_vec + mu_h_vec

        mu_e_lim, eb_lim = plasmon_exc_mu_lim(N_k, sys)
        n_id_lim = sys.density_ideal(mu_e_lim)

        getattr(ax[0], plot_func[plot_type][0])(
            n_id_vec * surf_area,
            y_vec,
            '-',
            color=c,
            label='T: %.0f K' % sys.T,
        )

        getattr(ax[0], plot_func[plot_type][0])(
            n_id_vec * surf_area,
            line_vec,
            ':',
            color=c,
        )

        getattr(ax[0], plot_func[plot_type][0])(
            [n_id_lim * surf_area],
            [eb_lim],
            's',
            color=c,
        )

    T_vec, mu_e_lim, eb_lim = loadtxt('extra/e_lim_data_higher_t_log.csv',
                                      delimiter=',').T

    ax[0].set_xlim(0.5e-5 * surf_area, 0.3e-2 * surf_area)
    ax[0].set_ylim(-0.195, -0.175)

    x_vec_top = ax[0].xaxis.get_majorticklocs()[1:-1]
    x_vec_vals = (x_vec_top / surf_area) * 1e3
    x_vec_vals = [('%%.%df' % (1 + abs(log10(v)))) % v for v in x_vec_vals]

    ax[0].set_xticklabels([('%%.%df' % abs(log10(v))) % v
                           for v in ax[0].xaxis.get_majorticklocs()])

    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel(r'$n_e$ ($10^{-3}$ nm$^{-2}$)')

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'energy_level_mb_limit_density'


def eb_lim_temperature(plot_type='linear'):
    T_vec = logspace(log10(100), log10(3e3), 48)
    sys_vec = [system_data(m_e, m_h, eps_r, T) for T in T_vec]

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, T_vec.size)
    ]

    ax[0].set_xlabel(r'$T$ (K)')
    ax[0].set_ylabel(r'$\epsilon(T)$ (eV)')
    """
    mu_e_lim, eb_lim = array(
        [time_func(plasmon_exc_mu_lim, N_k, sys) for sys in sys_vec]).T

    y_vec = array([
        time_func(plasmon_det_zero_ht, N_k, 0.0, sys.get_mu_h(0.0), sys)
        for sys in sys_vec
    ])

    export_data = zeros((T_vec.size, 4))
    export_data[:, 0] = T_vec
    export_data[:, 1] = mu_e_lim
    export_data[:, 2] = eb_lim
    export_data[:, 3] = y_vec

    savetxt("extra/e_lim_data_higher_t_log.csv", export_data, delimiter=",")
    """
    T_vec, mu_e_lim, eb_lim, y_vec = loadtxt(
        'extra/e_lim_data_higher_t_log.csv', delimiter=',').T
    """
    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_lim])
    y_lim_vec = array([
        time_func(plasmon_det_zero_ht, N_k, mu_e, mu_h, sys)
        for (mu_e, mu_h, sys) in zip(mu_e_lim, mu_h_vec, sys_vec)
    ])
    """

    ax[0].axhline(
        y=sys.get_E_n(0.5),
        color='r',
        linestyle='--',
        label=r'$\langle N_e\rangle$: %d' % 0,
    )
    getattr(ax[0], plot_func[plot_type][0])(
        T_vec,
        eb_lim,
        '-',
        color='b',
        label='$\epsilon(\mu_{e,0})$',
    )
    getattr(ax[0], plot_func[plot_type][0])(
        T_vec,
        y_vec,
        '-',
        color='m',
        label='$\epsilon(\mu_e = 0)$',
    )

    z_sys_lwl = time_func(plasmon_det_zero_lwl, N_k, sys.sys_ls, sys)
    ax[0].axhline(
        y=z_sys_lwl,
        color='k',
        linestyle='--',
        label=r'$\langle N_e\rangle\rightarrow\infty$',
    )
    ax[0].axvline(
        x=1208,
        color='m',
        linestyle=':',
        label='$T_{max}$',
    )
    ax[0].axvline(
        x=293,
        color='g',
        linestyle=':',
        label='$T_{exp}$',
    )

    #ax[0].set_xlim(0.5e-5 * surf_area, 0.3e-2 * surf_area)
    #ax[0].set_ylim(-0.195, -0.175)
    """
    x_vec_top = ax[0].xaxis.get_majorticklocs()[1:-1]
    x_vec_vals = (x_vec_top / surf_area) * 1e3
    x_vec_vals = [('%%.%df' % (1+abs(log10(v)))) % v for v in x_vec_vals]

    ax[0].set_xticklabels([('%%.%df' % abs(log10(v))) % v for v in ax[0].xaxis.get_majorticklocs()])

    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel(r'$n_e$ ($10^{-3}$ nm$^{-2}$)')
    """

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'eb_lim_temperature'


def mu_e_lim_temperature(plot_type='linear'):
    T_vec = logspace(log10(100), log10(3e3), 48)
    sys_vec = [system_data(m_e, m_h, eps_r, T) for T in T_vec]
    beta_vec = array([sys.beta for sys in sys_vec])

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, T_vec.size)
    ]

    ax[0].set_xlabel(r'$T$ (K)')
    ax[0].set_ylabel(r'$\beta\mu_{e,0}(T)$')
    """
    mu_e_lim, eb_lim = array(
        [time_func(plasmon_exc_mu_lim, N_k, sys) for sys in sys_vec]).T

    y_vec = array([
        time_func(plasmon_det_zero_ht, N_k, 0.0, sys.get_mu_h(0.0), sys)
        for sys in sys_vec
    ])

    export_data = zeros((T_vec.size, 4))
    export_data[:, 0] = T_vec
    export_data[:, 1] = mu_e_lim
    export_data[:, 2] = eb_lim
    export_data[:, 3] = y_vec

    savetxt("extra/e_lim_data_higher_t_log.csv", export_data, delimiter=",")
    """
    T_vec, mu_e_lim, eb_lim, y_vec = loadtxt(
        'extra/e_lim_data_higher_t_log.csv', delimiter=',').T

    getattr(ax[0], plot_func[plot_type][0])(
        T_vec,
        beta_vec * mu_e_lim,
        '-',
        color='b',
        #label='$\mu_{e,0}(T)$',
    )

    ax[0].axhline(
        y=0,
        color='k',
        linestyle='-',
    )
    ax[0].axvline(
        x=1208,
        color='m',
        linestyle=':',
        label='$T_{max}$',
    )
    ax[0].axvline(
        x=293,
        color='g',
        linestyle=':',
        label='$T_{exp}$',
    )

    #ax[0].set_xlim(0.5e-5 * surf_area, 0.3e-2 * surf_area)
    ax[0].set_ylim(-5, 1)
    """
    x_vec_top = ax[0].xaxis.get_majorticklocs()[1:-1]
    x_vec_vals = (x_vec_top / surf_area) * 1e3
    x_vec_vals = [('%%.%df' % (1+abs(log10(v)))) % v for v in x_vec_vals]

    ax[0].set_xticklabels([('%%.%df' % abs(log10(v))) % v for v in ax[0].xaxis.get_majorticklocs()])

    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel(r'$n_e$ ($10^{-3}$ nm$^{-2}$)')
    """

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'mu_e_lim_temperature'


def density_result(plot_type='plot'):
    sys = system_data(m_e, m_h, eps_r, 294)
    T_vec = linspace(294, 295, 1)
    sys_vec = [system_data(m_e, m_h, eps_r, T) for T in T_vec]

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, T_vec.size)
    ]

    n_vec = logspace(-5, 1, 1 << 7)

    data = loadtxt('../data2.txt', delimiter=',').reshape((1, n_vec.size + 2))
    p_data = loadtxt('../data_points.txt', delimiter=' ').reshape((8, ))

    exp_data = loadtxt('bin/quantum_yield_charges_versus_N.csv', delimiter=',')
    exp_points = loadtxt('bin/cdse_platelet_data.csv', delimiter=',')
    exp_fit = loadtxt('bin/cdse_platelet_fit_data_update.csv')

    p_eb_vec = loadtxt('../data_eb_points.txt', delimiter=' ')

    ax[0].set_xlabel(r'$n_\gamma a_0^2$')
    ax[0].set_ylabel(r'Average Number of Particles per Bohr Area')

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        exc_list = data[i]

        mu_e_lim, eb_lim = exc_list[:2]
        mu_e_vec = array(exc_list[2:])

        mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
        n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
        n_exc_vec = n_vec - n_id_vec

        print(eb_lim)
        print(mu_e_vec[-1] + mu_h_vec[-1])

        eb_v0, eb_v1 = [
            func(
                N_k,
                array([mu_e_vec[-1]]),
                sys,
            )[0] for func in [plasmon_det_zero_ht_v, plasmon_det_zero_ht_v1]
        ]

        print((eb_v0, eb_v1))

        n_exc_v1 = sys.density_exc_ht(mu_e_vec[-1] + mu_h_vec[-1], eb_v1)

        print(n_exc_v1 / (n_exc_vec[-1] + n_exc_v1) * 100)
        print(n_exc_v1 / n_vec[-1] * 100)

        getattr(ax[0], plot_type)(
            n_vec * a0**2,
            n_id_vec * a0**2,
            '--',
            color=c,
            label=r'Our model: $n_e a_0^2$',
        )
        getattr(ax[0], plot_type)(
            n_vec * a0**2,
            n_exc_vec * a0**2,
            '-',
            color=c,
            label=r'Our model: $n_{exc} a_0^2$',
        )

    ax[0].axvline(
        x=exp_points[0, 0] / surf_area * a0**2,
        color='m',
        linestyle=':',
    )

    ax[0].axvline(
        x=exp_points[-1, 0] / surf_area * a0**2,
        color='m',
        linestyle=':',
    )

    ax[0].set_xlim(1e-3, 1e2)

    lambda_th = sys.c_hbarc * sqrt(2 * pi * sys.beta / sys.m_p)
    ax[0].axvline(
        x=4 * a0**2 / lambda_th**2,
        color='g',
        label='Saha model limit',
    )

    x_vec_top = ax[0].xaxis.get_majorticklocs()[1:-1]
    x_vec_vals = x_vec_top / a0**2
    x_vec_vals = ['$10^{%.0f}$' % log10(v) for v in x_vec_vals]

    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel(r'$n_\gamma$ (nm$^{-2}$)')

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'density_result_%s' % plot_type


def eb_photo_density(plot_type='plot'):
    sys = system_data(m_e, m_h, eps_r, 294)
    T_vec = linspace(294, 295, 1)
    sys_vec = [system_data(m_e, m_h, eps_r, T) for T in T_vec]

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, T_vec.size)
    ]

    n_vec = logspace(-5, 1, 1 << 7)

    data = loadtxt('../data2.txt', delimiter=',').reshape((1, n_vec.size + 2))
    p_data = loadtxt('../data_points.txt', delimiter=' ').reshape((8, ))

    exp_data = loadtxt('bin/quantum_yield_charges_versus_N.csv', delimiter=',')
    exp_points = loadtxt('bin/cdse_platelet_data.csv', delimiter=',')
    exp_fit = loadtxt('bin/cdse_platelet_fit_data_update.csv')

    try:
        eb_vec = loadtxt('extra/eb_vec_pol.csv', delimiter=',')
    except:
        eb_vec = None

    ax[0].set_xlabel(r'$n_\gamma a_0^2$')
    ax[0].set_ylabel(r'$\epsilon$ (meV)')

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        exc_list = data[i]

        mu_e_lim, eb_lim = exc_list[:2]
        mu_e_vec = array(exc_list[2:])

        mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
        n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
        n_exc_vec = n_vec - n_id_vec

        if eb_vec is None:
            eb_vec = array(
                time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys, eb_lim))
            savetxt('extra/eb_vec_pol.csv', eb_vec, delimiter=',')
        """
        getattr(ax[0], plot_type)(
            n_vec * a0**2,
            (mu_e_vec + mu_h_vec) / eb_vec,
            '--',
            color=c,
            label=r'T: %.0f K' % T,
        )
        """

        getattr(ax[0], plot_type)(
            n_vec * a0**2,
            eb_vec * 1e3,
            '-',
            color=c,
            label=r'T: %.0f K' % T,
        )

    ax[0].axvline(
        x=exp_points[0, 0] / surf_area * a0**2,
        color='m',
        linestyle=':',
    )

    ax[0].axvline(
        x=exp_points[-1, 0] / surf_area * a0**2,
        color='m',
        linestyle=':',
    )

    ax[0].set_xlim(n_vec[0] * a0**2, n_vec[-1] * a0**2)
    """
    lambda_th = sys.c_hbarc * sqrt(2 * pi * sys.beta / sys.m_p)
    ax[0].axvline(
        x=4 * a0**2 / lambda_th**2,
        color='g',
        label='Saha model limit',
    )
    """

    x_vec_top = ax[0].xaxis.get_majorticklocs()[2:-2]
    x_vec_vals = x_vec_top / a0**2
    x_vec_vals = ['$10^{%.0f}$' % log10(v) for v in x_vec_vals]

    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel(r'$n_\gamma$ (nm$^{-2}$)')

    th_pol_vec = 21 / 2**8 * 16 * sys.c_aEM**2 * (
        sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / abs(eb_vec)**3

    y_vec_right = ax[0].yaxis.get_majorticklocs()[1:-1]
    #y_vec_vals_right = (y_vec_right - y_vec_right[0]) / (y_vec_right[-1] - y_vec_right[0]) * (th_pol_vec[-1] - th_pol_vec[0]) + th_pol_vec[0]
    y_vec_vals_right = 21 / 2**8 * 16 * sys.c_aEM**2 * (
        sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / abs(
            y_vec_right * 1e-3)**3

    y_vec_vals_right = ['%.2f' % (v * 1e36) for v in y_vec_vals_right]

    axins = ax[0].inset_axes([0.1, 0.5, 0.44, 0.47])
    axins.set_ylabel(r'$\alpha$ ($10^{-36}$ cm$^2$ / V)')

    axins.axvline(
        x=exp_points[0, 0] / surf_area * a0**2,
        color='m',
        linestyle=':',
    )

    axins.axvline(
        x=exp_points[-1, 0] / surf_area * a0**2,
        color='m',
        linestyle=':',
    )

    axins.set_xlim(n_vec[0] * a0**2, n_vec[-1] * a0**2)

    getattr(axins, plot_type)(
        n_vec * a0**2,
        th_pol_vec * 1e36,
        '-',
        color=c,
        label=r'T: %.0f K' % T,
    )

    #ax[0].indicate_inset_zoom(axins)
    """
    ax_right = ax[0].twinx()
    ax_right.set_ylim(ax[0].get_ylim())
    ax_right.set_yticks(y_vec_right)
    ax_right.set_yticklabels(y_vec_vals_right)
    ax_right.set_ylabel(r'$\alpha$ ($10^{-36}$ cm$^2$ / V)')
    """

    ax[0].legend(loc='lower right')

    fig.tight_layout()

    return 'eb_photo_density_%s' % plot_type


def cond_fit(plot_type='plot'):
    n_vec = logspace(-5, 1, 1 << 7)

    data = loadtxt('../data2.txt', delimiter=',').reshape((1, n_vec.size + 2))
    p_data = loadtxt('../data_points.txt', delimiter=' ').reshape((8, ))

    exp_data = loadtxt('bin/quantum_yield_charges_versus_N.csv', delimiter=',')
    exp_points = loadtxt('bin/cdse_platelet_data.csv', delimiter=',')
    exp_fit = loadtxt('bin/cdse_platelet_fit_data_update.csv')

    p_eb_vec = loadtxt('../data_eb_points.txt', delimiter=' ')

    try:
        eb_vec = loadtxt('extra/eb_vec.csv', delimiter=',')
    except:
        eb_vec = None

    sys = system_data(m_e, m_h, eps_r, 294)
    T_vec = linspace(294, 295, 1)
    sys_vec = [system_data(m_e, m_h, eps_r, T) for T in T_vec]

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, T_vec.size)
    ]

    L, mob_R, mob_I, pol, freq = 2e-3, 54e-4, 7e-4, 3.1e-36, 0.6e12
    p_Na_vec = exp_points[:, 3] * 1e4
    Na_vec = n_vec * p_Na_vec[0] / (exp_points[0, 0] / surf_area)

    print(p_Na_vec)

    err_eb_cou = 0.005

    th_pol = 21 / 2**8 * 16 * sys.c_aEM**2 * (sys.c_hbarc * 1e-9 /
                                              eps_r)**2 * sys.c_e_charge / abs(
                                                  sys.get_E_n(0.5))**3

    err_pol = 21 / 2**8 * (sys.c_e_charge * sys.c_hbarc
                           )**2 / sys.m_p / sys.c_aEM / eb_cou**3 * err_eb_cou

    #p_eps_r = sys.c_aEM * sqrt(2 * sys.m_p / abs(p_eb_vec))
    p_th_pol_vec = 21 / 2**8 * 16 * sys.c_aEM**2 * (
        sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / abs(p_eb_vec)**3

    p_mu_e_lim, p_eb_lim = p_data[:2]
    p_mu_e_vec = array(p_data[2:])
    p_n_vec = exp_points[:, 0] / surf_area
    p_mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in p_mu_e_vec])
    p_n_id_vec = array([sys.density_ideal(mu_e) for mu_e in p_mu_e_vec])
    p_n_exc_vec = p_n_vec - p_n_id_vec
    p_q_yield_vec = p_n_id_vec / p_n_vec
    p_q_yield_exc_vec = p_n_exc_vec / p_n_vec

    real_x = (sys.c_e_charge / L * p_Na_vec * p_q_yield_vec).reshape(-1, 1)
    real_y = exp_points[:, 1]

    real_model = sm.WLS(real_y,
                        real_x,
                        weights=1 / exp_points[:, 4]**2,
                        has_const=False)
    real_fit = real_model.fit(use_t=True)

    imag_x = (sys.c_e_charge * p_Na_vec / L * p_q_yield_vec).reshape(-1, 1)
    imag_y = exp_points[:,
                        2] - p_Na_vec / L * freq * 2 * pi * p_q_yield_exc_vec * p_th_pol_vec * 2 / 3

    imag_model = sm.WLS(
        imag_y,
        imag_x,
        weights=1 / exp_points[:, 5]**2,
        has_const=False,
    )
    imag_fit = imag_model.fit(use_t=True)

    fit_mob_R, = real_fit.params
    fit_mob_I, = imag_fit.params

    err_mob_R, err_mob_I = real_fit.bse, imag_fit.bse

    diff_model_mob = (70.67751788855645 + 117.73709882180198j) * 1e-4
    best_mob = (71.20458856 + 93.90863147j) * 1e-4

    print('mob_R: %f±%1.0e, mob_I: %e±%1.0e, pol: %e±%1.0e' %
          (fit_mob_R * 1e4, err_mob_R * 1e4, fit_mob_I * 1e4, err_mob_I * 1e4,
           th_pol, err_pol))

    ax[0].set_xlabel(r'$n_\gamma a_0^2$')
    ax[0].set_ylabel(r'$\Delta\sigma$ ($10^{-3}$ S m$^{-1}$)')

    getattr(ax[0], plot_type)(
        exp_points[:, 0] / surf_area * a0**2,
        exp_points[:, 1] * 1e3,
        'o',
        color='k',
        #label='T: $%.0f$ K, real part' % sys.T,
    )

    getattr(ax[0], plot_type)(
        exp_points[:, 0] / surf_area * a0**2,
        -exp_points[:, 2] * 1e3,
        '^',
        color='k',
        #label='T: $%.0f$ K, imag part' % sys.T,
    )
    """
    getattr(ax[0], plot_type)(
        exp_fit[:, 0],
        exp_fit[:, 1] * 1e3,
        '--',
        color='k',
        label='Saha fit, real',
    )

    getattr(ax[0], plot_type)(
        exp_fit[:, 0],
        -exp_fit[:, 2] * 1e3,
        '-',
        color='k',
        label='Saha fit, imag',
    )
    """

    ax[0].errorbar(exp_points[:, 0] / surf_area * a0**2,
                   exp_points[:, 1] * 1e3,
                   yerr=exp_points[:, 4] * 1e3,
                   fmt='none',
                   capsize=5,
                   color='k')

    ax[0].errorbar(exp_points[:, 0] / surf_area * a0**2,
                   -exp_points[:, 2] * 1e3,
                   yerr=exp_points[:, 5] * 1e3,
                   fmt='none',
                   capsize=5,
                   color='k')

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        sys = system_data(m_e, m_h, eps_r, T)
        exc_list = data[i]

        mu_e_lim, eb_lim = exc_list[:2]
        mu_e_vec = array(exc_list[2:])

        if eb_vec is None:
            eb_vec = array(
                time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys, eb_lim))
            savetxt('extra/eb_vec.csv', eb_vec, delimiter=',')

        th_pol_vec = 21 / 2**8 * 16 * sys.c_aEM**2 * (
            sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / abs(eb_vec)**3

        mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
        n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
        n_exc_vec = n_vec - n_id_vec

        q_yield_vec = n_id_vec / n_vec
        q_yield_exc_vec = n_exc_vec / n_vec

        cond_vec = (
            Na_vec * sys.c_e_charge / L * fit_mob_R * q_yield_vec) + 1j * (
                Na_vec * sys.c_e_charge / L * fit_mob_I * q_yield_vec + 2 / 3 *
                2 * pi * th_pol_vec * freq * Na_vec / L * q_yield_exc_vec)

        cond_diff_vec = (
            Na_vec * sys.c_e_charge / L * real(diff_model_mob) * q_yield_vec
        ) + 1j * (
            Na_vec * sys.c_e_charge / L * imag(diff_model_mob) * q_yield_vec +
            2 / 3 * 2 * pi * th_pol_vec * freq * Na_vec / L * q_yield_exc_vec)

        cond_best_vec = (
            Na_vec * sys.c_e_charge / L * real(best_mob) * q_yield_vec
        ) + 1j * (
            Na_vec * sys.c_e_charge / L * imag(best_mob) * q_yield_vec +
            2 / 3 * 2 * pi * th_pol_vec * freq * Na_vec / L * q_yield_exc_vec)

        getattr(ax[0], plot_type)(
            n_vec * a0**2,
            real(cond_vec) * 1e3,
            '--',
            color=c,
            label=r'Fitting. $\mu_R: %.0f±%d$ cm$^2$ V$^{-1}$ s$^{-1}$' %
            (fit_mob_R * 1e4, err_mob_R * 1e4),
        )
        getattr(ax[0], plot_type)(
            n_vec * a0**2,
            -imag(cond_vec) * 1e3,
            '-',
            color=c,
            label=r'Fitting. $\mu_I: %.0f±%d$ cm$^2$ V$^{-1}$ s$^{-1}$' %
            (fit_mob_I * 1e4, err_mob_I * 1e4),
        )

        getattr(ax[0], plot_type)(
            n_vec * a0**2,
            real(cond_best_vec) * 1e3,
            ':',
            color='b',
            label=r'Diffusion. $\mu_R: %.0f$ cm$^2$ V$^{-1}$ s$^{-1}$' %
            (real(best_mob) * 1e4),
        )
        getattr(ax[0], plot_type)(
            n_vec * a0**2,
            -imag(cond_best_vec) * 1e3,
            '-.',
            color='b',
            label=r'Diffusion. $\mu_I: %.0f$ cm$^2$ V$^{-1}$ s$^{-1}$' %
            (imag(best_mob) * 1e4),
        )

        """
        getattr(ax[0], plot_type)(
            n_vec * a0**2,
            real(cond_diff_vec) * 1e3,
            '--',
            color='b',
            label=r'Diffusion. $\mu_R: %.0f$ cm$^2$ V$^{-1}$ s$^{-1}$' %
            (real(diff_model_mob) * 1e4),
        )
        getattr(ax[0], plot_type)(
            n_vec * a0**2,
            -imag(cond_diff_vec) * 1e3,
            '-',
            color='b',
            label=r'Diffusion. $\mu_I: %.0f$ cm$^2$ V$^{-1}$ s$^{-1}$' %
            (imag(diff_model_mob) * 1e4),
        )

        getattr(ax[0], plot_type)(
            n_vec * a0**2,
            real(cond_best_vec) * 1e3,
            ':',
            color='g',
            label=r'Minimized. $\mu_R: %.0f$ cm$^2$ V$^{-1}$ s$^{-1}$' %
            (real(best_mob) * 1e4),
        )
        getattr(ax[0], plot_type)(
            n_vec * a0**2,
            -imag(cond_best_vec) * 1e3,
            '-.',
            color='g',
            label=r'Minimized. $\mu_I: %.0f$ cm$^2$ V$^{-1}$ s$^{-1}$' %
            (imag(best_mob) * 1e4),
        )
        """

    ax[0].axvline(
        x=exp_data[0, 0] / surf_area * a0**2,
        color='m',
        linestyle=':',
    )

    ax[0].axvline(
        x=exp_data[-1, 0] / surf_area * a0**2,
        color='m',
        linestyle=':',
    )

    ax[0].axhline(
        y=0,
        color='k',
        linestyle='-',
    )
    """
    lambda_th = sys.c_hbarc * sqrt(2 * pi * sys.beta / sys.m_p)
    ax[0].axvline(
        x=4 * a0**2 / lambda_th**2,
        color='g',
        label='Saha model limit',
    )
    """

    ax[0].set_xlim(1 / surf_area * a0**2, 60 / surf_area * a0**2)
    ax[0].set_ylim(-12.5, 5)

    x_vec_top = ax[0].xaxis.get_majorticklocs()[1:-1]
    x_vec_vals = (x_vec_top / a0**2)
    x_vec_vals = ['%.2f' % v for v in x_vec_vals]

    ax_top = ax[0].twiny()
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel(r'$n_\gamma$ (nm$^{-2}$)')

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'cond_fit_%s' % plot_type


def mobility_2d_integ(plot_type='loglog'):
    exp_power_data = loadtxt('extra/ef_power_spectrum.txt')

    w_vec = 2 * pi * exp_power_data[1:, 0]
    w_2_vec = linspace(w_vec[0], w_vec[-1], 2 * w_vec.size)
    power_norm_vec = exp_power_data[1:, 1] / simps(exp_power_data[1:, 1],
                                                   w_vec)

    w_peak = w_vec[power_norm_vec.argmax()]  # angular frequency, s^-1

    mu_dc_vec = array([795])  # cm^2 v^-1 s^-1

    # d = mu / beta / e
    diff_factor = 1e14 / sys.beta

    d_vec = mu_dc_vec * diff_factor  # nm^2 s^-1

    mob_vec, = (diffusion_cx(w_vec, Lx, Ly, d) / diff_factor for d in d_vec)

    mob = mob_integ_func(array([0.]), w_vec, power_norm_vec, mu_dc_vec, sys)

    print(mob)

    print('Re(μ): %.2f cm² / Vs, Im(μ): %.2f cm² / Vs' %
          (real(mob), imag(mob)))

    model_mob = 61.3515046681041 + 94.31802308693917j  # cm^2 v^-1 s^-1

    u_minzed = minimize(
        lambda u_vec: abs(
            mob_integ_func(u_vec, w_vec, power_norm_vec, mu_dc_vec, sys) -
            model_mob),
        (0.0),
        method='nelder-mead',
        options={
            'xtol': 1e-14,
        },
    )

    print('Minimized u values: %f' % tuple(u_minzed.x))

    mob_minzed = tuple(mu_dc_vec * exp(u_minzed.x))

    print('Minimized: μ: %.2f cm² / Vs' % mob_minzed)

    mu_minzed = mob_integ_func(u_minzed.x, w_vec, power_norm_vec, mu_dc_vec,
                               sys)

    print(mu_minzed)
    print('Minimized: Re(μ): %.2f cm² / Vs, Im(μ): %.2f cm² / Vs' %
          (real(mu_minzed), imag(mu_minzed)))

    ax[0].set_xlabel(r'$\omega$ $S$ $D^{-1}$')
    ax[0].set_ylabel(r'Mobility (cm$^2$ V$^{-1}$ s$^{-1}$)')

    mob_vec, = (diffusion_cx(w_2_vec, Lx, Ly, d) / diff_factor for d in d_vec)
    mob_minzed_vec = diffusion_cx(w_2_vec, Lx, Ly,
                                  diff_factor * mu_minzed[0]) / diff_factor

    w_factor = surf_area / sum(mu_dc_vec) / diff_factor
    x_vec = w_2_vec * w_factor
    x_minzed_vec = w_2_vec * w_factor

    getattr(ax[0], plot_type)(
        x_vec,
        real(mob_vec),
        'b--',
        label=r'$\mu_{R}(\omega)$',
    )
    getattr(ax[0], plot_type)(
        x_vec,
        imag(mob_vec),
        'b-',
        label=r'$\mu_{I}(\omega)$',
    )

    getattr(ax[0], plot_type)(
        x_minzed_vec,
        real(mob_minzed_vec),
        'g--',
        label=r'$\mu_{R}^{min}(\omega)$',
    )
    getattr(ax[0], plot_type)(
        x_minzed_vec,
        imag(mob_minzed_vec),
        'g-',
        label=r'$\mu_{I}^{min}(\omega)$',
    )

    ax[0].axvline(
        x=w_peak * w_factor,
        color='m',
        linestyle=':',
        label='$\omega_{peak}$',
    )
    ax[0].text(
        0.7,
        0.5,
        r"""$\mu_{dc}$ : %.0f cm$^2$ V$^{-1}$ s$^{-1}$
$\langle\mu_R\rangle$ : %.0f cm$^2$ V$^{-1}$ s$^{-1}$
$\langle\mu_I\rangle$ : %.0f cm$^2$ V$^{-1}$ s$^{-1}$

$\mu_{dc}^{min}$ : %.0f cm$^2$ V$^{-1}$ s$^{-1}$
$\langle\mu_R^{min}\rangle$ : %.0f cm$^2$ V$^{-1}$ s$^{-1}$
$\langle\mu_I^{min}\rangle$ : %.0f cm$^2$ V$^{-1}$ s$^{-1}$""" %
        (mu_dc_vec[0], real(mob), imag(mob), mob_minzed[0], real(mu_minzed),
         imag(mu_minzed)),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax[0].transAxes,
        bbox={
            'facecolor': 'none',
            'alpha': 0.5,
            'linewidth': 1,
            'antialiased': True,
            'edgecolor': (0.8, 0.8, 0.8)
        },
    )

    ax[0].set_xlim(x_vec[0], x_vec[-1])
    ax[0].legend(loc='lower right')

    fig.tight_layout()

    return 'mobility_2d_integ_%s' % plot_type


plots_list = [
    #('real_space_lwl_potential', 'linear'),
    #('real_space_mb_potential', 'linear'),
    #('real_space_mb_potential_density', 'linear'),
    #('energy_level_mb', 'linear'),
    #('energy_level_mb_density', 'log'),
    #('energy_level_mb_limit_density', 'log'),
    #('eb_lim_temperature', 'linear'),
    #('mu_e_lim_temperature', 'linear'),
    #('density_result', 'loglog'),
    ('eb_photo_density', 'semilogx'),
    #('cond_fit', 'plot'),
    #('mobility_2d_integ', 'loglog'),
]

for p, l in plots_list:
    print('Calling %s' % p)
    filename = locals()[p](l)

    plt.savefig('plots/papers/exciton1/%s.png' % filename)
    plt.savefig('plots/papers/exciton1/%s.pdf' % filename)
    plt.savefig('plots/papers/exciton1/%s.eps' % filename)

    plt.cla()
    #plt.clf()
    #plt.close()

from common import *
import statsmodels.api as sm

N_k = 1 << 12

fig_size = tuple(array([6.8, 5.3]) * 1)

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size, dpi=400)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]

Lx, Ly, Lz = 34.0, 9.6, 1.4  # nm
L_vec = array([Lx, Ly])
surf_area = Lx * Ly  # nm^2


def diffusion_cx(w_vec, L_vec, D):
    sqrt_factor_vec = sqrt(-1j * w_vec.reshape((-1, 1)) * L_vec.reshape(
        (1, -1))**2 / D)
    return D * (1.0 + 2.0 / L_vec.size *
                sum(tan(-0.5 * sqrt_factor_vec) / sqrt_factor_vec, axis=1))


def mob_integ_func(u_dc_vec, w_vec, power_norm_vec, mu_dc_bulk, sys):
    # d = mu / beta / e
    mu_dc_vec = mu_dc_bulk * exp(u_dc_vec)
    diff_factor = 1e14 / sys.beta

    d_vec = mu_dc_vec * diff_factor  # nm^2 s^-1

    mob_vec = array(
        [diffusion_cx(w_vec, L_vec, d) / diff_factor for d in d_vec])

    mob_norm_vec = mob_vec * power_norm_vec

    return simps(mob_norm_vec, w_vec, axis=1)


def real_space_lwl_potential(plot_type='linear'):
    ls_vec = logspace(log10(0.05 * sys.sys_ls), log10(0.9 * sys.sys_ls), 5)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, ls_vec.size)
    ]

    x_vec = linspace(1e-2, 8, 300) / sys.a0

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
    x_vec_vals = x_vec_top * sys.a0
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

    x_vec = linspace(1e-2, 8, 300) / sys.a0

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
    x_vec_vals = x_vec_top * sys.a0
    x_vec_vals = ['%.1f' % v for v in x_vec_vals]

    ax_top = ax[0].twiny()
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel('$r$ (nm)')

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'real_space_mb_potential_density'


def energy_level_mb(plot_type='plot'):
    file_id = 'aneuiPMlRLy4x8FlcAajaA'
    load_data('extra/mu_e_data_%s' % file_id, globals())

    T_vec = linspace(100, 400, 4)
    mu_e_vec = linspace(-0.12, 0.0, 18)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, T_vec.size)
    ]

    ax[0].set_xlabel(r'$\mu_e$ (meV)')
    ax[0].set_ylabel(r'$\epsilon(\mu_e)$ (meV)')

    sys = system_data(m_e, m_h, eps_r, 300)

    ax[0].axhline(
        y=sys.get_E_n(0.5) * 1e3,
        color='r',
        linestyle='--',
        label='Coulomb limit',
    )

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        sys = system_data(m_e, m_h, eps_r, T)

        y_vec = array(time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys))
        getattr(ax[0], plot_type)(
            mu_e_vec * 1e3,
            y_vec * 1e3,
            '-',
            color=c,
            label='T: %.0f K' % sys.T,
        )

    z_sys_lwl = time_func(plasmon_det_zero_lwl, N_k, sys.sys_ls, sys)

    ax[0].set_xlim(mu_e_vec[0] * 1e3, mu_e_vec[-1] * 1e3)
    ax[0].axhline(
        y=z_sys_lwl * 1e3,
        color='k',
        linestyle='--',
        label='$\epsilon(\lambda_{s,0}) = %0.2f$ eV' % z_sys_lwl,
    )

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'energy_level_mb_%s' % plot_type


def energy_level_mb_density(plot_type='log'):
    file_id = 'pa2MgrnmTKKg4u_gz3WY8Q'
    values_list = load_data('extra/eb_values_temp_%s' % file_id, globals())

    #values_list = []

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, len(T_vec))
    ]

    ax[0].set_xlabel(r'$\langle N_e \rangle$')
    ax[0].set_ylabel(r'$\epsilon$ (meV)')

    sys = system_data(m_e, m_h, eps_r, 300)

    z_cou_lwl = time_func(plasmon_det_zero_lwl, N_k, 1e-8, sys, -1e-3)
    ax[0].axhline(
        y=z_cou_lwl * 1e3,
        color='r',
        linestyle='--',
        label=r'$\langle N_e\rangle$: %d' % 0,
    )

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        sys = system_data(m_e, m_h, eps_r, T)
        #mu_e_vec = linspace(-6.0, 0.0, 32) / sys.beta
        mu_e_vec = values_list[2 * i]
        y_vec = values_list[2 * i + 1]

        mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
        n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])

        #y_vec = array(time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys))

        #values_list.append(mu_e_vec[:])
        #values_list.append(y_vec[:])

        getattr(ax[0], plot_type)(
            n_id_vec * surf_area,
            y_vec * 1e3,
            '-',
            color=c,
            label='T: %.0f K' % sys.T,
        )
        getattr(ax[0], plot_type)(
            n_id_vec[-1] * surf_area,
            y_vec[-1] * 1e3,
            'o',
            color=c,
        )
    """
    save_data(
        'extra/eb_values_temp_%s' %
        base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2],
        values_list,
        {
            'm_e': m_e,
            'm_h': m_h,
            'T_vec': T_vec.tolist(),
            'eps_r': eps_r
        },
    )
    """

    ax[0].set_xlim(1e-4 * surf_area, 2.1e-2 * surf_area)
    """
    z_sys_lwl = time_func(plasmon_det_zero_lwl, N_k, sys.sys_ls, sys, -1e-3)

    ax[0].axhline(
        y=z_sys_lwl * 1e3,
        color='k',
        linestyle='--',
        label=r'$\langle N_e\rangle\rightarrow\infty$',
    )
    """

    x_vec_top = ax[0].xaxis.get_majorticklocs()[2:-2]
    x_vec_vals = (x_vec_top / surf_area) * 1e3
    #x_vec_vals = [('%%.%df' % (2 + min(0, -log10(v)))) % v for v in x_vec_vals]
    x_vec_vals = ['%.2f' % v for v in x_vec_vals]

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


def density_result(plot_type='loglog'):
    #file_id = 'aneuiPMlRLy4x8FlcAajaA'
    #file_id = '7hNKZBFHQbGL6xOf9r_w2Q' # lower massses
    file_id = '9xk12W--Tl6efYR-K76hoQ'  # higher masses

    n_exp_vec, cond_real, cond_imag, N_a_exp_vec, cond_err_real, cond_err_imag = load_data(
        'bin/cdse_platelet_data')

    n_vec, exc_list, eb_vec = load_data('extra/mu_e_data_%s' % file_id,
                                        globals())
    n_vec, eb_vec = n_vec[2:], eb_vec[2:]
    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    sys = system_data(m_e, m_h, eps_r, T)

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, 1)
    ]

    ax[0].set_xlabel(r'$n_\gamma a_0^2$')
    ax[0].set_ylabel(r'Average Number of Particles per Bohr Area')

    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        n_id_vec * sys.a0**2,
        '--',
        color=colors[0],
        label=r'Our model: $n_e a_0^2$',
    )
    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        n_exc_vec * sys.a0**2,
        '-',
        color=colors[0],
        label=r'Our model: $n_{exc} a_0^2$',
    )

    ax[0].axvline(
        x=n_exp_vec[0] / surf_area * sys.a0**2,
        color='m',
        linestyle=':',
    )

    ax[0].axvline(
        x=n_exp_vec[-1] / surf_area * sys.a0**2,
        color='m',
        linestyle=':',
    )

    ax[0].set_xlim(n_vec[0] * sys.a0**2, n_vec[-1] * sys.a0**2)

    lambda_th = sys.c_hbarc * sqrt(2 * pi * sys.beta / sys.m_p)
    ax[0].axvline(
        x=4 * sys.a0**2 / lambda_th**2,
        color='g',
        label='Saha model limit',
    )

    x_vec_top = ax[0].xaxis.get_majorticklocs()[2:-2]
    x_vec_vals = x_vec_top / sys.a0**2
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


def eb_photo_density(plot_type='semilogx'):
    #file_id = '7hNKZBFHQbGL6xOf9r_w2Q'
    file_id = '9xk12W--Tl6efYR-K76hoQ'  # higher masses

    n_exp_vec, cond_real, cond_imag, N_a_exp_vec, cond_err_real, cond_err_imag = load_data(
        'bin/cdse_platelet_data')

    n_vec, exc_list, eb_vec = load_data('extra/mu_e_data_%s' % file_id,
                                        globals())
    n_vec, eb_vec = n_vec[2:], eb_vec[2:]
    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    sys = system_data(m_e, m_h, eps_r, T)

    print(sys.a0)


    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, 1)
    ]

    ax[0].set_xlabel(r'$n_\gamma a_0^2$')
    ax[0].set_ylabel(r'$E_B$ (meV)')

    z_cou_lwl = time_func(plasmon_det_zero_lwl, 1 << 10, 1e-8, sys, -1e-3)
    ax[0].axhline(
        y=z_cou_lwl * 1e3,
        color='r',
        linestyle='--',
        label=r'$\langle N_e\rangle$: %d' % 0,
    )

    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        eb_vec * 1e3,
        '-',
        color=colors[0],
        label=r'T: %.0f K' % T,
    )

    ax[0].axvline(
        x=n_exp_vec[0] / surf_area * sys.a0**2,
        color='m',
        linestyle=':',
    )

    ax[0].axvline(
        x=n_exp_vec[-1] / surf_area * sys.a0**2,
        color='m',
        linestyle=':',
    )

    ax[0].set_xlim(n_vec[0] * sys.a0**2, n_vec[-1] * sys.a0**2)
    ax[0].set_ylim(-193.5, None)

    x_vec_top = ax[0].xaxis.get_majorticklocs()[2:-2]
    x_vec_vals = x_vec_top / sys.a0**2
    x_vec_vals = ['$10^{%.0f}$' % log10(v) for v in x_vec_vals]

    y_vec_left = linspace(-193, eb_vec[-1] * 1e3, 5)
    y_vec_vals = ['%d' % v for v in y_vec_left]
    ax[0].set_yticks(y_vec_left)
    ax[0].set_yticklabels(y_vec_vals)

    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel(r'$n_\gamma$ (nm$^{-2}$)')

    th_pol_vec = 21 / 2**8 * 16 * sys.c_aEM**2 * (
        sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / abs(eb_vec)**3

    axins = ax[0].inset_axes([0.583, 0.11, 0.4, 0.36])
    axins.set_ylabel(r'$\alpha$ ($10^{-36}$ cm$^2$ / V)')

    axins.axvline(
        x=n_exp_vec[0] / surf_area * sys.a0**2,
        color='m',
        linestyle=':',
    )

    axins.axvline(
        x=n_exp_vec[-1] / surf_area * sys.a0**2,
        color='m',
        linestyle=':',
    )

    axins.set_xlim(n_vec[0] * sys.a0**2, n_vec[-1] * sys.a0**2)

    getattr(axins, plot_type)(
        n_vec * sys.a0**2,
        th_pol_vec * 1e36,
        '-',
        color=colors[0],
        label=r'T: %.0f K' % T,
    )

    ax[0].legend(loc='upper left')

    fig.tight_layout()

    return 'eb_photo_density_%s' % plot_type


def cond_fit(plot_type='plot'):
    file_id = '9xk12W--Tl6efYR-K76hoQ'
    fit_file_id = 'imDDS1DJRciMz_-rSvA1RQ'

    n_exp_vec, cond_real, cond_imag, N_a_exp_vec, cond_err_real, cond_err_imag = load_data(
        'bin/cdse_platelet_data')

    cond_factor = 3 / 2
    cond_factor_mob = 1
    cond_factor_pol = cond_factor_mob

    for i in [cond_real, cond_imag, cond_err_real, cond_err_imag]:
        i *= cond_factor

    n_vec, exc_list, eb_vec = load_data('extra/mu_e_data_%s' % file_id,
                                        globals())
    n_vec, eb_vec = n_vec[2:], eb_vec[2:]
    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    sys = system_data(m_e, m_h, eps_r, T)

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    n_fit_vec, exc_fit_list, eb_fit_vec = load_data('extra/mu_e_data_%s' %
                                                    fit_file_id)
    n_fit_vec, eb_fit_vec = n_fit_vec[2:], eb_fit_vec[2:]
    mu_e_lim_fit, eb_lim_fit = exc_fit_list[:2]
    mu_e_fit_vec = array(exc_fit_list[2:])

    mu_h_fit_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_fit_vec])
    n_id_fit_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_fit_vec])
    n_exc_fit_vec = n_fit_vec - n_id_fit_vec

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, 1)
    ]

    L, mob_R, mob_I, pol, freq = 2e-3, 54e-4, 7e-4, 3.1e-36, 0.6e12
    p_Na_vec = N_a_exp_vec * 1e4
    Na_vec = n_vec * p_Na_vec[0] / (n_exp_vec[0] / surf_area)

    th_pol = 21 / 2**8 * 16 * sys.c_aEM**2 * (sys.c_hbarc * 1e-9 /
                                              eps_r)**2 * sys.c_e_charge / abs(
                                                  sys.get_E_n(0.5))**3

    p_th_pol_vec = 21 / 2**8 * 16 * sys.c_aEM**2 * (
        sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / abs(eb_fit_vec)**3

    q_yield_fit_vec = n_id_fit_vec / n_fit_vec
    q_yield_exc_fit_vec = n_exc_fit_vec / n_fit_vec

    real_x = (sys.c_e_charge / L * p_Na_vec * q_yield_fit_vec *
              cond_factor_mob).reshape(-1, 1)
    real_y = cond_real

    real_model = sm.WLS(real_y,
                        real_x,
                        weights=1 / cond_err_real,
                        has_const=False)
    real_fit = real_model.fit(use_t=True)

    imag_x = (sys.c_e_charge * p_Na_vec / L * q_yield_fit_vec *
              cond_factor_mob).reshape(-1, 1)
    imag_y = cond_imag - p_Na_vec / L * freq * 2 * pi * q_yield_exc_fit_vec * p_th_pol_vec * cond_factor_pol

    imag_model = sm.WLS(
        imag_y,
        imag_x,
        weights=1 / cond_err_imag,
        has_const=False,
    )
    imag_fit = imag_model.fit(use_t=True)

    fit_mob_R, = real_fit.params
    fit_mob_I, = imag_fit.params

    err_mob_R, err_mob_I = real_fit.bse, imag_fit.bse

    print(complex(fit_mob_R, fit_mob_I) * 1e4)

    #best_mob = (71.33360959613807+99.31278244259349j) * 1e-4
    best_mob = (84.89884997035625 + 108.56680139651795j) * 1e-4
    #best_mob = (34.912722134765346 + 18.789772887540792j) * 1e-4

    print('mob_R: %f±%1.0e, mob_I: %e±%1.0e' %
          (fit_mob_R * 1e4, err_mob_R * 1e4, fit_mob_I * 1e4, err_mob_I * 1e4))

    ax[0].set_xlabel(r'$n_\gamma a_0^2$')
    ax[0].set_ylabel(r'$\Delta\sigma$ ($10^{-3}$ S m$^{-1}$)')

    getattr(ax[0], plot_type)(
        n_exp_vec / surf_area * sys.a0**2,
        cond_real * 1e3,
        'o',
        color='k',
    )

    getattr(ax[0], plot_type)(
        n_exp_vec / surf_area * sys.a0**2,
        -cond_imag * 1e3,
        '^',
        color='k',
    )

    ax[0].errorbar(n_exp_vec / surf_area * sys.a0**2,
                   cond_real * 1e3,
                   yerr=cond_err_real * 1e3,
                   fmt='none',
                   capsize=5,
                   color='k')

    ax[0].errorbar(n_exp_vec / surf_area * sys.a0**2,
                   -cond_imag * 1e3,
                   yerr=cond_err_imag * 1e3,
                   fmt='none',
                   capsize=5,
                   color='k')

    th_pol_vec = 21 / 2**8 * 16 * sys.c_aEM**2 * (
        sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / abs(eb_vec)**3

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    q_yield_vec = n_id_vec / n_vec
    q_yield_exc_vec = n_exc_vec / n_vec

    cond_vec = (Na_vec * sys.c_e_charge / L * fit_mob_R *
                q_yield_vec) * cond_factor_mob + 1j * (
                    Na_vec * sys.c_e_charge / L * fit_mob_I * q_yield_vec +
                    2 * pi * th_pol_vec * freq * Na_vec / L *
                    q_yield_exc_vec) * cond_factor_pol

    cond_best_vec = (Na_vec * sys.c_e_charge / L * real(best_mob) *
                     q_yield_vec) * cond_factor_mob + 1j * (
                         Na_vec * sys.c_e_charge / L * imag(best_mob) *
                         q_yield_vec + 2 * pi * th_pol_vec * freq * Na_vec /
                         L * q_yield_exc_vec) * cond_factor_pol

    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        real(cond_vec) * 1e3,
        '--',
        color=colors[0],
        label=r'Fit. $\mu_R: %.0f±%d$ cm$^2$ V$^{-1}$ s$^{-1}$' %
        (fit_mob_R * 1e4, err_mob_R * 1e4),
    )
    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        -imag(cond_vec) * 1e3,
        '-',
        color=colors[0],
        label=r'Fit. $\mu_I: %.0f±%d$ cm$^2$ V$^{-1}$ s$^{-1}$' %
        (fit_mob_I * 1e4, err_mob_I * 1e4),
    )

    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        real(cond_best_vec) * 1e3,
        '--',
        color='b',
        label=r'Diffusion. $\mu_R: %.0f$ cm$^2$ V$^{-1}$ s$^{-1}$' %
        (real(best_mob) * 1e4),
    )
    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        -imag(cond_best_vec) * 1e3,
        '-',
        color='b',
        label=r'Diffusion. $\mu_I: %.0f$ cm$^2$ V$^{-1}$ s$^{-1}$' %
        (imag(best_mob) * 1e4),
    )

    ax[0].axvline(
        x=n_exp_vec[0] / surf_area * sys.a0**2,
        color='m',
        linestyle=':',
    )

    ax[0].axvline(
        x=n_exp_vec[-1] / surf_area * sys.a0**2,
        color='m',
        linestyle=':',
    )

    ax[0].axhline(
        y=0,
        color='k',
        linestyle='-',
    )

    lambda_th = sys.c_hbarc * sqrt(2 * pi * sys.beta / sys.m_p)
    ax[0].axvline(
        x=4 * sys.a0**2 / lambda_th**2,
        color='g',
        label='Saha model limit',
    )

    ax[0].set_xlim(1 / surf_area * sys.a0**2, 60 / surf_area * sys.a0**2)
    ax[0].set_ylim(-cond_factor * 12.5, cond_factor * 5)

    x_vec_top = ax[0].xaxis.get_majorticklocs()[1:-1]
    x_vec_vals = (x_vec_top / sys.a0**2)
    x_vec_vals = ['%.2f' % v for v in x_vec_vals]

    ax_top = ax[0].twiny()
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel(r'$n_\gamma$ (nm$^{-2}$)')

    ax[0].legend(loc='lower left')

    fig.tight_layout()

    return 'cond_fit_%s' % plot_type


def mobility_2d_integ(plot_type='loglog'):
    exp_power_data = loadtxt('extra/ef_power_spectrum.txt')
    fit_file_id = 'imDDS1DJRciMz_-rSvA1RQ'
    load_data('extra/mu_e_data_%s' % fit_file_id, globals())

    sys = system_data(m_e, m_h, eps_r, T_vec[0])

    w_vec = 2 * pi * exp_power_data[1:, 0]
    w_2_vec = linspace(w_vec[0], w_vec[-1], 2 * w_vec.size)
    power_norm_vec = exp_power_data[1:, 1] / simps(exp_power_data[1:, 1],
                                                   w_vec)

    w_peak = w_vec[power_norm_vec.argmax()]  # angular frequency, s^-1

    mu_dc_vec = array([600, 20])  # cm^2 v^-1 s^-1

    # d = mu / beta / e
    diff_factor = 1e14 / sys.beta

    d_vec = mu_dc_vec * diff_factor  # nm^2 s^-1

    mob_vec = (diffusion_cx(w_vec, L_vec, d) / diff_factor for d in d_vec)

    mob = sum(
        mob_integ_func(zeros_like(mu_dc_vec), w_vec, power_norm_vec, mu_dc_vec,
                       sys))

    print(mob)

    print('Re(μ): %.2f cm² / Vs, Im(μ): %.2f cm² / Vs' %
          (real(mob), imag(mob)))

    model_mob = 64.49722036113633 + 149.26761240112566j
    #model_mob = 42.998146907424214 + 75.78389097812033j
    #model_mob = 28.665431271616136 + 26.794743362783418j

    u_minzed = minimize(
        lambda u_vec: abs(
            sum(mob_integ_func(u_vec, w_vec, power_norm_vec, mu_dc_vec, sys)) -
            model_mob),
        zeros_like(mu_dc_vec),
        method='nelder-mead',
        options={
            'xtol': 1e-14,
        },
    )

    print('Minimized u values: %s' % u_minzed.x)

    mob_minzed = mu_dc_vec * exp(u_minzed.x)

    print('Minimized: μ: %s cm² / Vs' % mob_minzed)

    mu_minzed = sum(
        mob_integ_func(u_minzed.x, w_vec, power_norm_vec, mu_dc_vec, sys))

    print(mu_minzed)
    print('Minimized: Re(μ): %.2f cm² / Vs, Im(μ): %.2f cm² / Vs' %
          (real(mu_minzed), imag(mu_minzed)))

    ax[0].set_xlabel(r'$\omega$ $S$ $D^{-1}$')
    ax[0].set_ylabel(r'Mobility (cm$^2$ V$^{-1}$ s$^{-1}$)')

    mob_vec = sum(
        [diffusion_cx(w_2_vec, L_vec, d) / diff_factor for d in d_vec], axis=0)
    mob_minzed_vec = diffusion_cx(w_2_vec, L_vec,
                                  diff_factor * sum(mu_minzed)) / diff_factor

    w_factor = surf_area / sum(mu_dc_vec) / diff_factor
    w_minzed_factor = surf_area / sum(mob_minzed) / diff_factor
    x_vec = w_2_vec  # * w_factor
    x_minzed_vec = w_2_vec  # * w_minzed_factor

    getattr(ax[0], plot_type)(
        x_vec,
        real(mob_vec),
        'g--',
        label=r'$\mu_{R}(\omega)$',
    )
    getattr(ax[0], plot_type)(
        x_vec,
        imag(mob_vec),
        'g-',
        label=r'$\mu_{I}(\omega)$',
    )

    getattr(ax[0], plot_type)(
        x_minzed_vec,
        real(mob_minzed_vec),
        'b--',
        label=r'$\mu_{R}^{min}(\omega)$',
    )
    getattr(ax[0], plot_type)(
        x_minzed_vec,
        imag(mob_minzed_vec),
        'b-',
        label=r'$\mu_{I}^{min}(\omega)$',
    )

    ax[0].axvline(
        x=w_peak,  # * w_factor,
        color='m',
        linestyle=':',
        label='$\omega_{peak}$',
    )
    ax[0].text(
        0.68,
        0.5,
        r"""$\mu_{dc}$ : %.0f cm$^2$ V$^{-1}$ s$^{-1}$
$\langle\mu_R\rangle$ : %.0f cm$^2$ V$^{-1}$ s$^{-1}$
$\langle\mu_I\rangle$ : %.0f cm$^2$ V$^{-1}$ s$^{-1}$

$\mu_{dc}^{min}$ : %.0f cm$^2$ V$^{-1}$ s$^{-1}$
$\langle\mu_R^{min}\rangle$ : %.0f cm$^2$ V$^{-1}$ s$^{-1}$
$\langle\mu_I^{min}\rangle$ : %.0f cm$^2$ V$^{-1}$ s$^{-1}$""" %
        (sum(mu_dc_vec), real(mob), imag(mob), sum(mob_minzed),
         real(mu_minzed), imag(mu_minzed)),
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

    ax[0].set_xlim(min(x_vec[0], x_minzed_vec[0]),
                   max(x_vec[-1], x_minzed_vec[-1]))
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
    #('eb_photo_density', 'semilogx'),
    #('cond_fit', 'plot'),
    #('mobility_2d_integ', 'loglog'),
]

plots_list = [pysys.argv[1:]]

for p, l in plots_list:
    print('Calling %s' % p)
    filename = locals()[p](l)

    plt.savefig('plots/papers/exciton1/%s.png' % filename)
    plt.savefig('plots/papers/exciton1/%s.pdf' % filename)
    plt.savefig('plots/papers/exciton1/%s.eps' % filename)

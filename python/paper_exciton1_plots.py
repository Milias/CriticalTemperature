from common import *
import statsmodels.api as sm

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

from matplotlib.legend_handler import HandlerBase


class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height,
                       fontsize, trans):
        l1 = plt.Line2D([x0, y0 + width], [0.7 * height, 0.7 * height],
                        linestyle=orig_handle[1],
                        color=orig_handle[0])
        l2 = plt.Line2D([x0, y0 + width], [0.3 * height, 0.3 * height],
                        color=orig_handle[0])
        return [l1, l2]


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
    #d              = mu / beta / e
    mu_dc_vec = mu_dc_bulk * exp(u_dc_vec)
    diff_factor = 1e14 / sys.beta

    d_vec = mu_dc_vec * diff_factor  # nm^2 s^-1

    return array([
        diffusion_cx(array([w_vec[power_norm_vec.argmax()]]), L_vec, d) /
        diff_factor for d in d_vec
    ])

    mob_vec = array(
        [diffusion_cx(w_vec, L_vec, d) / diff_factor for d in d_vec])

    return simps(mob_vec * power_norm_vec, w_vec, axis=1)


def cond_from_mob(mob, w_mean, Na_vec, L, q_yield_vec, eb_vec, eps_r, sys):
    th_pol_vec = 21 / 2**8 * 16 * sys.c_aEM**2 * (
        sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / abs(eb_vec)**3

    cond_mob_vec = Na_vec * sys.c_e_charge * mob * 1e-4 * q_yield_vec / L
    cond_pol_vec = Na_vec * th_pol_vec * (1 - q_yield_vec) / L * w_mean

    return cond_mob_vec + 1j * cond_pol_vec


def cond_from_diff(u_dc_vec, w_vec, power_norm_vec, w_mean, mu_dc_bulk, Na_vec,
                   L, q_yield_vec, eb_vec, eps_r, sys):
    mob = mob_integ_func(
        u_dc_vec,
        w_vec,
        power_norm_vec,
        mu_dc_bulk,
        sys,
    )

    return cond_from_mob(
        sum(mob),
        w_mean,
        Na_vec,
        L,
        q_yield_vec,
        eb_vec,
        eps_r,
        sys,
    )


def scr_length_density(plot_type='semilogx'):
    file_id = 'ohmBtM4fTgiypsA8GPlMpQ'
    load_data('extra/eb_vals_temp_%s' % file_id, globals())

    sys = system_data(m_e, m_h, eps_r, 294)

    n_vec = logspace(-1.3, 2.3, 100) / surf_area
    T_vec = linspace(130, 350, 5)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, len(T_vec))
    ]

    ax[0].set_xlabel(r'$\langle N_q \rangle$')
    ax[0].set_ylabel(r'$\lambda_{s}$ (nm)')

    x_vec = n_vec * surf_area

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        sys = system_data(m_e, m_h, eps_r, T)
        y_vec = sys.sys_ls * (
            1 - sys.m_ph * exp(-sys.m_pe * pi * sys.c_hbarc**2 / sys.m_p *
                               sys.beta * n_vec) - sys.m_pe *
            exp(-sys.m_ph * pi * sys.c_hbarc**2 / sys.m_p * sys.beta * n_vec))

        getattr(ax[0], plot_type)(
            x_vec,
            1 / y_vec,
            '-',
            color=c,
            label=r'$T$: %.0f K' % T,
        )

    ax[0].set_xlim(x_vec[0], x_vec[-1])
    #ax[0].set_ylim(0, None)
    """
    ax[0].axhline(
        y=1 / sys.sys_ls,
        linestyle='--',
        color='k',
#label           = r '$\lambda_{s,0}$',
    )
    ax[0].set_yticks([1 / sys.sys_ls, 1, 10, Lx])
    ax[0].set_yticklabels([r'$\lambda_{s,0}$', '$1$', '$10$', r'$L_x$'])
    """
    ax[0].set_yticks([1, 10, Lx])
    ax[0].set_yticklabels(['$1$', '$10$', r'$L_x$'])

    x_top_factor = 3
    x_vec_top = ax[0].xaxis.get_majorticklocs()[2:-2]
    x_vec_vals = x_vec_top / surf_area * 10**x_top_factor
    #x_vec_vals      = [r '$10^{%.0f}$' % log10(v) for v in x_vec_vals]
    x_vec_vals = [r'%.1f' % v for v in x_vec_vals]

    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel('$n_q$ (10$^{-%.0f}$ nm$^{-2}$)' % x_top_factor)

    y_vec_right = [1 / sys.sys_ls]
    y_vec_right_labels = [r'$\lambda_{s,0}$']

    ax_right = ax[0].twinx()
    ax_right.set_yscale('log')
    ax_right.set_ylim(ax[0].get_ylim())
    ax_right.set_yticks(y_vec_right)
    ax_right.set_yticklabels(y_vec_right_labels)

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'scr_length_density_%s' % plot_type


def real_space_lwl_potential(plot_type='plot'):
    file_id = 'imDDS1DJRciMz_-rSvA1RQ'
    load_data('extra/mu_e_data_%s' % file_id, globals())

    sys = system_data(m_e, m_h, eps_r, T_vec[0])

    print(1 / sys.sys_ls)

    ls_vec = logspace(log10(0.01 * sys.sys_ls), log10(0.5 * sys.sys_ls), 5)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, ls_vec.size)
    ]

    x_vec = linspace(0.3, 12, 100) / sys.a0

    ax[0].set_xlabel(r'$r$ $a_0^{-1}$')
    ax[0].set_ylabel(r'$V_{sc}(r;\lambda_s)$ (meV)')

    cou_vec = array(time_func(plasmon_rpot_lwl_v, x_vec * sys.a0, 1e-8,
                              sys)) * 1e3
    getattr(ax[0], plot_type)(
        x_vec,
        cou_vec,
        '--',
        color='r',
        label=r'$\lambda_s \rightarrow \infty$',
    )

    for c, (i, ls) in zip(colors, enumerate(ls_vec)):
        y_vec = array(time_func(plasmon_rpot_lwl_v, x_vec * sys.a0, ls,
                                sys)) * 1e3
        y_dot_vec = array(time_func(plasmon_rpot_lwl_v, [1 / ls], ls,
                                    sys)) * 1e3

        getattr(ax[0], plot_type)(
            x_vec,
            y_vec,
            '-',
            color=c,
            label='$\lambda_s$: $%.1f$ nm' % (1 / ls),
        )

        if 1 / ls / sys.a0 < x_vec[-1]:
            getattr(ax[0], plot_type)(
                [1 / ls / sys.a0],
                y_dot_vec,
                'o',
                color=c,
            )

    sys_vec = array(
        time_func(plasmon_rpot_lwl_v, x_vec * sys.a0, sys.sys_ls, sys)) * 1e3
    sys_dot_vec = array(
        time_func(plasmon_rpot_lwl_v, [1 / sys.sys_ls], sys.sys_ls, sys)) * 1e3
    getattr(ax[0], plot_type)(
        x_vec,
        sys_vec,
        '--',
        color='k',
        label='$\lambda_{s}$: $\lambda_{s,0}$',
    )
    getattr(ax[0], plot_type)(
        [1 / sys.sys_ls / sys.a0],
        sys_dot_vec,
        'o',
        color='k',
    )

    ax[0].set_ylim(-0.12e3, 0.005e3)
    ax[0].set_xlim(0, x_vec[-1])
    ax[0].axhline(y=0, color='k')

    ax[0].set_yticks(linspace(-0.1e3, 0.0, 5))
    ax[0].set_yticklabels(
        ['$%.0f$' % v for v in ax[0].yaxis.get_majorticklocs()])

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

    return 'real_space_lwl_potential_%s' % plot_type


def real_space_mb_potential_density(plot_type='plot'):
    file_id = 'imDDS1DJRciMz_-rSvA1RQ'
    load_data('extra/mu_e_data_%s' % file_id, globals())

    sys = system_data(m_e, m_h, eps_r, T_vec[0])

    mu_e_vec = -logspace(log10(0.07), -2.4, 5)
    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])

    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, mu_e_vec.size)
    ]

    x_vec = linspace(0.3, 12, 100) / sys.a0

    #ax[0].set_title('Real space potential\nClassical Limit')
    ax[0].set_xlabel(r'$r$ $a_0^{-1}$')
    ax[0].set_ylabel(r'$V_{sc}(r;n_q)$ (meV)')

    cou_vec = array(time_func(plasmon_rpot_lwl_v, x_vec * sys.a0, 1e-8,
                              sys)) * 1e3
    getattr(ax[0], plot_type)(
        x_vec,
        cou_vec,
        '--',
        color='r',
        label=r'$\langle N_q\rangle$: %d' % 0,
    )

    for c, (i, (mu_e, mu_h,
                n_id)) in zip(colors,
                              enumerate(zip(mu_e_vec, mu_h_vec, n_id_vec))):

        num_e = n_id * surf_area
        y_vec = array(
            time_func(plasmon_rpot_ht_v, x_vec * sys.a0, mu_e, mu_h,
                      sys)) * 1e3
        getattr(ax[0], plot_type)(
            x_vec,
            y_vec,
            '-',
            color=c,
            label=r'$\langle N_q\rangle$: %.1f' % num_e,
        )

    sys_vec = array(
        time_func(plasmon_rpot_lwl_v, x_vec * sys.a0, sys.sys_ls, sys)) * 1e3
    getattr(ax[0], plot_type)(
        x_vec,
        sys_vec,
        '--',
        color='k',
        label=r'$\langle N_q\rangle\rightarrow\infty$',
    )

    ax[0].set_ylim(-0.12e3, 0.005e3)
    ax[0].set_xlim(0, x_vec[-1])
    ax[0].axhline(y=0, color='k')

    ax[0].set_yticks(linspace(-0.1e3, 0.0, 5))
    ax[0].set_yticklabels(
        ['$%.0f$' % v for v in ax[0].yaxis.get_majorticklocs()])

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

    return 'real_space_mb_potential_density_%s' % plot_type


def energy_level_mb_density(plot_type='semilogx'):
    file_id = 'ohmBtM4fTgiypsA8GPlMpQ'
    values_list = load_data('extra/eb_vals_temp_%s' % file_id, globals())

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, len(T_vec))
    ]

    ax[0].set_xlabel(r'$\langle N_q \rangle$')
    ax[0].set_ylabel(r'$E_B$ (meV)')

    sys = system_data(m_e, m_h, eps_r, T_vec[0])
    ax[0].axhline(
        y=z_cou_lwl * 1e3,
        color='r',
        linestyle='--',
        label=r'$\langle N_q\rangle$: %d' % 0,
    )

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        sys = system_data(m_e, m_h, eps_r, T)
        mu_e_vec = values_list[2 * i]
        y_vec = values_list[2 * i + 1]

        mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
        n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])

        getattr(ax[0], plot_type)(
            n_id_vec * surf_area,
            y_vec * 1e3,
            '-',
            color=c,
            label='$T$: %.0f K' % sys.T,
        )

        getattr(ax[0], plot_type)(
            n_id_vec[-1] * surf_area,
            y_vec[-1] * 1e3,
            'o',
            color=c,
        )

    ax[0].set_xlim(1e-4 * surf_area, 2.1e-2 * surf_area)
    ax[0].set_ylim(-195, -118)
    """
    z_sys_lwl = time_func(plasmon_det_zero_lwl, N_k, sys.sys_ls, sys, -1e-3)

    ax[0].axhline(
        y=z_sys_lwl * 1e3,
        color='k',
        linestyle='--',
        label=r'$\langle N_e\rangle\rightarrow\infty$',
    )
    """

    y_vec_left = [-193] + (linspace(-0.18, -0.12, 5) * 1e3).tolist()
    y_vec_left_vals = ['$%.0f$' % v for v in y_vec_left]
    ax[0].set_yticks(y_vec_left)
    ax[0].set_yticklabels(y_vec_left_vals)

    x_vec_top = ax[0].xaxis.get_majorticklocs()[2:-2]
    x_vec_vals = (x_vec_top / surf_area) * 1e3
    #x_vec_vals    = [('%%.%df' % (2 + min(0, -log10(v)))) % v for v in x_vec_vals]
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

    return 'energy_level_mb_density_%s' % plot_type


def energy_level_mb_limit_density(plot_type='semilogx'):
    file_id = 'ohmBtM4fTgiypsA8GPlMpQ'
    values_list = load_data('extra/eb_vals_temp_%s' % file_id, globals())

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, len(T_vec))
    ]

    ax[0].set_xlabel(r'$\langle N_e \rangle$')
    ax[0].set_ylabel(r'$E_B$ (meV)')

    sys = system_data(m_e, m_h, eps_r, T_vec[0])
    ax[0].axhline(
        y=z_cou_lwl * 1e3,
        color='r',
        linestyle='--',
        label=r'$\langle N_e\rangle$: %d' % 0,
    )

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        sys = system_data(m_e, m_h, eps_r, T)
        mu_e_vec = values_list[2 * i]
        y_vec = values_list[2 * i + 1]

        mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
        n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
        """
        mu_e_lim, eb_lim = plasmon_exc_mu_lim(N_k, sys)
        n_id_lim = sys.density_ideal(mu_e_lim)

        line_vec = (mu_e_vec + mu_h_vec) * 1e3

        getattr(ax[0], plot_type)(
            n_id_vec * surf_area,
            line_vec,
            ':',
            color=c,
        )

        getattr(ax[0], plot_type)(
            [n_id_lim * surf_area],
            [eb_lim * 1e3],
            's',
            color=c,
        )
        """

        getattr(ax[0], plot_type)(
            n_id_vec * surf_area,
            y_vec,
            '-',
            color=c,
            label='$T$: %.0f K' % sys.T,
        )

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

    return 'energy_level_mb_limit_density_%s' % plot_type


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
        #label       = '$\mu_{e,0}(T)$',
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
    #file_id                    = 'aneuiPMlRLy4x8FlcAajaA'
    #file_id                    = '7hNKZBFHQbGL6xOf9r_w2Q' #lower massses
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

    n_q0 = sys.density_ideal(mu_e_lim)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, 1)
    ]

    ax[0].set_xlabel(r'$n_\gamma a_0^2$')
    ax[0].set_ylabel('Average Number of Particles\n per Bohr Area')

    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        n_id_vec * sys.a0**2,
        '--',
        color=colors[0],
        label=r'$n_q a_0^2$',
    )
    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        n_exc_vec * sys.a0**2,
        '-',
        color=colors[0],
        label=r'$n_{exc} a_0^2$',
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
        label='Quantum limit',
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

    y_vec_right = [n_q0 * sys.a0**2]
    y_vec_right_labels = [r'$n_{q,0}$']

    ax_right = ax[0].twinx()
    ax_right.set_yscale('log')
    ax_right.set_ylim(ax[0].get_ylim())
    ax_right.set_yticks(y_vec_right)
    ax_right.set_yticklabels(y_vec_right_labels)

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'density_result_%s' % plot_type


def eb_photo_density(plot_type='semilogx'):
    #file_id                      = '7hNKZBFHQbGL6xOf9r_w2Q'
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
        label=r'$\langle N_q\rangle$: %d' % 0,
    )

    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        eb_vec * 1e3,
        '-',
        color=colors[0],
        label=r'$T$: %.0f K' % T,
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
    y_vec_vals = ['$%d$' % v for v in y_vec_left]
    ax[0].set_yticks(y_vec_left)
    ax[0].set_yticklabels(y_vec_vals)

    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel(r'$n_\gamma$ (nm$^{-2}$)')

    y_vec_right = [eb_lim * 1e3]
    y_vec_right_labels = [r'$E_{B,0}$']

    ax_right = ax[0].twinx()
    ax_right.set_ylim(ax[0].get_ylim())
    ax_right.set_yticks(y_vec_right)
    ax_right.set_yticklabels(y_vec_right_labels)
    """
    th_pol_vec = 21 / 2**8 * 16 * sys.c_aEM**2 * (
        sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / abs(eb_vec)**3 * 1e36
    """

    th_pol_vec = 21 / 2**8 * 16 * sys.c_aEM**2 * (
        sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / abs(eb_vec)**3 * 1e36

    axins = ax[0].inset_axes([0.57, 0.05, 0.43, 0.45])
    axins.set_ylabel(r'$\alpha$ ($10^{-36}$ cm$^2$ V$^{-1}$)')

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
    axins.get_xaxis().set_visible(False)

    ins_y_vec_left = linspace(th_pol_vec[0], th_pol_vec[-1], 3)
    ins_y_vec_vals = ['%.1f' % v for v in ins_y_vec_left]
    axins.set_yticks(ins_y_vec_left)
    axins.set_yticklabels(ins_y_vec_vals)

    axins.yaxis.set_label_position("right")
    axins.yaxis.tick_right()

    getattr(axins, plot_type)(
        n_vec * sys.a0**2,
        th_pol_vec,
        '-',
        color=colors[0],
        label=r'$T$: %.0f K' % T,
    )

    ax[0].legend(loc='upper left')

    fig.tight_layout()

    return 'eb_photo_density_%s' % plot_type


def cond_fit_calc():
    fit_file_id = 'imDDS1DJRciMz_-rSvA1RQ'
    exp_power_data = loadtxt('extra/ef_power_spectrum.txt')

    w_vec = 2 * pi * exp_power_data[1:, 0]
    w_2_vec = linspace(w_vec[0], w_vec[-1], 2 * w_vec.size)
    power_norm_vec = exp_power_data[1:, 1] / simps(exp_power_data[1:, 1],
                                                   w_vec)
    #w_mean = simps(w_vec * power_norm_vec, w_vec)
    w_mean = w_vec[(w_vec * power_norm_vec).argmax()]
    #w_mean = w_vec[(power_norm_vec).argmax()]
    """
    ax[0].loglog(w_vec, w_vec * power_norm_vec / w_mean, 'r')
    ax[0].loglog(w_vec, power_norm_vec, 'b')
    ax[0].axvline(x = w_mean, color = 'g')
    ax[0].axvline(x = w_vec[power_norm_vec.argmax()], color = 'm')
    ax[0].axvline(x = simps(w_vec * power_norm_vec, w_vec), color = 'k')
    plt.show()
    exit()
    """

    n_exp_vec, cond_real, cond_imag, Na_exp_vec, cond_err_real, cond_err_imag = load_data(
        'bin/cdse_platelet_data')

    for i in [cond_real, cond_imag, cond_err_real, cond_err_imag]:
        i *= 3 / 2

    n_fit_vec, exc_fit_list, eb_fit_vec = load_data('extra/mu_e_data_%s' %
                                                    fit_file_id)

    sys = system_data(m_e, m_h, eps_r, T)
    n_fit_vec, eb_fit_vec = n_fit_vec[2:], eb_fit_vec[2:]
    mu_e_lim_fit, eb_lim_fit = exc_fit_list[:2]
    mu_e_fit_vec = array(exc_fit_list[2:])

    mu_h_fit_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_fit_vec])
    n_id_fit_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_fit_vec])
    n_exc_fit_vec = n_fit_vec - n_id_fit_vec

    L = 2e-3
    Na_fit_vec = Na_exp_vec * 1e4

    q_yield_fit_vec = n_id_fit_vec / n_fit_vec

    mu_dc_vec = array([720, 75])  # cm^2 v^-1 s^-1

    cond_fit = cond_real + 1j * cond_imag
    cond_err = abs(cond_err_real + 1j * cond_err_imag)**2

    def minimize_func(u_vec):
        vec = (cond_from_diff(
            u_vec,
            w_vec,
            power_norm_vec,
            w_mean,
            mu_dc_vec,
            Na_fit_vec,
            L,
            q_yield_fit_vec,
            eb_fit_vec,
            eps_r,
            sys,
        ) - cond_fit) / cond_err

        return real(sum(vec.conjugate() * vec))

    u_minzed = minimize(
        minimize_func,
        zeros_like(mu_dc_vec),
        method='nelder-mead',
        options={
            'xtol': 1e-14,
        },
    )

    print(u_minzed)

    print('Minimized u values: %s' % u_minzed.x)

    mob_minzed = mu_dc_vec * exp(u_minzed.x)

    print('Minimized: μ: %s cm² / Vs' % mob_minzed)

    file_id = '9xk12W--Tl6efYR-K76hoQ'

    n_vec, exc_list, eb_vec = load_data('extra/mu_e_data_%s' % file_id,
                                        globals())
    n_vec, eb_vec = n_vec[2:], eb_vec[2:]
    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    Na_vec = n_vec * Na_fit_vec[0] / (n_exp_vec[0] / surf_area)
    sys = system_data(m_e, m_h, eps_r, T)

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    q_yield_vec = n_id_vec / n_vec

    print(0.5 * sum(exp(u_minzed.x)))
    print(exp(u_minzed.x))

    return (mob_minzed,
            mob_integ_func(
                u_minzed.x,
                w_vec,
                power_norm_vec,
                mu_dc_vec,
                sys,
            ),
            cond_from_diff(
                u_minzed.x,
                w_vec,
                power_norm_vec,
                w_mean,
                mu_dc_vec,
                Na_vec,
                L,
                q_yield_vec,
                eb_vec,
                eps_r,
                sys,
            ))


def cond_fit(plot_type='plot'):
    file_id = '9xk12W--Tl6efYR-K76hoQ'

    n_vec, exc_list, eb_vec = load_data('extra/mu_e_data_%s' % file_id,
                                        globals())
    n_vec, eb_vec = n_vec[2:], eb_vec[2:]
    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    sys = system_data(m_e, m_h, eps_r, T)

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    q_yield_vec = n_id_vec / n_vec

    n_exp_vec, cond_real, cond_imag, Na_exp_vec, cond_err_real, cond_err_imag = load_data(
        'bin/cdse_platelet_data')

    cond_factor = 3 / 2
    for i in [cond_real, cond_imag, cond_err_real, cond_err_imag]:
        i *= cond_factor

    mob_dc_minzed, mob_minzed, cond_vec = cond_fit_calc()

    print('mu_dc: %s\nmu: %s' % (mob_dc_minzed, mob_minzed))
    print('Sum:\nmu_dc: %s\nmu: %s' % (sum(mob_dc_minzed), sum(mob_minzed)))

    ax[0].set_xlabel(r'$n_\gamma a_0^2$')
    ax[0].set_ylabel(r'$\Delta\sigma$ ($10^{-3}$ S m$^{-1}$)')

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, 1)
    ]

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

    getattr(ax[0], plot_type)(
        n_exp_vec / surf_area * sys.a0**2,
        cond_real * 1e3,
        'o',
        markeredgecolor='k',
        markerfacecolor='#FFFFFF'
    )

    getattr(ax[0], plot_type)(
        n_exp_vec / surf_area * sys.a0**2,
        -cond_imag * 1e3,
        'o',
        color='k',
    )

    ax[0].errorbar(
        n_exp_vec / surf_area * sys.a0**2,
        cond_real * 1e3,
        yerr=cond_err_real * 1e3,
        fmt='none',
        capsize=5,
        color='k',
    )

    ax[0].errorbar(
        n_exp_vec / surf_area * sys.a0**2,
        -cond_imag * 1e3,
        yerr=cond_err_imag * 1e3,
        fmt='none',
        capsize=5,
        color='k',
    )

    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        real(cond_vec) * 1e3,
        '--',
        color=colors[0],
    )
    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        -imag(cond_vec) * 1e3,
        '-',
        color=colors[0],
    )

    ax[0].axhline(
        y=0,
        color='k',
        linestyle='-',
    )

    ax[0].axvline(
        x=4 * sys.a0**2 / sys.lambda_th**2,
        color='g',
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

    ax[0].legend(
        [('r', '--')],
        [
            """$\mu_{dc,e}$: $%.0f$ cm$^2$ (Vs)$^{-1}$
$\mu_{dc,h}$: $%.0f$ cm$^2$ (Vs)$^{-1}$
$\mu$: $%.0f+%.0fi$ cm$^2$ (Vs)$^{-1}$""" %
            (mob_dc_minzed[0], mob_dc_minzed[1] if len(mob_dc_minzed) > 1 else
             0, real(sum(mob_minzed)), imag(sum(mob_minzed)))
        ],
        handler_map={tuple: AnyObjectHandler()},
        loc=(0.43, 0.43),
        #loc='center right',
        #loc='lower left',
    )

    fig.tight_layout()

    return 'cond_fit_%s' % plot_type


def mobility_2d_sample(plot_type='semilogx'):
    file_id = '9xk12W--Tl6efYR-K76hoQ'
    load_data('extra/mu_e_data_%s' % file_id, globals())
    sys = system_data(m_e, m_h, eps_r, T)

    w_vec = logspace(11, 16, 100)
    w_peak = 5.659114e+12  #w_vec[(power_norm_vec).argmax()]  # angular frequency, s^-1

    print('%e' % (w_peak * 0.5 / pi))

    mu_dc_vec = array([469.79873636, 51.58045606])  # cm^2 v^-1 s^-1
    diff_factor = 1e14 / sys.beta
    d_vec = mu_dc_vec * diff_factor  # nm^2 s^-1
    mob_vec = (diffusion_cx(w_vec, L_vec, d) / diff_factor for d in d_vec)

    ax[0].set_xlabel(r'$\omega$ (10$^{12}$ s$^{-1}$)')
    #ax[0].set_xlabel(r'$\omega$ (THz)')
    ax[0].set_ylabel(r'Mobility (cm$^2$ V$^{-1}$ s$^{-1}$)')

    mob_vec = sum([diffusion_cx(w_vec, L_vec, d) / diff_factor for d in d_vec],
                  axis=0)

    mob_peak = sum([diffusion_cx(array([w_peak]), L_vec, d) / diff_factor for d in d_vec],
                  axis=0)

    w_factor = 1e-12  #surf_area / sum(mu_dc_vec) / diff_factor
    x_vec = w_vec * w_factor

    ax[0].axvline(
        x=w_peak * w_factor,
        color='b',
        linestyle=':',
        label='$\omega_{peak}$',
    )

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
        [w_peak * w_factor],
        real(mob_peak),
        'o',
        markeredgecolor='g',
        markerfacecolor='#FFFFFF'
    )
    getattr(ax[0], plot_type)(
        [w_peak * w_factor],
        imag(mob_peak),
        'go',
    )

    ax[0].axhline(
        y=sum(mu_dc_vec),
        color='k',
        linestyle='--',
    )

    ax[0].set_xlim(x_vec[0], x_vec[-1])

    ax[0].legend([('g', '--')], [r'$\mu_{ac}(\omega)$'],
                 handler_map={tuple: AnyObjectHandler()},
                 loc=0)

    y_vec_right = [sum(mu_dc_vec)]
    y_vec_right_labels = [r'$\mu_{dc}$']

    ax_right = ax[0].twinx()
    ax_right.set_ylim(ax[0].get_ylim())
    ax_right.set_yticks(y_vec_right)
    ax_right.set_yticklabels(y_vec_right_labels)

    fig.tight_layout()

    return 'mobility_2d_sample_%s' % plot_type


def mobility_2d_integ(plot_type='loglog'):
    file_id = '9xk12W--Tl6efYR-K76hoQ'

    n_vec, exc_list, eb_vec = load_data('extra/mu_e_data_%s' % file_id,
                                        globals())
    n_vec, eb_vec = n_vec[2:], eb_vec[2:]
    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    sys = system_data(m_e, m_h, eps_r, T)

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    q_yield_vec = n_id_vec / n_vec

    n_exp_vec, cond_real, cond_imag, Na_exp_vec, cond_err_real, cond_err_imag = load_data(
        'bin/cdse_platelet_data')

    cond_factor = 3 / 2
    for i in [cond_real, cond_imag, cond_err_real, cond_err_imag]:
        i *= cond_factor

    mob_dc_minzed, mob_minzed, cond_vec = cond_fit_calc()

    exp_power_data = loadtxt('extra/ef_power_spectrum.txt')

    w_vec = 2 * pi * exp_power_data[1:, 0]
    w_2_vec = linspace(w_vec[0], w_vec[-1], 2 * w_vec.size)
    power_norm_vec = exp_power_data[1:, 1] / simps(exp_power_data[1:, 1],
                                                   w_vec)

    w_peak = w_vec[(power_norm_vec).argmax()]  # angular frequency, s^-1

    print('%e' % w_peak)

    mu_dc_vec = array([600, 20])  # cm^2 v^-1 s^-1
    diff_factor = 1e14 / sys.beta
    d_vec = mu_dc_vec * diff_factor  # nm^2 s^-1
    mob_vec = (diffusion_cx(w_vec, L_vec, d) / diff_factor for d in d_vec)

    ax[0].set_xlabel(r'$\omega$ (10$^{12}$ s$^{-1}$)')
    #ax[0].set_xlabel(r'$\omega$ (THz)')
    ax[0].set_ylabel(r'Mobility (cm$^2$ V$^{-1}$ s$^{-1}$)')

    mob_vec = sum(
        [diffusion_cx(w_2_vec, L_vec, d) / diff_factor for d in d_vec], axis=0)
    mob_minzed_vec = diffusion_cx(w_2_vec, L_vec,
                                  diff_factor * sum(mu_minzed)) / diff_factor

    w_factor = 1e-12  #surf_area / sum(mu_dc_vec) / diff_factor
    x_vec = w_2_vec * w_factor

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

    ax[0].axvline(
        x=w_peak * w_factor,
        color='m',
        linestyle=':',
        label='$\omega_{peak}$',
    )

    ax[0].set_xlim(min(x_vec[0], x_minzed_vec[0]),
                   max(x_vec[-1], x_minzed_vec[-1]))

    ax[0].legend([('g', '--'), ('b', '--')],
                 [r'$\mu_{ac}(\omega)$', r'$\mu^{*}_{ac}(\omega)$'],
                 handler_map={tuple: AnyObjectHandler()},
                 loc='lower right')

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
    print('Calling %s(%s)' % (p, l))
    filename = locals()[p](l)

    #plt.savefig('plots/papers/exciton1/%s.png' % filename)
    plt.savefig('plots/papers/exciton1/%s.pdf' % filename)
#plt.savefig('plots/papers/exciton1/%s.eps' % filename)

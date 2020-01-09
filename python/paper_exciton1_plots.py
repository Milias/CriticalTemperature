from common import *

plt.style.use('dark_background')
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


N_k = 1 << 10

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
    """
    th_pol_vec = 21 / 2**8 * 16 * sys.c_aEM**2 * (
        sys.c_hbarc * 1e-9 / eps_r)**2 * sys.c_e_charge / abs(eb_vec)**3
    """

    th_pol_vec = 4 * sqrt(2) * 21 / 2**8 * (
        sys.c_hbarc * 1e-9)**2 * eps_r / sys.c_aEM * sys.c_e_charge / (
            -sys.m_p * eb_vec)**1.5

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

    n_vec = logspace(-0.7, 2.3, 100) / surf_area
    T_vec = linspace(130, 350, 5)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, len(T_vec))
    ]

    ax[0].set_xlabel(r'$\langle N_q \rangle$')
    ax[0].set_ylabel(r'$\lambda_{s}(n_q)$ (nm)')

    x_vec = n_vec * surf_area

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        sys = system_data(m_e, m_h, eps_r, T)
        y_vec = array([sys.ls_ideal(n) for n in n_vec])

        getattr(ax[0], plot_type)(
            x_vec,
            1 / y_vec,
            '-',
            color=c,
            label=r'$T$: %.0f K' % T,
        )

    ax[0].set_xlim(x_vec[0], x_vec[-1])
    #ax[0].set_ylim(0, None)

    ax[0].set_yticks([1, 10])
    ax[0].set_yticklabels(['$1$', '$10$'])

    x_vec_top = logspace(-3, 0, 4) * surf_area
    x_vec_vals = x_vec_top / surf_area
    x_vec_minor_top = array([(v * linspace(0.1, 1., 9) * surf_area).tolist()
                             for v in x_vec_vals]).ravel()
    x_vec_vals = ['$10^{%.0f}$' % log10(v) for v in x_vec_vals]

    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticks(x_vec_minor_top, minor=True)
    ax_top.set_xticklabels(x_vec_vals,
                           fontdict={'verticalalignment': 'baseline'})
    ax_top.set_xticklabels([], minor=True)
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xlabel('$n_q$ (nm$^{-2}$)', labelpad=8.)

    y_vec_right = [1 / sys.sys_ls]
    y_vec_right_labels = [r'$\lambda_s^\mathrm{Sat}$']

    ax_right = ax[0].twinx()
    ax_right.set_yscale('log')
    ax_right.set_ylim(ax[0].get_ylim())
    ax_right.set_yticks(y_vec_right)
    ax_right.set_yticklabels(y_vec_right_labels)

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'scr_length_density_%s' % plot_type


def real_space_lwl_potential_density(plot_type='plot'):
    file_id = 'imDDS1DJRciMz_-rSvA1RQ'
    load_data('extra/mu_e_data_%s' % file_id, globals())

    sys = system_data(m_e, m_h, eps_r, T_vec[0])

    print(1 / sys.sys_ls)

    n_max = sys.density_ideal(-1.0 / sys.beta)
    n_vec = logspace(log10(0.1), log10(n_max * surf_area), 5) / surf_area
    mu_e_vec = array([sys.mu_ideal(n) for n in n_vec])
    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])

    #n_vec     = logspace(log10(0.1), log10(8), 5) / surf_area
    ls_vec = array([sys.ls_ideal(n) for n in n_vec])

    print(1 / ls_vec)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, ls_vec.size)
    ]

    x_vec = linspace(0.3, 6, 100) / sys.a0

    ax[0].set_xlabel(r'$r$ $a_0^{-1}$')
    ax[0].set_ylabel(r'$V_{sc}(r;n_q)$ (meV)')

    cou_vec = array(time_func(plasmon_rpot_lwl_v, x_vec * sys.a0, 1e-8,
                              sys)) * 1e3
    getattr(ax[0], plot_type)(
        x_vec,
        cou_vec,
        '--',
        color='r',
        label=r'$\langle N_q \rangle$: $0$',
        linewidth=0.9,
    )

    for c, (i, ls) in zip(colors, enumerate(ls_vec)):
        y_vec = array(time_func(plasmon_rpot_lwl_v, x_vec * sys.a0, ls,
                                sys)) * 1e3

        getattr(ax[0], plot_type)(
            x_vec,
            y_vec,
            '--',
            color=c,
            dashes=(0.2, 3.),
            dash_capstyle='round',
            linewidth=2,
        )

        y_vec = array(
            time_func(plasmon_rpot_ht_v, x_vec * sys.a0, mu_e_vec[i],
                      mu_h_vec[i], sys)) * 1e3
        getattr(ax[0], plot_type)(
            x_vec,
            y_vec,
            '-',
            color=c,
            label=r'$\langle N_q \rangle$: $%.1f$' % (n_vec[i] * surf_area),
        )

    sys_vec = array(
        time_func(plasmon_rpot_lwl_v, x_vec * sys.a0, sys.sys_ls, sys)) * 1e3
    getattr(ax[0], plot_type)(
        x_vec,
        sys_vec,
        '--',
        color='k',
        label=r'$\langle N_q \rangle \rightarrow \infty$',
        linewidth=0.9,
    )

    ax[0].set_ylim(-0.25e3, 0.005e3)
    ax[0].set_xlim(0, x_vec[-1])
    ax[0].axhline(
        y=0,
        color='k',
        linewidth=0.9,
    )

    ax[0].set_yticks(arange(0, -250, -50).tolist())
    ax[0].set_yticklabels(
        ['$%.0f$' % v for v in ax[0].yaxis.get_majorticklocs()])

    x_vec_top = linspace(0, 12, 7) / sys.a0
    x_vec_vals = x_vec_top * sys.a0
    x_vec_vals = ['%.0f' % v for v in x_vec_vals]

    ax_top = ax[0].twiny()
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals,
                           fontdict={'verticalalignment': 'baseline'})
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xlabel('$r$ (nm)')

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'real_space_lwl_potential_density_%s' % plot_type


def energy_level_mb_density(plot_type='semilogx'):
    file_id = 'cJBn0Vb3SICHkdj4QL7NHA'
    #file_id = 'XGXk201XRguBy0lTo2kS9Q'
    values_list = load_data('extra/eb_mb_temp_%s' % file_id, globals())

    lwl_file_id = '4EU_uHkYR2C1v_esi5Hexg'
    #lwl_file_id        = '0RLLrVMrTgqRKCWw3jF0ow'
    lwl_values_list = load_data('extra/eb_lwl_temp_%s' % lwl_file_id,
                                globals())

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, len(T_vec))
    ]

    ax[0].set_xlabel(r'$\langle N_q \rangle$')
    ax[0].set_ylabel(r'$E_B(n_q)$ (meV)')

    sys = system_data(m_e, m_h, eps_r, T_vec[0])
    ax[0].axhline(
        y=z_cou_lwl * 1e3,
        color='r',
        linestyle='--',
        label=r'$\langle N_q\rangle$: %d' % 0,
        linewidth=0.9,
    )

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        sys = system_data(m_e, m_h, eps_r, T)
        mu_e_vec = values_list[2 * i]
        y_vec = values_list[2 * i + 1]

        mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
        n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])

        lwl_mu_e_vec = lwl_values_list[2 * i]
        lwl_y_vec = lwl_values_list[2 * i + 1]

        lwl_mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in lwl_mu_e_vec])
        lwl_n_id_vec = array(
            [sys.density_ideal(mu_e) for mu_e in lwl_mu_e_vec])

        getattr(ax[0], plot_type)(
            n_id_vec * surf_area,
            y_vec * 1e3,
            '-',
            color=c,
            label='$T$: %.0f K' % sys.T,
            zorder=10,
        )

        getattr(ax[0], plot_type)(
            lwl_n_id_vec * surf_area,
            lwl_y_vec * 1e3,
            '--',
            color=c,
            zorder=9,
            dashes=(0.2, 3.),
            dash_capstyle='round',
            linewidth=2,
        )

        getattr(ax[0], plot_type)(
            n_id_vec[-1] * surf_area,
            y_vec[-1] * 1e3,
            'o',
            color=c,
            zorder=11,
        )

    z_sys_lwl = time_func(plasmon_det_zero_lwl, N_k, sys.sys_ls, sys)
    #z_sys_lwl    = -0.05709309806861098
    #z_sys_lwl    = lwl_y_vec[-1]

    ax[0].axhline(
        y=z_sys_lwl * 1e3,
        color='w',
        linestyle='--',
        label=r'$\langle N_q\rangle\rightarrow\infty$',
        linewidth=0.9,
    )

    print(z_sys_lwl)
    print(z_sys_lwl / sys.c_kB)

    ax[0].set_xlim(1e-2, 1e2)
    ax[0].set_ylim(z_cou_lwl * 1e3 - 3, z_sys_lwl * 1e3 + 5)

    y_vec_left = [z_cou_lwl * 1e3] + arange(-70, -185,
                                            -25).tolist() + [z_sys_lwl * 1e3]
    y_vec_left_vals = ['$%.0f$' % v for v in y_vec_left]
    ax[0].set_yticks(y_vec_left)
    ax[0].set_yticklabels(y_vec_left_vals)

    x_vec_top = logspace(-4, 0, 5) * surf_area
    x_vec_vals = x_vec_top / surf_area
    x_vec_minor_top = array([(v * linspace(0.1, 1., 9) * surf_area).tolist()
                             for v in x_vec_vals]).ravel()
    x_vec_vals = ['$10^{%.0f}$' % log10(v) for v in x_vec_vals]

    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticks(x_vec_minor_top, minor=True)
    ax_top.set_xticklabels(x_vec_vals,
                           fontdict={'verticalalignment': 'baseline'})
    ax_top.set_xticklabels([], minor=True)
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xlabel('$n_q$ (nm$^{-2}$)', labelpad=8.)

    y_vec_right = [z_sys_lwl * 1e3]
    y_vec_right_labels = [r'$E_B^\mathrm{Sat}$']

    ax_right = ax[0].twinx()
    ax_right.set_yticks(y_vec_right)
    ax_right.set_yticklabels(y_vec_right_labels)
    ax_right.set_ylim(ax[0].get_ylim())
    #ax[0].set_ylim(z_cou_lwl * 1e3 - 3, -175)

    ax[0].legend(loc=(0.02, 0.3))

    fig.tight_layout()

    return 'energy_level_mb_density_dark_%s' % plot_type


def density_result(plot_type='loglog'):
    #file_id                    = '9xk12W--Tl6efYR-K76hoQ' #higher masses
    file_id = 'yzpPQfeOQHeIHTffgrNgng'

    n_exp_vec, cond_real, cond_imag, N_a_exp_vec, cond_err_real, cond_err_imag = load_data(
        'bin/cdse_platelet_data')

    n_vec, exc_list, eb_vec = load_data('extra/mu_e_data_%s' % file_id,
                                        globals())
    n_vec, eb_vec = n_vec[2:], eb_vec[2:]
    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    sys = system_data(m_e, m_h, eps_r, T_vec[0])

    print('beta * mu_e_lim = %.2f' % (sys.beta * mu_e_lim))

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    n_q0 = sys.density_ideal(mu_e_lim)

    print('n_q^inf = %.2e' % n_q0)

    n_exc2_vec = -2**4 * 2 * sys.m_p * (1 / sys.m_pe + 1 / sys.m_ph) / (
        2 * pi * sys.c_hbarc**2 *
        sys.beta) * log(1 - exp(sys.beta *
                                (2 * (mu_e_vec + mu_h_vec) - 2 * eb_vec -
                                 (-30e-3))))

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, 1)
    ]

    ax[0].set_xlabel(r'$n_\gamma a_0^2$')
    ax[0].set_ylabel('Number of Particles in a Bohr Area')

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
        label=r'$n_{\mathrm{exc}} a_0^2$',
    )

    ax[0].axvline(
        x=n_exp_vec[0] / surf_area * sys.a0**2,
        color='m',
        linestyle='--',
        dashes=(0.2, 3.),
        dash_capstyle='round',
        linewidth=2,
        zorder=1,
    )

    ax[0].axvline(
        x=n_exp_vec[-1] / surf_area * sys.a0**2,
        color='m',
        linestyle='--',
        dashes=(0.2, 3.),
        dash_capstyle='round',
        linewidth=2,
        zorder=1,
    )

    ax[0].set_xlim(n_vec[0] * sys.a0**2, n_vec[-1] * sys.a0**2)

    lambda_th = sys.c_hbarc * sqrt(2 * pi * sys.beta / sys.m_p)
    ax[0].axvline(
        x=4 * sys.a0**2 / lambda_th**2,
        color='g',
        label=r'$n_{\mathrm{exc}} \lambda_\mathrm{th}^2 \simeq g_s^2$',
        linewidth=0.9,
        zorder=1,
    )

    x_vec_top = logspace(-3, 0, 4) * sys.a0**2
    x_vec_vals = x_vec_top / sys.a0**2
    x_vec_minor_top = array([(v * linspace(0.1, 1., 9) * sys.a0**2).tolist()
                             for v in x_vec_vals]).ravel()
    x_vec_vals = ['$10^{%.0f}$' % log10(v) for v in x_vec_vals]

    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticks(x_vec_minor_top, minor=True)
    ax_top.set_xticklabels(x_vec_vals,
                           fontdict={'verticalalignment': 'baseline'})
    ax_top.set_xticklabels([], minor=True)
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xlabel(r'$n_\gamma$ (nm$^{-2}$)', labelpad=8.)

    y_vec_right = [n_q0 * sys.a0**2]
    y_vec_right_labels = [r'$n_q^{\infty} a_0^2$']

    ax_right = ax[0].twinx()
    ax_right.set_yscale('log')
    ax_right.set_ylim(ax[0].get_ylim())
    ax_right.set_yticks(y_vec_right)
    ax_right.set_yticklabels(y_vec_right_labels)

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'density_result_dark_%s' % plot_type


def eb_photo_density(plot_type='semilogx'):
    #file_id                      = '9xk12W--Tl6efYR-K76hoQ'
    file_id = 'yzpPQfeOQHeIHTffgrNgng'

    n_exp_vec, cond_real, cond_imag, N_a_exp_vec, cond_err_real, cond_err_imag = load_data(
        'bin/cdse_platelet_data')

    n_vec, exc_list, eb_vec = load_data('extra/mu_e_data_%s' % file_id,
                                        globals())
    n_vec, eb_vec = n_vec[2:], eb_vec[2:]
    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    sys = system_data(m_e, m_h, eps_r, T_vec[0])

    print(sys.a0)

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    th_pol_vec = 4 * sqrt(2) * 21 / 2**8 * (
        sys.c_hbarc * 1e-9)**2 * eps_r / sys.c_aEM * sys.c_e_charge / (
            -sys.m_p * eb_vec)**1.5 * 1e36

    print(0.6668 / 0.8056)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, 1)
    ]

    ax[0].set_xlabel(r'$n_\gamma a_0^2$')
    ax[0].set_ylabel(r'$\alpha$ ($10^{-36}$ cm$^2$ V$^{-1}$)')

    z_cou_lwl = time_func(plasmon_det_zero_ht, 1 << 10, -30, sys.get_mu_h(-30),
                          sys)

    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        th_pol_vec,
        '-',
        color=colors[0],
        label=r'$T$: %.0f K' % T_vec[0],
    )

    ax[0].axhline(
        y=0,
        color='r',
        linestyle='--',
        label=r'$\langle N_q\rangle$: %d' % 0,
        linewidth=0.9,
    )

    ax[0].axvline(
        x=n_exp_vec[0] / surf_area * sys.a0**2,
        color='m',
        linestyle='--',
        dashes=(0.2, 3.),
        dash_capstyle='round',
        linewidth=2,
        zorder=1,
    )

    ax[0].axvline(
        x=n_exp_vec[-1] / surf_area * sys.a0**2,
        color='m',
        linestyle='--',
        dashes=(0.2, 3.),
        dash_capstyle='round',
        linewidth=2,
        zorder=1,
    )

    ax[0].set_xlim(n_vec[0] * sys.a0**2, n_vec[-1] * sys.a0**2)
    ax[0].set_ylim(1.5, 1.72)

    x_vec_top = logspace(-3, 0,
                         4) * sys.a0**2  #ax[0].xaxis.get_majorticklocs()[2:-2]
    x_vec_vals = x_vec_top / sys.a0**2
    x_vec_minor_top = array([(v * linspace(0.1, 1., 9) * sys.a0**2).tolist()
                             for v in x_vec_vals]).ravel()
    x_vec_vals = ['$10^{%.0f}$' % log10(v) for v in x_vec_vals]

    y_vec_left = arange(1.5, 1.75,
                        0.05)  #linspace(th_pol_vec[0], th_pol_vec[-1], 5)
    y_vec_vals = ['%.2f' % v for v in y_vec_left]
    ax[0].set_yticks(y_vec_left)
    ax[0].set_yticklabels(y_vec_vals)

    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticks(x_vec_minor_top, minor=True)
    ax_top.set_xticklabels(x_vec_vals,
                           fontdict={'verticalalignment': 'baseline'})
    ax_top.set_xticklabels([], minor=True)
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xlabel(r'$n_\gamma$ (nm$^{-2}$)', labelpad=8.0)

    y_vec_right = [eb_lim * 1e3]
    y_vec_right_labels = [r'$E_B^\infty$']

    ax_right = ax[0].twinx()
    ax_right.set_ylim(ax[0].get_ylim())
    ax_right.set_yticks(y_vec_right)
    ax_right.set_yticklabels(y_vec_right_labels)

    axins = ax[0].inset_axes([0.57, 0.05, 0.43, 0.45])
    axins.set_ylabel(r'$E_B$ (meV)')

    axins.axvline(
        x=n_exp_vec[0] / surf_area * sys.a0**2,
        color='m',
        linestyle='--',
        dashes=(0.2, 3.),
        dash_capstyle='round',
        linewidth=2,
        zorder=1,
    )

    axins.axvline(
        x=n_exp_vec[-1] / surf_area * sys.a0**2,
        color='m',
        linestyle='--',
        dashes=(0.2, 3.),
        dash_capstyle='round',
        linewidth=2,
        zorder=1,
    )

    axins.set_xlim(n_vec[0] * sys.a0**2, n_vec[-1] * sys.a0**2)
    axins.get_xaxis().set_visible(False)

    ins_y_vec_left = [z_cou_lwl * 1e3] + [eb_vec[-1] * 1e3]
    ins_y_vec_vals = [r'$E_B^\mathrm{Cou}$'
                      ] + ['$%d$' % v for v in ins_y_vec_left[1:]]
    axins.set_yticks(ins_y_vec_left)
    axins.set_yticklabels(ins_y_vec_vals)

    axins.yaxis.set_label_position("right")
    axins.yaxis.tick_right()
    axins.yaxis.set_label_coords(1.1, 0.5)

    axins.axhline(
        y=z_cou_lwl * 1e3,
        color='r',
        linestyle='--',
        label=r'$\langle N_q\rangle$: %d' % 0,
        linewidth=0.9,
    )

    getattr(axins, plot_type)(
        n_vec * sys.a0**2,
        eb_vec * 1e3,
        '-',
        color=colors[0],
        label=r'$T$: %.1f K' % T_vec[0],
    )

    ax[0].legend(loc='upper left')

    fig.tight_layout()

    return 'eb_photo_density_%s' % plot_type


def cond_fit_calc():
    #fit_file_id       = 'imDDS1DJRciMz_-rSvA1RQ'
    fit_file_id = 'zYt4kfzmTSiJo4O05ZNEQQ'
    exp_power_data = loadtxt('extra/ef_power_spectrum.txt')

    w_vec = 2 * pi * exp_power_data[1:, 0]
    w_2_vec = linspace(w_vec[0], w_vec[-1], 2 * w_vec.size)
    power_norm_vec = exp_power_data[1:, 1] / simps(exp_power_data[1:, 1],
                                                   w_vec)
    #w_mean    = simps(w_vec * power_norm_vec, w_vec)
    #w_mean    = w_vec[(w_vec * power_norm_vec).argmax()]
    w_mean = w_vec[(power_norm_vec).argmax()]
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

    sys = system_data(m_e, m_h, eps_r, T_vec[0])
    n_fit_vec, eb_fit_vec = n_fit_vec[2:], eb_fit_vec[2:]
    mu_e_lim_fit, eb_lim_fit = exc_fit_list[:2]
    mu_e_fit_vec = array(exc_fit_list[2:])

    mu_h_fit_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_fit_vec])
    n_id_fit_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_fit_vec])
    n_exc_fit_vec = n_fit_vec - n_id_fit_vec

    L = 2e-3
    Na_fit_vec = Na_exp_vec * 1e4

    q_yield_fit_vec = n_id_fit_vec / n_fit_vec

    mu_dc_vec = array([1000, 200])  # cm^2 v^-1 s^-1

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

    #file_id    = '9xk12W--Tl6efYR-K76hoQ'
    file_id = 'yzpPQfeOQHeIHTffgrNgng'

    n_vec, exc_list, eb_vec = load_data('extra/mu_e_data_%s' % file_id,
                                        globals())
    n_vec, eb_vec = n_vec[2:], eb_vec[2:]
    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    Na_vec = n_vec * Na_fit_vec[0] / (n_exp_vec[0] / surf_area)
    sys = system_data(m_e, m_h, eps_r, T_vec[0])

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
    #file_id              = '9xk12W--Tl6efYR-K76hoQ'
    file_id = 'yzpPQfeOQHeIHTffgrNgng'

    n_vec, exc_list, eb_vec = load_data('extra/mu_e_data_%s' % file_id,
                                        globals())
    n_vec, eb_vec = n_vec[2:], eb_vec[2:]
    mu_e_lim, eb_lim = exc_list[:2]
    mu_e_vec = array(exc_list[2:])

    sys = system_data(m_e, m_h, eps_r, T_vec[0])

    mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
    n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])
    n_exc_vec = n_vec - n_id_vec

    print(sys.beta * mu_e_lim)
    print(sys.beta * sys.get_mu_h(mu_e_lim))

    print(eb_lim - mu_e_vec[-1] - sys.get_mu_h(mu_e_vec[-1]))

    z_eb1 = plasmon_det_zero_ht_v1(N_k, array([mu_e_vec[-1]]), sys)[0]
    n_exc_1 = sys.density_exc(mu_e_vec[-1] + sys.get_mu_h(mu_e_vec[-1]), z_eb1)

    print('%.2e' % n_exc_1)
    print(n_exc_vec[-1])

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
    ax[0].set_ylabel(r'$\sigma_\parallel$ ($10^{-3}$ S m$^{-1}$)')

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, 1)
    ]

    ax[0].axvline(
        x=n_exp_vec[0] / surf_area * sys.a0**2,
        color='m',
        linestyle='--',
        dashes=(0.2, 3.),
        dash_capstyle='round',
        linewidth=2,
    )

    ax[0].axvline(
        x=n_exp_vec[-1] / surf_area * sys.a0**2,
        color='m',
        linestyle='--',
        dashes=(0.2, 3.),
        dash_capstyle='round',
        linewidth=2,
    )

    getattr(ax[0], plot_type)(
        n_exp_vec / surf_area * sys.a0**2,
        cond_real * 1e3,
        'o',
        markeredgecolor='w',
        markerfacecolor='#000000',
        zorder=10,
    )

    getattr(ax[0], plot_type)(
        n_exp_vec / surf_area * sys.a0**2,
        -cond_imag * 1e3,
        'o',
        color='w',
        zorder=10,
    )

    ax[0].errorbar(
        n_exp_vec / surf_area * sys.a0**2,
        cond_real * 1e3,
        yerr=cond_err_real * 1e3,
        fmt='none',
        capsize=5,
        color='w',
        zorder=10,
    )

    ax[0].errorbar(
        n_exp_vec / surf_area * sys.a0**2,
        -cond_imag * 1e3,
        yerr=cond_err_imag * 1e3,
        fmt='none',
        capsize=5,
        color='w',
        zorder=10,
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
        color='w',
        linestyle='-',
        linewidth=0.9,
        zorder=10,
    )

    ax[0].axvline(
        x=4 * sys.a0**2 / sys.lambda_th**2,
        color='g',
        linewidth=0.9,
        zorder=1,
    )

    ax[0].set_xlim(1 / surf_area * sys.a0**2, 60 / surf_area * sys.a0**2)
    ax[0].set_ylim(-cond_factor * 12.5, cond_factor * 5)

    x_vec_top = linspace(0.02, 0.18, 5) * sys.a0**2
    x_vec_vals = x_vec_top / sys.a0**2
    x_vec_vals = ['%.2f' % v for v in x_vec_vals]

    ax_top = ax[0].twiny()
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals,
                           fontdict={'verticalalignment': 'baseline'})
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xlabel(r'$n_\gamma$ (nm$^{-2}$)')
    """
    ax[0].legend(
        [('r', '--')],
        [
            '$\mu_{\mathrm{DC},e}$: $%.0f$ cm$^2$ V$^{-1}$ s$^{-1}$\n$\mu_{\mathrm{DC},h}$: $%.0f$ cm$^2$ V$^{-1}$ s$^{-1}$'
            % (mob_dc_minzed[0],
               mob_dc_minzed[1] if len(mob_dc_minzed) > 1 else 0)
        ],
        handler_map={tuple: AnyObjectHandler()},
        #loc = (0.405, 0.44),
        loc=(0.445, 0.5),
    )
    """

    ax[0].text(
        0.57,
        0.55,
        '$\mu_{\mathrm{DC},e}$: $%.0f$ cm$^2$ V$^{-1}$ s$^{-1}$\n$\mu_{\mathrm{DC},h}$: $%.0f$ cm$^2$ V$^{-1}$ s$^{-1}$'
        % (mob_dc_minzed[0], mob_dc_minzed[1]),
        transform=ax[0].transAxes,
        bbox=dict(boxstyle='round', fc='#00000099', ec='#666666'),
    )

    fig.tight_layout()

    return 'cond_fit_dark_%s' % plot_type


def mobility_2d_sample(plot_type='semilogx'):
    #file_id = '94ndrNKPTE67MoCLoChR2Q'
    file_id = 'yzpPQfeOQHeIHTffgrNgng'
    load_data('extra/mu_e_data_%s' % file_id, globals())
    sys = system_data(m_e, m_h, eps_r, T_vec[0])

    w_vec = logspace(11, 16, 100)
    w_peak = 5.659114e12  #w_vec[(power_norm_vec).argmax()]  # angular frequency, s^-1

    print(sqrt(sys.m_pe + sys.m_ph) * 2 * log(2) / sys.lambda_th**2)

    print('%e' % (w_peak * 0.5 / pi))

    mob_dc_minzed, mob_minzed, cond_vec = cond_fit_calc()

    mu_dc_vec = mob_dc_minzed

    diff_factor = 1e14 / sys.beta
    d_vec = mu_dc_vec * diff_factor  # nm^2 s^-1
    mob_vec = (diffusion_cx(w_vec, L_vec, d) / diff_factor for d in d_vec)

    ax[0].set_xlabel(r'$\omega$ (10$^{12}$ s$^{-1}$)')
    ax[0].set_ylabel(r'$\mu(\omega)$ (cm$^2$ V$^{-1}$ s$^{-1}$)')

    mob_vec = sum([diffusion_cx(w_vec, L_vec, d) / diff_factor for d in d_vec],
                  axis=0)

    mob_peak = sum(
        [diffusion_cx(array([w_peak]), L_vec, d) / diff_factor for d in d_vec],
        axis=0)

    w_factor = 1e-12  #surf_area / sum(mu_dc_vec) / diff_factor
    x_vec = w_vec * w_factor

    ax[0].axvline(
        x=w_peak * w_factor,
        color='b',
        linestyle='--',
        label='$\omega_{peak}$',
        dashes=(0.2, 3.),
        dash_capstyle='round',
        linewidth=2,
    )

    getattr(ax[0], plot_type)(
        x_vec,
        real(mob_vec),
        'g--',
    )
    getattr(ax[0], plot_type)(
        x_vec,
        imag(mob_vec),
        'g-',
    )

    getattr(ax[0], plot_type)(
        [w_peak * w_factor],
        real(mob_peak),
        'o',
        markeredgecolor='g',
        markerfacecolor='#FFFFFF',
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
        linewidth=0.9,
    )

    ax[0].set_xlim(x_vec[0], x_vec[-1])
    ax[0].set_ylim(0, None)
    """
    ax[0].legend([('g', '--')], [r'$\mu(\omega)$'],
                 handler_map={tuple: AnyObjectHandler()},
                 loc=0)
    """

    x_vec_top = array([w_peak])
    x_vec_vals = x_vec_top
    x_vec_vals = [r'$\omega_\mathrm{peak}$']

    ax_top = ax[0].twiny()
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals,
                           fontdict={'verticalalignment': 'baseline'})
    ax_top.set_xlim(ax[0].get_xlim())

    y_vec_left = arange(0,
                        sum(mu_dc_vec) * 0.9, 100).tolist() + [sum(mu_dc_vec)]
    y_vec_vals = ['$%d$' % v for v in y_vec_left]
    ax[0].set_yticks(y_vec_left)
    ax[0].set_yticklabels(y_vec_vals)

    y_vec_right = [sum(mu_dc_vec)]
    y_vec_right_labels = [r'$\mu_{\mathrm{DC}}$']

    ax_right = ax[0].twinx()
    ax_right.set_ylim(ax[0].get_ylim())
    ax_right.set_yticks(y_vec_right)
    ax_right.set_yticklabels(y_vec_right_labels)

    fig.tight_layout()

    return 'mobility_2d_sample_%s' % plot_type


plots_list = [pysys.argv[1:]]

for p, l in plots_list:
    print('Calling %s(%s)' % (p, l))
    filename = locals()[p](l)

    plt.savefig(
        'plots/papers/exciton1/%s.pdf' % filename,
        transparent=True,
    )

from common import *
plt.style.use('dark_background')
#plt.xkcd()
#plt.rcParams['font.family'] = 'SF Pro Display'

import statsmodels.api as sm

N_k = 1 << 12

mm_to_inches = 0.03937008

fig_size = (6.8, 5.3)
fig_size_mm = tuple(array(pysys.argv[3:5], dtype=float) * mm_to_inches)

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size_mm, dpi=300)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]

Lx, Ly, Lz = 34.0, 9.6, 1.4  # nm
L_vec = array([Lx, Ly])
surf_area = Lx * Ly  # nm^2
eb_cou = 0.193


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

    x_vec = linspace(1, 12, 300) / sys.a0

    #ax[0].set_title('Real space potential\nClassical Limit')
    ax[0].set_xlabel(r'$r$ / $a_0$')
    ax[0].set_ylabel('')

    cou_vec = array(time_func(plasmon_rpot_lwl_v, x_vec, 1e-8, sys))
    getattr(ax[0], plot_type)(
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
        getattr(ax[0], plot_type)(
            x_vec,
            y_vec,
            '-',
            color=c,
            label=r'$\langle N_e\rangle$: %.1f' % num_e,
        )

    sys_vec = array(time_func(plasmon_rpot_lwl_v, x_vec, sys.sys_ls, sys))
    getattr(ax[0], plot_type)(
        x_vec,
        sys_vec,
        '--',
        color='w',
        label=r'$\langle N_e\rangle\rightarrow\infty$',
    )

    ax[0].set_yticks([0.0])

    ax[0].set_ylim(-0.3, 0.03)
    ax[0].set_xlim(0, x_vec[-1])
    ax[0].axhline(y=0, color='w')

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


def energy_level_mb_density(plot_type='loglog'):
    file_id = 'pa2MgrnmTKKg4u_gz3WY8Q'
    values_list = load_data('extra/eb_values_temp_%s' % file_id, globals())

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, len(T_vec))
    ]

    ax[0].set_xlabel(r'$\langle N_e \rangle$')
    ax[0].set_ylabel(r'$E_b$ (meV)')

    sys = system_data(m_e, m_h, eps_r, 300)

    z_cou_lwl = -eb_cou  #time_func(plasmon_det_zero_lwl, N_k, 1e-8, sys, -1e-3)
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
    z_sys_lwl = time_func(plasmon_det_zero_lwl, N_k, sys.sys_ls, sys, -1e-3)

    ax[0].axhline(
        y=z_sys_lwl * 1e3,
        color='k',
        linestyle='--',
        label=r'$\langle N_e\rangle\rightarrow\infty$',
    )

    ax[0].yaxis.set_ticks([1e3 * z_sys_lwl, 1e3 * z_cou_lwl])
    ax[0].set_yticklabels(['%d' % z for z in [1e3 * z_sys_lwl, 1e3 * z_cou_lwl]])
    """

    ax[0].set_xlim(1e-4 * surf_area, 2.1e-2 * surf_area)

    ax[0].yaxis.set_ticks([1e3 * z_cou_lwl, y_vec[-1] * 1e3])
    ax[0].set_yticklabels(
        ['%d' % z for z in [1e3 * z_cou_lwl, y_vec[-1] * 1e3]])
    ax[0].yaxis.set_label_coords(-0.05, 0.5)

    x_vec_top = ax[0].xaxis.get_majorticklocs()[2:-2]
    x_vec_vals = (x_vec_top / surf_area) * 1e3
    x_vec_vals = ['%.2f' % v for v in x_vec_vals]

    ax[0].set_xticks(x_vec_top)
    ax[0].set_xticklabels(['0.1', '1'])
    """
    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel(r'$n_e$ ($10^{-3}$ nm$^{-2}$)')
    """

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'energy_level_mb_density_%s' % plot_type


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
    ax[0].set_ylabel('Average Number of Particles\n per Bohr Area')

    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        n_id_vec * sys.a0**2,
        '--',
        color=colors[0],
        label=r'$n_e a_0^2$. T: %d K' % T,
    )
    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        n_exc_vec * sys.a0**2,
        '-',
        color=colors[0],
        label=r'$n_{exc} a_0^2$',
    )

    ax[0].axvline(x=n_exp_vec[0] / surf_area * sys.a0**2,
                  color='m',
                  linestyle=':',
                  label='Experiment range')

    ax[0].axvline(
        x=n_exp_vec[-1] / surf_area * sys.a0**2,
        color='m',
        linestyle=':',
    )

    ax[0].set_xlim(1e-2, 1.1e1)
    ax[0].set_ylim(None, 4e1)

    lambda_th = sys.c_hbarc * sqrt(2 * pi * sys.beta / sys.m_p)
    ax[0].axvline(
        x=4 * sys.a0**2 / lambda_th**2,
        color='g',
        label='Degeneracy limit',
    )

    x_vec_top = ax[0].xaxis.get_majorticklocs()[2:-2]
    x_vec_vals = x_vec_top
    x_vec_vals = ['0.1', '1', '10']

    ax[0].set_xticks(x_vec_top)
    ax[0].set_xticklabels(x_vec_vals)
    """
    ax_top = ax[0].twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel(r'$n_\gamma$ (nm$^{-2}$)')
    """

    ax[0].legend(loc='upper left')

    fig.tight_layout()

    return 'density_result_%s' % plot_type


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
        color='w',
    )

    getattr(ax[0], plot_type)(
        n_exp_vec / surf_area * sys.a0**2,
        -cond_imag * 1e3,
        '^',
        color='w',
    )

    ax[0].errorbar(n_exp_vec / surf_area * sys.a0**2,
                   cond_real * 1e3,
                   yerr=cond_err_real * 1e3,
                   fmt='none',
                   capsize=5,
                   color='w')

    ax[0].errorbar(n_exp_vec / surf_area * sys.a0**2,
                   -cond_imag * 1e3,
                   yerr=cond_err_imag * 1e3,
                   fmt='none',
                   capsize=5,
                   color='w')

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
        label=r'Re$(\Delta\sigma)$. T: %d K' % T,
    )
    getattr(ax[0], plot_type)(
        n_vec * sys.a0**2,
        -imag(cond_vec) * 1e3,
        '-',
        color=colors[0],
        label=r'Im$(\Delta\sigma)$',
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
        color='w',
        linestyle='-',
    )

    lambda_th = sys.c_hbarc * sqrt(2 * pi * sys.beta / sys.m_p)
    ax[0].axvline(
        x=4 * sys.a0**2 / lambda_th**2,
        color='g',
        #label='Saha model limit',
    )

    ax[0].set_xlim(1 / surf_area * sys.a0**2, 60 / surf_area * sys.a0**2)
    ax[0].set_ylim(-cond_factor * 12.5, cond_factor * 5)

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'cond_fit_%s' % plot_type

plots_list = [pysys.argv[1:3]]

for p, l in plots_list:
    print('Calling %s' % p)
    filename = locals()[p](l)

    plt.savefig('plots/poster/symp2019/%s.png' % filename)
    plt.savefig('plots/poster/symp2019/%s.pdf' % filename)
    plt.savefig('plots/poster/symp2019/%s.eps' % filename)

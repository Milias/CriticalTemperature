from common import *
#plt.xkcd()
plt.style.use('dark_background')
#plt.rcParams['font.family'] = 'SF Pro Display'

N_k = 1 << 8

eb_cou = 0.193
err_eb_cou = 0.005

m_e, m_h, eps_r, T = 0.12, 0.3, 4.90185, 294  # K
sys = system_data(m_e, m_h, eps_r, T)

eps_r = sys.c_aEM * sqrt(2 * sys.m_p / eb_cou)
sys = system_data(m_e, m_h, eps_r, T)

figsize_default = (6.8, 5.3)
#figsize_mm = tuple(array([108.4, 132.5]) * 0.03937008) # box4, fig1
figsize_mm = tuple(array([108.4, 132.5]) * 0.03937008) # box4, fig2

surf_area = 326.4  # nm^2

n_x, n_y = 1, 1
fig = plt.figure(figsize=figsize_mm, dpi=300)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]
plot_func = {'log': ('semilogx', 'semilogx'), 'linear': ('plot', 'plot')}

a0 = sys.eps_r / sys.c_aEM * sys.c_hbarc / sys.m_p

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
    #ax[0].set_ylabel(r'$V(r / a_0)$ (eV)')

    cou_vec = array(time_func(plasmon_rpot_lwl_v, x_vec, 1e-8, sys))
    getattr(ax[0], plot_func[plot_type][0])(
        x_vec,
        cou_vec,
        '--',
        color='r',
        label=r'$\langle N_e\rangle$: %d' % 0,
    )

    for c, (i, (mu_e, mu_h, n_id)) in zip(
            colors, enumerate(zip(mu_e_vec, mu_h_vec, n_id_vec))):

        num_e = n_id * surf_area

        y_vec = array(time_func(plasmon_rpot_ht_v, x_vec, mu_e, mu_h, sys))
        getattr(ax[0], plot_func[plot_type][0])(
            x_vec,
            y_vec,
            '-',
            color=c,
            #label='$n_e$: $%.1f\cdot10^{-4}$ nm$^{-2}$' % (1e4 * n_id),
            label=r'$\langle N_e\rangle$: %.1f' % num_e,
        )

    sys_vec = array(time_func(plasmon_rpot_lwl_v, x_vec, sys.sys_ls, sys))
    getattr(ax[0], plot_func[plot_type][0])(
        x_vec,
        sys_vec,
        '--',
        color='w',
        label=r'$\langle N_e\rangle\rightarrow\infty$',
    )

    ax[0].set_ylim(-0.8, 0.1)
    ax[0].set_xlim(0, x_vec[-1])
    ax[0].axhline(y=0, color='w')

    x_vec_top = ax[0].xaxis.get_majorticklocs()[:-1]
    x_vec_vals = x_vec_top * a0
    x_vec_vals = ['%.1f' % v for v in x_vec_vals]

    #ax[0].yaxis.set_visible(False)
    ax[0].yaxis.set_ticks([0.0])
    ax[0].set_yticklabels(['0.0'])

    ax_top = ax[0].twiny()
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel('$r$ (nm)')

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'real_space_mb_potential_density'


def energy_level_mb_density(plot_type='log'):
    T_vec = linspace(100, 400, 5)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, T_vec.size)
    ]

    #ax[0].set_title('Exciton energy levels\nClassical Limit')
    ax[0].set_xlabel(r'$\langle N_e \rangle$')
    ax[0].set_ylabel(r'$\epsilon(\langle N_e \rangle)$ (meV)')

    sys = system_data(m_e, m_h, eps_r, 300)

    ax[0].axhline(
        y=sys.get_E_n(0.5) * 1e3,
        color='r',
        linestyle='--',
        label=r'$\langle N_e\rangle$: %d' % 0,
    )

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        sys = system_data(m_e, m_h, eps_r, T)
        mu_e_vec = linspace(-6, 0.0, 24) / sys.beta

        mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
        n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])

        y_vec = array(time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys))
        getattr(ax[0], plot_func[plot_type][0])(
            log10(n_id_vec * surf_area),
            y_vec * 1e3,
            '-',
            color=c,
            label='T: %.0f K' % sys.T,
        )
        getattr(ax[0], plot_func[plot_type][0])(
            log10(n_id_vec[-1] * surf_area),
            y_vec[-1] * 1e3,
            'o',
            color=c,
        )

    z_sys_lwl = time_func(plasmon_det_zero_lwl, N_k, sys.sys_ls, sys)

    ax[0].set_xlim(log10(1e-4 * surf_area), log10(1.1e-2 * surf_area))
    ax[0].axhline(
        y=z_sys_lwl * 1e3,
        color='w',
        linestyle='--',
        label=r'$\langle N_e\rangle\rightarrow\infty$',
    )

    ax[0].legend(loc=0)

    #ax[0].yaxis.set_visible(False)
    #ax[0].yaxis.set_ticks([])
    ax[0].yaxis.set_ticks([1e3 * z_sys_lwl, 1e3 * sys.get_E_n(0.5)])
    ax[0].set_yticklabels(['%d' % z for z in [1e3 * z_sys_lwl, 1e3 * sys.get_E_n(0.5)]])
    ax[0].yaxis.set_label_coords(-0.05,0.5)

    x_vec_top = ax[0].xaxis.get_majorticklocs()[1:-1]
    x_vec_vals = (10**x_vec_top / surf_area) * 1e3
    x_vec_vals = ['%.1f' % v for v in x_vec_vals]

    ax[0].set_xticklabels(['%.1f' % n for n in ((10**ax[0].xaxis.get_majorticklocs()))])

    ax_top = ax[0].twiny()
    ax_top.set_xlim(ax[0].get_xlim())
    ax_top.set_xticks(x_vec_top)
    ax_top.set_xticklabels(x_vec_vals)
    ax_top.set_xlabel(r'$n_e$ ($10^{-3}$ nm$^{-2}$)')

    fig.tight_layout()

    return 'energy_level_mb_density'

def energy_level_mb_limit_density(plot_type='linear'):
    T_vec = linspace(150, 350, 6)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, T_vec.size)
    ]

    #ax[0].set_title('Exciton energy levels\nClassical Limit')
    ax[0].set_xlabel(r'$n_e$ (nm$^{-2}$)')
    ax[0].set_ylabel(r'$\epsilon(n_e)$ (eV)')

    sys = system_data(m_e, m_h, eps_r, 300)

    ax[0].axhline(
        y=sys.get_E_n(0.5),
        color='r',
        linestyle='--',
        label='Coulomb limit',
    )

    for c, (i, T) in zip(colors, enumerate(T_vec)):
        sys = system_data(m_e, m_h, eps_r, T)
        mu_e_vec = linspace(-8, 0.0, 24) / sys.beta

        mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])
        n_id_vec = array([sys.density_ideal(mu_e) for mu_e in mu_e_vec])

        y_vec = array(time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys))
        line_vec = mu_e_vec + mu_h_vec

        getattr(ax[0], plot_func[plot_type][0])(
            n_id_vec,
            y_vec,
            '-',
            color=c,
            label='T: %.0f K' % sys.T,
        )

        getattr(ax[0], plot_func[plot_type][0])(
            n_id_vec,
            line_vec,
            '--',
            color=c,
            #label='T: %.0f K, $\mu_{\text{exc}}$' % sys.T,
        )

    ax[0].set_xlim(3e-6, 5e-3)
    ax[0].set_ylim(-0.2, -0.15)

    ax[0].legend(loc=0)

    fig.tight_layout()

    return 'energy_level_mb_limit_density'


plots_list = [
    ('real_space_mb_potential_density', 'linear'),
    #('energy_level_mb_density', 'linear'),
    #('energy_level_mb_limit_density', 'log'),
]

for p, l in plots_list:
    print('Calling %s' % p)
    filename = locals()[p](l)

    plt.savefig('plots/papers/drstp_symp2019/%s.png' % filename)
    plt.savefig('plots/papers/drstp_symp2019/%s.pdf' % filename)
    plt.savefig('plots/papers/drstp_symp2019/%s.eps' % filename)

    plt.cla()
    #plt.clf()
    #plt.close()


from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8, 5.3]))

#n_x, n_y = 1, 1
n_x, n_y = 1, 2
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]


def size_d_eff(sys):
    return sys.size_d * sys.eps_mat / sys.eps_r


def eta_func(eps_mat, eps_sol):
    return 0.5 * log((eps_mat + eps_sol) / (eps_mat - eps_sol))


def ke_k_pot(u, x, sys):
    if abs(sys.eps_mat - sys.eps_r) < 1e-8:
        pot = 1
    else:
        pot = 1 / tanh(u * sys.size_d * 0.5 / x +
                       eta_func(sys.eps_mat, sys.eps_r))

    return -special.j0(u) * pot / x / sys.eps_mat * sys.c_aEM * sys.c_hbarc


def qc_x_pot(x, sys):
    return -(3 * special.k0(2 * pi * x / sys.size_d) +
             2 * pi * x / sys.size_d * special.k1(2 * pi * x / sys.size_d)
             ) / sys.size_d / sys.eps_mat * sys.c_aEM * sys.c_hbarc


def qccougg_k_pot(u, x, sys):
    u_scaled = u * sys.size_d / x
    pot = 32 * pi**4 * (u_scaled - 1.0 +
                        exp(-u_scaled)) / (u_scaled *
                                           (u_scaled**2 + 4 * pi**2))**2

    return -special.j0(u) * pot / x / sys.eps_mat * sys.c_aEM * sys.c_hbarc


def qckegg_k_pot(u, x, sys):
    u_scaled = u * sys.size_d / x

    if abs(sys.eps_mat - sys.eps_r) < 1e-8:
        return qccougg_k_pot(u, x, sys)
    else:
        eta = eta_func(sys.eps_mat, sys.eps_r)

        if u_scaled > 700:
            pot = 32 * pi**4 * (u_scaled -
                                2 * sinh(eta)) / (u_scaled *
                                                  (u_scaled**2 + 4 * pi**2))**2
        else:
            pot = 32 * pi**4 * (
                u_scaled -
                (sinh(u_scaled + eta) * sinh(eta) + sinh(u_scaled + eta) *
                 sinh(eta) - 2 * sinh(eta) * sinh(eta)) /
                (sinh(u_scaled + 2 * eta))) / (u_scaled *
                                               (u_scaled**2 + 4 * pi**2))**2

    return -special.j0(u) * pot / x / sys.eps_mat * sys.c_aEM * sys.c_hbarc


def integ_pot(pot_func, x, sys):
    return sum([
        quad(
            pot_func,
            4 * n * pi,
            4 * (n + 1) * pi,
            limit=500,
            args=(x, sys),
        )[0] for n in arange(0.5e4)
    ])


N_x, N_eps = 32, 2

size_d = 1  # nm
eps_sol = 1
m_e, m_h, T = 0.27, 0.45, 294  # K

eps_vec = eps_sol / array([1.0, 0.2])

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)
sys_vec = [system_data(m_e, m_h, eps_sol, T, size_d, eps) for eps in eps_vec]


def plot_qccou_r_pot():
    sys_vec = [system_data(m_e, m_h, eps, T, size_d, eps) for eps in [1]]
    N_eps = len(sys_vec)
    x_vec = logspace(log10(1e-2), log10(10), N_x)

    integ_args = [(
        integ_pot,
        qccougg_k_pot,
        x * size_d_eff(sys) / sys.size_d,
        sys,
    ) for x, sys in itertools.product(x_vec, sys_vec)]

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    gg_vec = array(time_func(
        pool.starmap,
        time_func,
        integ_args,
    )).reshape((N_x, N_eps))

    pool.terminate()

    qc_vec = array([
        qc_x_pot(
            x_vec * size_d_eff(sys) / sys.size_d,
            sys,
        ) for sys in sys_vec
    ]).reshape((N_eps, N_x)).T
    full_vec = gg_vec + qc_vec

    cou_vec = -size_d / x_vec  #+ (size_d / x_vec)**3 - 9 * (size_d / x_vec)**5

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, len(sys_vec))
    ]

    for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
        ax[0].axvline(
            x=1,
            linestyle='--',
            color=c,
            dashes=(3., 5.),
            dash_capstyle='round',
            linewidth=0.8,
        )

        ax[0].semilogx(
            x_vec,
            cou_vec,
            color=c,
            linestyle='-',
            linewidth=0.7,
            label=r'$V^{C}(r)$',
        )

    for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
        energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / sys.eps_mat / sys.size_d

        ax[0].semilogx(
            x_vec,
            gg_vec[:, n_eps] / energy_norm,
            color=c,
            linestyle='--',
            dashes=(3., 5.),
            dash_capstyle='round',
            linewidth=0.9,
        )

        ax[0].semilogx(
            x_vec,
            full_vec[:, n_eps] / energy_norm,
            color=c,
            linestyle='-',
            linewidth=2.0,
            label=r'$V_{qc}^{C}(r)$',
        )

        ax[0].semilogx(
            x_vec,
            qc_vec[:, -1] / energy_norm,
            color=c,
            linestyle='--',
            dashes=(0.8, 4.),
            dash_capstyle='round',
            linewidth=1.0,
        )

    ax[0].set_yticks([0])
    ax[0].yaxis.set_label_coords(-0.01, 0.5)

    ax[0].set_ylabel(r'$V(r)$')
    ax[0].set_xlabel('$r / d$')

    ax[0].set_ylim(-10, 0)
    ax[0].set_xlim(1e-2, 1e1)

    ax[0].legend()

    plt.tight_layout()

    plt.savefig(
        '/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
        'r_qccou_pot_v4',
    )


def plot_qcke_r_pot():
    x_vec = logspace(log10(1e-2), log10(10), N_x)

    integ_args = [(
        integ_pot,
        qckegg_k_pot,
        x * size_d_eff(sys) / sys.size_d,
        sys,
    ) for x, sys in itertools.product(x_vec, sys_vec)]

    integ_args2 = [(
        integ_pot,
        qckegg_k_pot,
        x,
        sys,
    ) for x, sys in itertools.product(x_vec, sys_vec)]

    integ_ke_args = [(
        integ_pot,
        ke_k_pot,
        x * size_d_eff(sys) / sys.size_d,
        sys,
    ) for x, sys in itertools.product(x_vec, sys_vec)]

    integ_ke_args2 = [(
        integ_pot,
        ke_k_pot,
        x,
        sys,
    ) for x, sys in itertools.product(x_vec, sys_vec)]

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    gg_vec = array(time_func(
        pool.starmap,
        time_func,
        integ_args,
    )).reshape((N_x, N_eps))

    gg_vec2 = array(time_func(
        pool.starmap,
        time_func,
        integ_args2,
    )).reshape((N_x, N_eps))

    ke_vec = array(time_func(
        pool.starmap,
        time_func,
        integ_ke_args,
    )).reshape((N_x, N_eps))

    ke_vec2 = array(time_func(
        pool.starmap,
        time_func,
        integ_ke_args2,
    )).reshape((N_x, N_eps))

    pool.terminate()

    qc_vec = array([
        qc_x_pot(
            x_vec * size_d_eff(sys) / sys.size_d,
            sys,
        ) for sys in sys_vec
    ]).reshape((N_eps, N_x)).T

    qc_vec2 = array([qc_x_pot(
        x_vec,
        sys,
    ) for sys in sys_vec]).reshape((N_eps, N_x)).T

    full_vec = gg_vec + qc_vec
    full_vec2 = gg_vec2 + qc_vec2

    ax[0].axhline(y=0, color='k', linewidth=0.7)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, eps_vec.size)
    ]

    for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
        energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / sys.eps_mat / sys.size_d

        ax[0].semilogx(
            x_vec,
            ke_vec[:, n_eps] / energy_norm,
            color=c,
            linestyle='-',
            linewidth=0.7,
        )

    for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
        ax[0].axvline(
            x=size_d / size_d_eff(sys),
            linestyle='--',
            color=c,
            dashes=(3., 5.),
            dash_capstyle='round',
            linewidth=0.8,
        )

    for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
        energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / sys.eps_mat / sys.size_d

        ax[0].semilogx(
            x_vec,
            gg_vec[:, n_eps] / energy_norm,
            color=c,
            linestyle='--',
            dashes=(3., 5.),
            dash_capstyle='round',
            linewidth=0.9,
        )

        ax[0].semilogx(
            x_vec,
            full_vec[:, n_eps] / energy_norm,
            color=c,
            linestyle='-',
            linewidth=2.0,
            label=r'$d / d^*$: $%.1f$' % (size_d / size_d_eff(sys)),
        )

        ax[0].semilogx(
            x_vec,
            qc_vec[:, n_eps] / energy_norm,
            color=c,
            linestyle='--',
            dashes=(0.8, 4.),
            dash_capstyle='round',
            linewidth=1.0,
        )

    for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
        energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / sys.eps_mat / sys.size_d

        ax[1].semilogx(
            x_vec,
            ke_vec2[:, n_eps] / energy_norm,
            color=c,
            linestyle='-',
            linewidth=0.7,
        )

    ax[1].axvline(
        x=1,
        linestyle='--',
        color='m',
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=0.8,
    )

    for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
        energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / sys.eps_mat / sys.size_d

        ax[1].semilogx(
            x_vec / sys.size_d,
            gg_vec2[:, n_eps] / energy_norm,
            color=c,
            linestyle='--',
            dashes=(3., 5.),
            dash_capstyle='round',
            linewidth=0.9,
        )

        ax[1].semilogx(
            x_vec / sys.size_d,
            full_vec2[:, n_eps] / energy_norm,
            color=c,
            linestyle='-',
            linewidth=2.0,
            label=r'$d^* / d$: $%d$' % (size_d_eff(sys) / size_d),
        )

    ax[1].semilogx(
        x_vec / sys.size_d,
        qc_vec2[:, n_eps] / energy_norm,
        color='m',
        linestyle='--',
        dashes=(0.8, 4.),
        dash_capstyle='round',
        linewidth=1.0,
    )

    ax[0].set_yticks([0])
    ax[0].yaxis.set_label_coords(-0.01, 0.5)

    ax[0].set_ylabel(r'$V_{qc}^{RK}(r)$')
    ax[0].set_xlabel('$r / d^*$')

    ax[0].legend(title=r'$d / d^* = \epsilon_{sol} / \epsilon$')

    ax[0].set_ylim(-10, 0)
    ax[0].set_xlim(1e-2, 1e1 * 0.99)

    ax[1].set_yticks([])

    ax[1].set_xlabel('$r / d$')

    ax[1].legend(title=r'$d^* / d = \epsilon / \epsilon_{sol}$')

    ax[1].set_ylim(-10, 0)
    ax[1].set_xlim(1e-2 * 1.01, 1e1)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0)

    plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
                'r_qcke_pot_dual_v4')


#plot_qccou_r_pot()
plot_qcke_r_pot()
plt.show()
